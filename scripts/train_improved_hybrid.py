"""
Improved Hybrid Model Training - Implements improvements #1, #2, #3:
1. Train on ALL KTH activities (not just walking)
2. Generate 7+ diverse fall types
3. Use LSTM for temporal modeling

This script significantly improves upon the original hybrid approach.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_model import PoseAutoencoder
from utils.pose_processing import normalize_pose_sequence, pad_or_trim
from utils.synthetic_falls import create_synthetic_fall_dataset


class ImprovedFallDataset(Dataset):
    """Dataset combining ALL KTH activities with diverse synthetic falls."""
    
    def __init__(self, normal_sequences: list, fall_sequences: list):
        """
        Args:
            normal_sequences: List of (T, J, 2) normal poses, label=0
            fall_sequences: List of (T, J, 2) fall poses, label=1
        """
        self.sequences = normal_sequences + fall_sequences
        self.labels = [0] * len(normal_sequences) + [1] * len(fall_sequences)
        
        print(f"Dataset created: {len(normal_sequences)} normal + {len(fall_sequences)} falls = {len(self)} total")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert to tensor: (2, T, J)
        tensor = torch.from_numpy(seq.transpose(2, 0, 1)).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return tensor, label_tensor


class ImprovedHybridDetectorLSTM(nn.Module):
    """
    Improved hybrid model with LSTM temporal modeling.
    Combines spatial features from autoencoder with temporal dynamics from LSTM.
    """
    
    def __init__(self, pretrained_autoencoder: Optional[PoseAutoencoder] = None,
                 lstm_hidden: int = 128, lstm_layers: int = 2):
        super().__init__()
        
        if pretrained_autoencoder is not None:
            self.autoencoder = pretrained_autoencoder
            # Freeze autoencoder initially (we'll fine-tune later)
            for param in self.autoencoder.parameters():
                param.requires_grad = False
        else:
            self.autoencoder = PoseAutoencoder()
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,  # Latent dimension from autoencoder
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head on LSTM output
        lstm_output_size = lstm_hidden * 2  # Bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 2),  # Binary: Normal vs Fall
        )
        
    def unfreeze_autoencoder(self):
        """Unfreeze autoencoder for fine-tuning."""
        for param in self.autoencoder.parameters():
            param.requires_grad = True
        
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: (B, 2, T, J) pose tensor
            return_features: If True, return intermediate features
            
        Returns:
            If return_features=False: (reconstructed, logits)
            If return_features=True: (reconstructed, logits, latent_sequence, lstm_out, recon_error)
        """
        B, C, T, J = x.shape
        
        # Process each time step through autoencoder
        reconstructed_frames = []
        latent_frames = []
        
        for t in range(T):
            frame = x[:, :, t:t+1, :]  # (B, 2, 1, J)
            recon_frame, latent_frame = self.autoencoder(frame)
            # Squeeze the time dimension from autoencoder output
            recon_frame = recon_frame.squeeze(2)  # (B, 2, J)
            reconstructed_frames.append(recon_frame)
            latent_frames.append(latent_frame)
        
        # Stack reconstructed frames
        reconstructed = torch.stack(reconstructed_frames, dim=2)  # (B, 2, T, J)
        latent_sequence = torch.stack(latent_frames, dim=1)  # (B, T, 128)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(latent_sequence)  # (B, T, lstm_hidden*2)
        
        # Use last LSTM output for classification
        lstm_final = lstm_out[:, -1, :]  # (B, lstm_hidden*2)
        logits = self.classifier(lstm_final)
        
        if return_features:
            recon_error = torch.mean((x - reconstructed) ** 2, dim=(1, 2, 3))
            return reconstructed, logits, latent_sequence, lstm_out, recon_error
        
        return reconstructed, logits


def load_all_kth_sequences(data_dir: Path, sequence_length: int = 64,
                           max_sequences: Optional[int] = None) -> list:
    """
    Load ALL KTH sequences (all activities, all people).
    
    Args:
        data_dir: Path to NPZ files
        sequence_length: Target sequence length
        max_sequences: Optional limit on number of sequences
        
    Returns:
        List of (T, J, 2) normalized sequences
    """
    npz_files = sorted(data_dir.glob("*.npz"))
    
    if max_sequences:
        npz_files = npz_files[:max_sequences]
    
    sequences = []
    activity_counts = {}
    
    print(f"Loading sequences from {len(npz_files)} NPZ files...")
    
    for npz_file in tqdm(npz_files, desc="Loading KTH sequences"):
        try:
            data = np.load(npz_file)
            
            # Try different keys
            if 'poses' in data:
                pose_seq = data['poses']
            elif 'keypoints' in data:
                pose_seq = data['keypoints']
            elif 'pose_sequence' in data:
                pose_seq = data['pose_sequence']
            else:
                continue
            
            # Extract activity from filename
            filename = npz_file.stem
            activity = filename.split('_')[1]  # e.g., person01_walking_d1 -> walking
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
            
            # Normalize and pad/trim
            pose_seq = normalize_pose_sequence(pose_seq)
            pose_seq = pad_or_trim(pose_seq, sequence_length)
            
            sequences.append(pose_seq)
            
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue
    
    print(f"\nLoaded {len(sequences)} sequences:")
    for activity, count in sorted(activity_counts.items()):
        print(f"  {activity}: {count}")
    
    return sequences


def train_epoch(model: ImprovedHybridDetectorLSTM,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: str,
                recon_weight: float = 1.0,
                class_weight: float = 1.0) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_class_loss = 0.0
    correct = 0
    total = 0
    
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, logits = model(inputs)
        
        # Compute losses
        recon_loss = criterion_recon(reconstructed, inputs)
        class_loss = criterion_class(logits, labels)
        
        # Combined loss
        loss = recon_weight * recon_loss + class_weight * class_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_class_loss += class_loss.item()
        
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'class': f'{class_loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'recon_loss': total_recon_loss / len(dataloader),
        'class_loss': total_class_loss / len(dataloader),
        'accuracy': 100 * correct / total
    }


def validate(model: ImprovedHybridDetectorLSTM,
             dataloader: DataLoader,
             device: str,
             recon_weight: float = 1.0,
             class_weight: float = 1.0) -> dict:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_class_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class metrics
    correct_per_class = {0: 0, 1: 0}
    total_per_class = {0: 0, 1: 0}
    
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            reconstructed, logits = model(inputs)
            
            # Compute losses
            recon_loss = criterion_recon(reconstructed, inputs)
            class_loss = criterion_class(logits, labels)
            loss = recon_weight * recon_loss + class_weight * class_loss
            
            # Statistics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for label in [0, 1]:
                mask = labels == label
                if mask.sum() > 0:
                    total_per_class[label] += mask.sum().item()
                    correct_per_class[label] += (predicted[mask] == label).sum().item()
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for label in [0, 1]:
        if total_per_class[label] > 0:
            class_accuracies[f'class_{label}_acc'] = 100 * correct_per_class[label] / total_per_class[label]
        else:
            class_accuracies[f'class_{label}_acc'] = 0.0
    
    return {
        'loss': total_loss / len(dataloader),
        'recon_loss': total_recon_loss / len(dataloader),
        'class_loss': total_class_loss / len(dataloader),
        'accuracy': 100 * correct / total,
        **class_accuracies
    }


def main():
    parser = argparse.ArgumentParser(description="Improved Hybrid Fall Detection Training")
    
    # Data
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/pose_keypoints/npz"),
        help="Directory with KTH NPZ files",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=64,
        help="Sequence length",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Maximum number of KTH sequences to load (None = all)",
    )
    
    # Synthetic falls (IMPROVEMENT #2: Multiple fall types)
    parser.add_argument(
        "--num-synthetic-falls",
        type=int,
        default=5,
        help="Number of synthetic falls per normal sequence (default: 5)",
    )
    parser.add_argument(
        "--fall-types",
        nargs="+",
        default=None,
        help="Fall types to generate. If None, uses all 8 types: "
             "forward, backward, sideways_left, sideways_right, collapse, stumble, rotation, standing",
    )
    
    # Model (IMPROVEMENT #3: LSTM)
    parser.add_argument(
        "--lstm-hidden",
        type=int,
        default=128,
        help="LSTM hidden size",
    )
    parser.add_argument(
        "--lstm-layers",
        type=int,
        default=2,
        help="Number of LSTM layers",
    )
    parser.add_argument(
        "--pretrained-autoencoder",
        type=Path,
        default=Path("models/pose_autoencoder.pt"),
        help="Path to pretrained autoencoder",
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    
    # Loss weights
    parser.add_argument(
        "--recon-weight",
        type=float,
        default=0.5,
        help="Weight for reconstruction loss (lower than original to focus on classification)",
    )
    parser.add_argument(
        "--class-weight",
        type=float,
        default=1.0,
        help="Weight for classification loss",
    )
    parser.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=20,
        help="Epoch to unfreeze autoencoder for fine-tuning",
    )
    
    # Output
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("models/improved_hybrid_detector.pt"),
        help="Path to save trained model",
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        default=Path("results/improved_hybrid_training_log.json"),
        help="Path to save training log",
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 80)
    print("IMPROVED HYBRID FALL DETECTION TRAINING")
    print("Improvements:")
    print("  #1: Train on ALL KTH activities (not just walking)")
    print("  #2: Generate diverse fall types (7+ types)")
    print("  #3: Use LSTM for temporal modeling")
    print("=" * 80)
    print()
    
    # IMPROVEMENT #1: Load ALL KTH sequences (all activities)
    print("STEP 1: Loading ALL KTH sequences...")
    all_normal_sequences = load_all_kth_sequences(
        args.data_dir,
        sequence_length=args.sequence_length,
        max_sequences=args.max_sequences
    )
    print(f"Loaded {len(all_normal_sequences)} normal sequences from all activities")
    print()
    
    # IMPROVEMENT #2: Generate diverse synthetic falls
    print(f"STEP 2: Generating {args.num_synthetic_falls} synthetic falls per sequence...")
    print(f"Fall types: {args.fall_types if args.fall_types else 'all 8 types'}")
    fall_sequences, metadata_list = create_synthetic_fall_dataset(
        all_normal_sequences,
        num_falls_per_sequence=args.num_synthetic_falls,
        methods=args.fall_types
    )
    print(f"Generated {len(fall_sequences)} synthetic falls")
    
    # Count fall types
    fall_type_counts = {}
    for meta in metadata_list:
        method = meta.get('method', 'unknown')
        fall_type_counts[method] = fall_type_counts.get(method, 0) + 1
    print("Fall type distribution:")
    for fall_type, count in sorted(fall_type_counts.items()):
        print(f"  {fall_type}: {count}")
    print()
    
    # Split into train/val
    print("STEP 3: Creating train/val split...")
    num_normal = len(all_normal_sequences)
    num_falls = len(fall_sequences)
    
    # Calculate split indices
    val_normal = int(num_normal * args.val_split)
    val_falls = int(num_falls * args.val_split)
    
    # Shuffle indices
    normal_indices = list(range(num_normal))
    fall_indices = list(range(num_falls))
    random.shuffle(normal_indices)
    random.shuffle(fall_indices)
    
    # Split
    train_normal = [all_normal_sequences[i] for i in normal_indices[val_normal:]]
    val_normal = [all_normal_sequences[i] for i in normal_indices[:val_normal]]
    train_falls = [fall_sequences[i] for i in fall_indices[val_falls:]]
    val_falls = [fall_sequences[i] for i in fall_indices[:val_falls]]
    
    print(f"Train: {len(train_normal)} normal + {len(train_falls)} falls = {len(train_normal) + len(train_falls)} total")
    print(f"Val: {len(val_normal)} normal + {len(val_falls)} falls = {len(val_normal) + len(val_falls)} total")
    print()
    
    # Create datasets
    train_dataset = ImprovedFallDataset(train_normal, train_falls)
    val_dataset = ImprovedFallDataset(val_normal, val_falls)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # IMPROVEMENT #3: Create LSTM-based model
    print("STEP 4: Creating improved hybrid model with LSTM...")
    
    # Load pretrained autoencoder
    pretrained_autoencoder = None
    if args.pretrained_autoencoder.exists():
        print(f"Loading pretrained autoencoder from {args.pretrained_autoencoder}")
        pretrained_autoencoder = PoseAutoencoder()
        checkpoint = torch.load(args.pretrained_autoencoder, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            pretrained_autoencoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            pretrained_autoencoder.load_state_dict(checkpoint)
        print("Autoencoder loaded and will be frozen initially")
    else:
        print("No pretrained autoencoder found, training from scratch")
    
    model = ImprovedHybridDetectorLSTM(
        pretrained_autoencoder=pretrained_autoencoder,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers
    )
    model = model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("STEP 5: Training...")
    training_log = []
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Unfreeze autoencoder after certain epoch
        if epoch == args.unfreeze_epoch and pretrained_autoencoder is not None:
            print(f"Unfreezing autoencoder for fine-tuning...")
            model.unfreeze_autoencoder()
            # Re-create optimizer with all parameters
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)  # Lower LR for fine-tuning
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device,
            recon_weight=args.recon_weight,
            class_weight=args.class_weight
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, args.device,
            recon_weight=args.recon_weight,
            class_weight=args.class_weight
        )
        
        # Log
        log_entry = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics
        }
        training_log.append(log_entry)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        print(f"        Normal Acc: {val_metrics.get('class_0_acc', 0):.2f}%, "
              f"Fall Acc: {val_metrics.get('class_1_acc', 0):.2f}%")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            print(f"New best validation accuracy: {best_val_acc:.2f}%")
            
            args.output_model.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, args.output_model)
            print(f"Model saved to {args.output_model}")
    
    # Save training log
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_log, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"\nTraining log saved to {args.output_log}")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    main()
