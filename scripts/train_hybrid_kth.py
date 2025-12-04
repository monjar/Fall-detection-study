"""
Improved KTH-only fall detection training with multiple strategies:
1. Synthetic fall generation
2. Feature-based detection
3. Optional: train only on walking
4. Hybrid model combining all approaches
"""
from __future__ import annotations

import argparse
import json
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
from utils.pose_processing import PoseSequenceDataset, normalize_pose_sequence, pad_or_trim
from utils.synthetic_falls import (
    create_synthetic_fall_dataset,
    extract_fall_features,
    is_fall_by_rules,
)


class SyntheticFallDataset(Dataset):
    """Dataset combining normal KTH sequences and synthetic falls."""
    
    def __init__(self, normal_sequences: list, fall_sequences: list):
        """
        Args:
            normal_sequences: List of (T, J, 2) normal poses, label=0
            fall_sequences: List of (T, J, 2) fall poses, label=1
        """
        self.sequences = normal_sequences + fall_sequences
        self.labels = [0] * len(normal_sequences) + [1] * len(fall_sequences)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert to tensor: (2, T, J)
        tensor = torch.from_numpy(seq.transpose(2, 0, 1)).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return tensor, label_tensor


class HybridFallDetectorLSTM(nn.Module):
    """
    Hybrid model with LSTM temporal modeling:
    1. Autoencoder for spatial features
    2. LSTM for temporal dynamics
    3. Binary fall classifier
    """
    
    def __init__(self, pretrained_autoencoder: Optional[PoseAutoencoder] = None,
                 lstm_hidden: int = 128, lstm_layers: int = 2):
        super().__init__()
        
        if pretrained_autoencoder is not None:
            self.autoencoder = pretrained_autoencoder
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
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),  # Binary: Normal vs Fall
        )
        
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: (B, 2, T, J) pose tensor
            return_features: If True, return intermediate features
            
        Returns:
            If return_features=False: (reconstructed, logits)
            If return_features=True: (reconstructed, logits, latent, lstm_out, recon_error)
        """
        B, C, T, J = x.shape
        
        # Process each time step through autoencoder
        reconstructed_frames = []
        latent_frames = []
        
        for t in range(T):
            frame = x[:, :, t:t+1, :]  # (B, 2, 1, J)
            recon_frame, latent_frame = self.autoencoder(frame)
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


class HybridFallDetector(HybridFallDetectorLSTM):
    """Alias for backward compatibility."""
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Improved KTH-only fall detection training")
    
    # Data
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/pose_keypoints/npz"),
        help="Directory with KTH NPZ files",
    )
    parser.add_argument(
        "--activity-filter",
        type=str,
        default=None,
        help="Train only on specific activity (e.g., 'walking', 'jogging')",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=64,
        help="Sequence length",
    )
    
    # Synthetic falls
    parser.add_argument(
        "--num-synthetic-falls",
        type=int,
        default=5,
        help="Number of synthetic falls per normal sequence (recommended: 5-10)",
    )
    parser.add_argument(
        "--fall-generation-methods",
        nargs="+",
        default=None,  # None = use all 8 types
        help="Methods for generating synthetic falls. Options: forward, backward, "
             "sideways_left, sideways_right, collapse, stumble, rotation, standing. "
             "If not specified, uses all 8 types.",
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
        default=1.0,
        help="Weight for reconstruction loss",
    )
    parser.add_argument(
        "--class-weight",
        type=float,
        default=1.0,
        help="Weight for classification loss",
    )
    
    # Model
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Path to pretrained autoencoder checkpoint",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder weights during training",
    )
    
    # Output
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/hybrid_fall_detector.pt"),
        help="Path to save checkpoint",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    
    return parser.parse_args()


def load_kth_sequences(data_dir: Path, 
                       activity_filter: Optional[str] = None,
                       sequence_length: int = 64) -> list:
    """Load and preprocess KTH sequences."""
    
    npz_files = sorted(list(data_dir.glob("*.npz")))
    
    if activity_filter:
        npz_files = [f for f in npz_files if activity_filter in f.name]
        print(f"Filtered to {len(npz_files)} files with activity: {activity_filter}")
    
    sequences = []
    
    print(f"Loading {len(npz_files)} NPZ files...")
    for npz_file in tqdm(npz_files):
        try:
            data = np.load(npz_file, allow_pickle=True)
            # Try different possible keys
            if 'poses' in data:
                keypoints = data['poses']
            elif 'keypoints' in data:
                keypoints = data['keypoints']
            elif 'pose_sequence' in data:
                keypoints = data['pose_sequence']
            else:
                print(f"Unknown keys in {npz_file}: {list(data.keys())}")
                continue
            
            # Normalize and pad/trim
            keypoints = normalize_pose_sequence(keypoints)
            keypoints = pad_or_trim(keypoints, sequence_length)
            
            if not np.isnan(keypoints).all():
                sequences.append(keypoints)
                
        except Exception as e:
            print(f"Error loading {npz_file.name}: {e}")
            continue
    
    print(f"✅ Loaded {len(sequences)} sequences")
    return sequences


def train_epoch(model: HybridFallDetector,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                recon_weight: float,
                class_weight: float) -> dict:
    """Train for one epoch."""
    
    model.train()
    
    total_loss = 0
    total_recon_loss = 0
    total_class_loss = 0
    correct = 0
    total = 0
    
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (sequences, labels) in enumerate(dataloader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, logits = model(sequences)
        
        # Compute losses
        recon_loss = recon_criterion(reconstructed, sequences)
        class_loss = class_criterion(logits, labels)
        
        # Combined loss
        loss = recon_weight * recon_loss + class_weight * class_loss
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_class_loss += class_loss.item()
        
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return {
        'loss': total_loss / len(dataloader),
        'recon_loss': total_recon_loss / len(dataloader),
        'class_loss': total_class_loss / len(dataloader),
        'accuracy': correct / total,
    }


def validate(model: HybridFallDetector,
            dataloader: DataLoader,
            device: torch.device,
            recon_weight: float,
            class_weight: float) -> dict:
    """Validate the model."""
    
    model.eval()
    
    total_loss = 0
    total_recon_loss = 0
    total_class_loss = 0
    correct = 0
    total = 0
    
    # Per-class accuracy
    correct_per_class = {0: 0, 1: 0}
    total_per_class = {0: 0, 1: 0}
    
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            reconstructed, logits = model(sequences)
            
            # Compute losses
            recon_loss = recon_criterion(reconstructed, sequences)
            class_loss = class_criterion(logits, labels)
            loss = recon_weight * recon_loss + class_weight * class_loss
            
            # Statistics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for label in [0, 1]:
                mask = labels == label
                if mask.sum() > 0:
                    correct_per_class[label] += (predicted[mask] == label).sum().item()
                    total_per_class[label] += mask.sum().item()
    
    accuracy_normal = correct_per_class[0] / total_per_class[0] if total_per_class[0] > 0 else 0
    accuracy_fall = correct_per_class[1] / total_per_class[1] if total_per_class[1] > 0 else 0
    
    return {
        'loss': total_loss / len(dataloader),
        'recon_loss': total_recon_loss / len(dataloader),
        'class_loss': total_class_loss / len(dataloader),
        'accuracy': correct / total,
        'accuracy_normal': accuracy_normal,
        'accuracy_fall': accuracy_fall,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    print("=" * 70)
    print("IMPROVED KTH-ONLY FALL DETECTION TRAINING")
    print("=" * 70)
    
    # Load KTH sequences
    print(f"\n📂 Loading KTH sequences from: {args.data_dir}")
    if args.activity_filter:
        print(f"   Activity filter: {args.activity_filter}")
    
    normal_sequences = load_kth_sequences(
        args.data_dir,
        activity_filter=args.activity_filter,
        sequence_length=args.sequence_length
    )
    
    if len(normal_sequences) == 0:
        print("❌ No sequences loaded!")
        return
    
    # Generate synthetic falls
    print(f"\n🔄 Generating synthetic falls...")
    print(f"   Methods: {args.fall_generation_methods}")
    print(f"   Num per sequence: {args.num_synthetic_falls}")
    
    fall_sequences, metadata = create_synthetic_fall_dataset(
        normal_sequences,
        num_falls_per_sequence=args.num_synthetic_falls,
        methods=args.fall_generation_methods
    )
    
    print(f"✅ Generated {len(fall_sequences)} synthetic falls")
    
    # Create dataset
    print(f"\n📊 Creating dataset...")
    full_dataset = SyntheticFallDataset(normal_sequences, fall_sequences)
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"   Train samples: {train_size}")
    print(f"   Val samples: {val_size}")
    print(f"   Normal: {len(normal_sequences)}, Falls: {len(fall_sequences)}")
    
    # Create model
    print(f"\n🔧 Creating model...")
    
    pretrained_autoencoder = None
    if args.pretrained_checkpoint and args.pretrained_checkpoint.exists():
        print(f"   Loading pretrained autoencoder: {args.pretrained_checkpoint}")
        pretrained_autoencoder = PoseAutoencoder()
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            pretrained_autoencoder.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            pretrained_autoencoder.load_state_dict(checkpoint['state_dict'])
        else:
            pretrained_autoencoder.load_state_dict(checkpoint)
        
        print(f"   ✅ Loaded pretrained weights")
    
    model = HybridFallDetector(pretrained_autoencoder)
    
    if args.freeze_encoder:
        print(f"   🔒 Freezing encoder weights")
        for param in model.autoencoder.encoder.parameters():
            param.requires_grad = False
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Recon weight: {args.recon_weight}")
    print(f"   Class weight: {args.class_weight}")
    print(f"   Device: {device}")
    
    best_val_loss = float('inf')
    best_val_accuracy = 0
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_accuracy_normal': [],
        'val_accuracy_fall': [],
    }
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            args.recon_weight, args.class_weight
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, device,
            args.recon_weight, args.class_weight
        )
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_accuracy_normal'].append(val_metrics['accuracy_normal'])
        history['val_accuracy_fall'].append(val_metrics['accuracy_fall'])
        
        # Print progress
        print(f"Epoch {epoch:3d}/{args.epochs}: "
              f"train_loss={train_metrics['loss']:.4f} "
              f"train_acc={train_metrics['accuracy']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} "
              f"(N:{val_metrics['accuracy_normal']:.4f} F:{val_metrics['accuracy_fall']:.4f})")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_accuracy_normal': val_metrics['accuracy_normal'],
                'val_accuracy_fall': val_metrics['accuracy_fall'],
                'args': vars(args),
            }, args.checkpoint)
            
            print(f"   💾 Saved best model (val_acc={best_val_accuracy:.4f})")
        
        # Periodic checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = args.checkpoint.parent / f"hybrid_fall_detector_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
            }, checkpoint_path)
    
    # Save training history
    history_path = args.checkpoint.parent / "hybrid_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"✅ Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"✅ Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Model saved to: {args.checkpoint}")
    print(f"📊 History saved to: {history_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
