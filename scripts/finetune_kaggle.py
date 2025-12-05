"""Fine-tune the pose autoencoder on Kaggle Fall dataset with supervised learning."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_model import PoseAutoencoder
from utils.pose_processing import normalize_pose_sequence, pad_or_trim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune model on Kaggle dataset")
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        default=Path("data/KaggleRealDataset"),
        help="Path to Kaggle dataset",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=Path("models/checkpoints/pose_autoencoder.pt"),
        help="Path to pretrained KTH model",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("models/checkpoints/pose_autoencoder_kaggle_finetuned.pt"),
        help="Path to save fine-tuned model",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=64,
        help="Sequence length for model input",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of fine-tuning epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (lower for fine-tuning)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Test split ratio",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Maximum samples per class (for quick testing)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train from scratch instead of fine-tuning",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder weights during fine-tuning",
    )
    return parser.parse_args()


def load_csv_to_pose_array(csv_path: Path) -> np.ndarray:
    """Load CSV file and convert to (T, J, 2) pose array."""
    df = pd.read_csv(csv_path)
    
    frames = sorted(df['Frame'].unique())
    keypoints = [
        'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
        'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
        'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
        'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
    ]
    
    num_frames = len(frames)
    num_joints = 17
    
    pose_array = np.zeros((num_frames, num_joints, 2), dtype=np.float32)
    
    for frame_idx, frame in enumerate(frames):
        frame_data = df[df['Frame'] == frame]
        for joint_idx, keypoint in enumerate(keypoints):
            kp_data = frame_data[frame_data['Keypoint'] == keypoint]
            if not kp_data.empty:
                pose_array[frame_idx, joint_idx, 0] = kp_data['X'].values[0]
                pose_array[frame_idx, joint_idx, 1] = kp_data['Y'].values[0]
    
    # Pad to 18 joints to match model expectations
    pose_array_18 = np.zeros((num_frames, 18, 2), dtype=np.float32)
    pose_array_18[:, :17, :] = pose_array
    
    return pose_array_18


class KaggleFallDataset(Dataset):
    """Dataset for Kaggle Fall detection with sliding windows."""
    
    def __init__(
        self,
        file_paths: List[Path],
        labels: List[int],
        sequence_length: int = 64,
        normalize: bool = True,
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Build index of all windows
        self.samples = []
        print("Building dataset index...")
        for file_path, label in tqdm(zip(file_paths, labels), total=len(file_paths)):
            try:
                pose_array = load_csv_to_pose_array(file_path)
                
                if self.normalize:
                    pose_array = normalize_pose_sequence(pose_array)
                
                # Pad or trim to sequence length
                pose_array = pad_or_trim(pose_array, sequence_length)
                
                # Convert to tensor: (2, T, J)
                tensor = torch.from_numpy(pose_array.transpose(2, 0, 1)).float()
                self.samples.append((tensor, label))
                
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                continue
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx]


class FallClassifier(nn.Module):
    """Wrapper around PoseAutoencoder with classification head."""
    
    def __init__(self, pretrained_autoencoder: PoseAutoencoder = None):
        super().__init__()
        
        if pretrained_autoencoder is not None:
            self.autoencoder = pretrained_autoencoder
        else:
            self.autoencoder = PoseAutoencoder()
        
        # Classification head on top of latent space
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 2),  # Binary classification: Fall vs No Fall
        )
    
    def forward(self, x: torch.Tensor, return_reconstruction: bool = False):
        """Forward pass.
        
        Args:
            x: Input pose sequence
            return_reconstruction: If True, also return reconstruction
        
        Returns:
            logits: Classification logits (batch_size, 2)
            recon: Reconstruction (if return_reconstruction=True)
        """
        recon, latent = self.autoencoder(x)
        logits = self.classifier(latent)
        
        if return_reconstruction:
            return logits, recon
        return logits


def prepare_datasets(args: argparse.Namespace):
    """Load and split Kaggle dataset."""
    
    fall_dir = args.kaggle_dir / "Fall" / "Keypoints_CSV"
    no_fall_dir = args.kaggle_dir / "No_Fall" / "Keypoints_CSV"
    
    fall_files = sorted(list(fall_dir.glob("*.csv")))
    no_fall_files = sorted(list(no_fall_dir.glob("*.csv")))
    
    if args.max_samples_per_class:
        fall_files = fall_files[:args.max_samples_per_class]
        no_fall_files = no_fall_files[:args.max_samples_per_class]
    
    print(f"\nDataset size:")
    print(f"  Fall: {len(fall_files)}")
    print(f"  No Fall: {len(no_fall_files)}")
    
    # Combine and create labels
    all_files = fall_files + no_fall_files
    all_labels = [1] * len(fall_files) + [0] * len(no_fall_files)
    
    # Split into train/val/test
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=args.test_split, stratify=all_labels, random_state=42
    )
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=args.val_split, stratify=train_labels, random_state=42
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_files)} (Fall: {sum(train_labels)}, No Fall: {len(train_labels)-sum(train_labels)})")
    print(f"  Val:   {len(val_files)} (Fall: {sum(val_labels)}, No Fall: {len(val_labels)-sum(val_labels)})")
    print(f"  Test:  {len(test_files)} (Fall: {sum(test_labels)}, No Fall: {len(test_labels)-sum(test_labels)})")
    
    # Create datasets
    train_dataset = KaggleFallDataset(train_files, train_labels, args.sequence_length)
    val_dataset = KaggleFallDataset(val_files, val_labels, args.sequence_length)
    test_dataset = KaggleFallDataset(test_files, test_labels, args.sequence_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(
    model: FallClassifier,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_reconstruction: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_recon_loss = 0.0
    correct = 0
    total = 0
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()
    
    for batch_poses, batch_labels in tqdm(train_loader, desc="Training"):
        batch_poses = batch_poses.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        
        if use_reconstruction:
            logits, recon = model(batch_poses, return_reconstruction=True)
            cls_loss = criterion_cls(logits, batch_labels)
            recon_loss = criterion_recon(recon, batch_poses)
            loss = cls_loss + 0.1 * recon_loss  # Weighted combination
            total_recon_loss += recon_loss.item()
        else:
            logits = model(batch_poses)
            loss = criterion_cls(logits, batch_labels)
            cls_loss = loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        
        _, predicted = torch.max(logits, 1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)
    
    metrics = {
        'loss': total_loss / len(train_loader),
        'cls_loss': total_cls_loss / len(train_loader),
        'accuracy': correct / total,
    }
    
    if use_reconstruction:
        metrics['recon_loss'] = total_recon_loss / len(train_loader)
    
    return metrics


def validate(
    model: FallClassifier,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_poses, batch_labels in tqdm(val_loader, desc="Validation"):
            batch_poses = batch_poses.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = model(batch_poses)
            loss = criterion(logits, batch_labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Compute per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    fall_mask = all_labels == 1
    no_fall_mask = all_labels == 0
    
    fall_acc = (all_preds[fall_mask] == all_labels[fall_mask]).mean() if fall_mask.sum() > 0 else 0.0
    no_fall_acc = (all_preds[no_fall_mask] == all_labels[no_fall_mask]).mean() if no_fall_mask.sum() > 0 else 0.0
    
    metrics = {
        'loss': total_loss / len(val_loader),
        'accuracy': correct / total,
        'fall_accuracy': fall_acc,
        'no_fall_accuracy': no_fall_acc,
    }
    
    return metrics


def finetune(args: argparse.Namespace):
    """Main fine-tuning function."""
    
    device = torch.device(args.device)
    
    print("=" * 70)
    print("FINE-TUNING ON KAGGLE FALL DATASET")
    print("=" * 70)
    
    # Load pretrained model
    if args.from_scratch:
        print("\n⚠️  Training from scratch (not using pretrained weights)")
        model = FallClassifier()
    else:
        print(f"\n📥 Loading pretrained model from: {args.pretrained_checkpoint}")
        if not args.pretrained_checkpoint.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_checkpoint}")
        
        pretrained = PoseAutoencoder()
        state = torch.load(args.pretrained_checkpoint, map_location=device)
        pretrained.load_state_dict(state["state_dict"])
        
        model = FallClassifier(pretrained_autoencoder=pretrained)
        print("✅ Loaded pretrained autoencoder")
    
    model = model.to(device)
    
    # Optionally freeze encoder
    if args.freeze_encoder and not args.from_scratch:
        print("🔒 Freezing encoder weights")
        for param in model.autoencoder.encoder.parameters():
            param.requires_grad = False
        for param in model.autoencoder.latent_head.parameters():
            param.requires_grad = False
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Prepare datasets
    print("\n" + "=" * 70)
    print("PREPARING DATASETS")
    print("=" * 70)
    train_loader, val_loader, test_loader = prepare_datasets(args)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print(f"  Use reconstruction loss: {not args.freeze_encoder}")
    print("=" * 70)
    
    best_val_acc = 0.0
    best_epoch = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*70)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            use_reconstruction=(not args.freeze_encoder and not args.from_scratch)
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['accuracy'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"        Fall Acc: {val_metrics['fall_accuracy']:.4f}, No-Fall Acc: {val_metrics['no_fall_accuracy']:.4f}")
        print(f"        LR: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            
            args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'train_accuracy': train_metrics['accuracy'],
            }, args.output_checkpoint)
            print(f"✅ Saved best model (val_acc: {best_val_acc:.4f})")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': current_lr,
        })
    
    # Test on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    # Load best model
    best_state = torch.load(args.output_checkpoint, map_location=device)
    model.load_state_dict(best_state['model_state_dict'])
    
    test_metrics = validate(model, test_loader, device)
    
    print(f"\n🎯 Test Results:")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"   Fall Accuracy: {test_metrics['fall_accuracy']:.4f}")
    print(f"   No-Fall Accuracy: {test_metrics['no_fall_accuracy']:.4f}")
    print(f"\n✅ Best model from epoch {best_epoch}")
    print(f"   Validation Accuracy: {best_val_acc:.4f}")
    
    # Save training history
    history_path = args.output_checkpoint.parent / "kaggle_finetuning_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n💾 Model saved to: {args.output_checkpoint}")
    print(f"📊 History saved to: {history_path}")
    print("=" * 70)


def main():
    args = parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
