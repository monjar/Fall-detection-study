"""Training / evaluation script for the pose autoencoder anomaly model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.cnn_model import PoseAutoencoder
from utils.pose_processing import PoseSequenceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pose anomaly detector")
    parser.add_argument("--pose-dir", type=Path, default=Path("data/pose_keypoints/npz"))
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate (default: 5e-3, increased from 1e-3)")
    parser.add_argument("--latent-reg", type=float, default=1e-5, help="Latent regularization (default: 1e-5, reduced from 1e-4)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/pose_autoencoder.pt"))
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--eval-threshold", type=float, default=None)
    parser.add_argument("--scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--early-stop", type=int, default=10, help="Early stopping patience (epochs)")
    return parser.parse_args()


def collate_fn(batch):
    tensors = torch.stack([item[0] for item in batch], dim=0)
    metas = [item[1] for item in batch]
    return tensors, metas


def make_dataloaders(args: argparse.Namespace):
    dataset = PoseSequenceDataset(
        pose_dir=args.pose_dir,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )
    if args.mode == "train" and len(dataset) > 1:
        val_size = max(int(len(dataset) * args.val_split), 1)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )
    return train_loader, val_loader


def train(args: argparse.Namespace):
    device = torch.device(args.device)
    model = PoseAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Use MSELoss instead of SmoothL1Loss for better gradient flow
    criterion = torch.nn.MSELoss()
    
    # Optional learning rate scheduler
    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    train_loader, val_loader = make_dataloaders(args)
    best_val = float("inf")
    patience_counter = 0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Latent regularization: {args.latent_reg}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Loss function: MSELoss")
    print(f"  Scheduler: {scheduler is not None}")
    print(f"  Early stopping patience: {args.early_stop}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch, _ in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, latent = model(batch)
            loss = criterion(recon, batch) + args.latent_reg * latent.pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            running += loss.item() * batch.size(0)
        train_loss = running / len(train_loader.dataset)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for batch, _ in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                    batch = batch.to(device)
                    recon, latent = model(batch)
                    loss = criterion(recon, batch) + args.latent_reg * latent.pow(2).mean()
                    total += loss.item() * batch.size(0)
            val_loss = total / len(val_loader.dataset)
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val:
                improvement = best_val - val_loss
                best_val = val_loss
                patience_counter = 0
                torch.save({"state_dict": model.state_dict(), "epoch": epoch}, args.checkpoint)
                print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} (✓ best, improved by {improvement:.4f})")
            else:
                patience_counter += 1
                print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} (patience: {patience_counter}/{args.early_stop})")
                
                # Early stopping
                if patience_counter >= args.early_stop:
                    print(f"\n⚠️  Early stopping triggered after {epoch} epochs (no improvement for {args.early_stop} epochs)")
                    break
        else:
            print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

    if val_loader is None:
        torch.save({"state_dict": model.state_dict(), "epoch": args.epochs}, args.checkpoint)

    stats = compute_reconstruction_stats(model, train_loader, device)
    save_threshold(stats)


def compute_reconstruction_stats(model, loader, device) -> dict:
    model.eval()
    criterion = torch.nn.MSELoss(reduction="none")
    errors: List[float] = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch).mean(dim=(1, 2, 3))
            errors.extend(loss.cpu().tolist())
    errors = torch.tensor(errors)
    threshold = errors.quantile(0.95).item()
    return {"mean": errors.mean().item(), "std": errors.std().item(), "threshold": threshold}


def save_threshold(stats: dict):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "anomaly_threshold.json"
    with open(out_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved threshold stats to {out_path}")


def evaluate(args: argparse.Namespace):
    device = torch.device(args.device)
    model = PoseAutoencoder().to(device)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["state_dict"])

    dataset = PoseSequenceDataset(
        pose_dir=args.pose_dir,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    threshold = args.eval_threshold
    threshold_meta = Path("results/anomaly_threshold.json")
    if threshold is None and threshold_meta.exists():
        data = json.loads(threshold_meta.read_text())
        threshold = data.get("threshold")

    if threshold is None:
        raise ValueError("Provide --eval-threshold or run training to compute one")

    criterion = torch.nn.MSELoss(reduction="mean")
    print(f"Using anomaly threshold: {threshold:.4f}")
    model.eval()
    with torch.no_grad():
        for batch, metas in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            error = criterion(recon, batch).item()
            meta = metas[0]["meta"]
            is_anomaly = error > threshold
            print(
                f"{meta['source_file']:30s} start={meta['start_index']:04d} error={error:.4f} anomaly={is_anomaly}"
            )


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
