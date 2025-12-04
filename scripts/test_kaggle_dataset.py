"""Test the trained anomaly detection model on Kaggle Real Fall dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_model import PoseAutoencoder
from utils.pose_processing import normalize_pose_sequence, pad_or_trim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test model on Kaggle Fall dataset")
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        default=Path("data/KaggleRealDataset"),
        help="Path to Kaggle dataset",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/pose_autoencoder.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--threshold-file",
        type=Path,
        default=Path("results/anomaly_threshold.json"),
        help="Path to threshold JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manual threshold (overrides file)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=64,
        help="Sequence length for model input",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Stride for sliding window",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples per class (for quick testing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/kaggle_test_results.json"),
        help="Output file for results",
    )
    return parser.parse_args()


def load_csv_to_pose_array(csv_path: Path) -> np.ndarray:
    """Load CSV file and convert to (T, J, 2) pose array.
    
    Args:
        csv_path: Path to CSV file with columns [Frame, Keypoint, X, Y, Confidence]
    
    Returns:
        Array of shape (T, J, 2) where T=frames, J=17 joints, 2=x,y coordinates
    """
    df = pd.read_csv(csv_path)
    
    # Get number of frames and keypoints
    frames = sorted(df['Frame'].unique())
    keypoints = [
        'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
        'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
        'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
        'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
    ]
    
    num_frames = len(frames)
    num_joints = 17
    
    # Initialize array with zeros
    pose_array = np.zeros((num_frames, num_joints, 2), dtype=np.float32)
    
    # Fill in the pose data
    for frame_idx, frame in enumerate(frames):
        frame_data = df[df['Frame'] == frame]
        for joint_idx, keypoint in enumerate(keypoints):
            kp_data = frame_data[frame_data['Keypoint'] == keypoint]
            if not kp_data.empty:
                pose_array[frame_idx, joint_idx, 0] = kp_data['X'].values[0]
                pose_array[frame_idx, joint_idx, 1] = kp_data['Y'].values[0]
    
    # Pad to 18 joints to match model expectations (add zero column)
    pose_array_18 = np.zeros((num_frames, 18, 2), dtype=np.float32)
    pose_array_18[:, :17, :] = pose_array
    
    return pose_array_18


def create_windows(pose_sequence: np.ndarray, sequence_length: int, stride: int) -> List[np.ndarray]:
    """Create sliding windows from pose sequence."""
    windows = []
    total_frames = pose_sequence.shape[0]
    
    if total_frames < sequence_length:
        # Pad if sequence is too short
        padded = pad_or_trim(pose_sequence, sequence_length)
        windows.append(padded)
    else:
        # Create sliding windows
        start = 0
        while start + sequence_length <= total_frames:
            windows.append(pose_sequence[start:start + sequence_length])
            start += stride
        
        # Add final window if there's remaining data
        if start < total_frames:
            windows.append(pose_sequence[total_frames - sequence_length:total_frames])
    
    return windows


def compute_reconstruction_error(
    model: torch.nn.Module,
    pose_tensor: torch.Tensor,
    device: torch.device
) -> float:
    """Compute reconstruction error for a single sequence."""
    model.eval()
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        pose_tensor = pose_tensor.unsqueeze(0).to(device)  # Add batch dimension
        recon, _ = model(pose_tensor)
        error = criterion(recon, pose_tensor).item()
    
    return error


def test_single_video(
    csv_path: Path,
    model: torch.nn.Module,
    threshold: float,
    sequence_length: int,
    stride: int,
    device: torch.device,
) -> Tuple[float, bool]:
    """Test a single video and return average error and prediction.
    
    Returns:
        Tuple of (average_error, is_anomaly)
    """
    try:
        # Load and preprocess
        pose_array = load_csv_to_pose_array(csv_path)
        
        # Normalize
        pose_array = normalize_pose_sequence(pose_array)
        
        # Create windows
        windows = create_windows(pose_array, sequence_length, stride)
        
        # Compute errors for all windows
        errors = []
        for window in windows:
            # Convert to tensor: (2, T, J)
            tensor = torch.from_numpy(window.transpose(2, 0, 1)).float()
            error = compute_reconstruction_error(model, tensor, device)
            errors.append(error)
        
        # Use maximum error (most anomalous window)
        avg_error = max(errors) if errors else 0.0
        is_anomaly = avg_error > threshold
        
        return avg_error, is_anomaly
    
    except Exception as e:
        print(f"Error processing {csv_path.name}: {e}")
        return 0.0, False


def test_dataset(args: argparse.Namespace):
    """Test the model on the entire Kaggle dataset."""
    
    # Load model
    device = torch.device(args.device)
    model = PoseAutoencoder().to(device)
    
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    
    # Load threshold
    if args.threshold is not None:
        threshold = args.threshold
    elif args.threshold_file.exists():
        with open(args.threshold_file, "r") as f:
            data = json.load(f)
            threshold = data.get("threshold")
    else:
        raise ValueError("No threshold provided or threshold file not found")
    
    print("=" * 70)
    print("TESTING ON KAGGLE FALL DATASET")
    print("=" * 70)
    print(f"\nModel checkpoint: {args.checkpoint}")
    print(f"Anomaly threshold: {threshold:.4f}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Stride: {args.stride}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Get file lists
    fall_dir = args.kaggle_dir / "Fall" / "Keypoints_CSV"
    no_fall_dir = args.kaggle_dir / "No_Fall" / "Keypoints_CSV"
    
    fall_files = sorted(list(fall_dir.glob("*.csv")))
    no_fall_files = sorted(list(no_fall_dir.glob("*.csv")))
    
    if args.max_samples:
        fall_files = fall_files[:args.max_samples]
        no_fall_files = no_fall_files[:args.max_samples]
    
    print(f"\nFall videos: {len(fall_files)}")
    print(f"No Fall videos: {len(no_fall_files)}")
    print(f"Total: {len(fall_files) + len(no_fall_files)}")
    
    # Test all videos
    y_true = []
    y_pred = []
    y_scores = []
    
    print("\n" + "=" * 70)
    print("Testing Fall videos (label=1, expecting high error)...")
    print("=" * 70)
    for csv_path in tqdm(fall_files, desc="Fall"):
        error, is_anomaly = test_single_video(
            csv_path, model, threshold, args.sequence_length, args.stride, device
        )
        y_true.append(1)  # Fall = positive class
        y_pred.append(1 if is_anomaly else 0)
        y_scores.append(error)
    
    print("\n" + "=" * 70)
    print("Testing No Fall videos (label=0, expecting low error)...")
    print("=" * 70)
    for csv_path in tqdm(no_fall_files, desc="No Fall"):
        error, is_anomaly = test_single_video(
            csv_path, model, threshold, args.sequence_length, args.stride, device
        )
        y_true.append(0)  # No Fall = negative class
        y_pred.append(1 if is_anomaly else 0)
        y_scores.append(error)
    
    # Compute metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.0
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} (of predicted falls, how many are real)")
    print(f"   Recall:    {recall:.4f} (of real falls, how many detected)")
    print(f"   F1 Score:  {f1:.4f} (harmonic mean of precision & recall)")
    print(f"   ROC AUC:   {auc:.4f} (area under ROC curve)")
    
    print(f"\n🎯 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              No Fall  Fall")
    print(f"   Actual No   {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"          Fall {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\n   True Positives (TP):  {tp:5d} - Correctly detected falls")
    print(f"   False Positives (FP): {fp:5d} - False alarms (no fall detected as fall)")
    print(f"   True Negatives (TN):  {tn:5d} - Correctly detected no fall")
    print(f"   False Negatives (FN): {fn:5d} - Missed falls (fall detected as no fall)")
    
    print(f"\n📈 Error Statistics:")
    fall_scores = y_scores[y_true == 1]
    no_fall_scores = y_scores[y_true == 0]
    print(f"   Fall errors:    mean={fall_scores.mean():.4f}, std={fall_scores.std():.4f}")
    print(f"   No Fall errors: mean={no_fall_scores.mean():.4f}, std={no_fall_scores.std():.4f}")
    print(f"   Threshold:      {threshold:.4f}")
    
    print(f"\n🔍 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Fall', 'Fall']))
    
    # Save results
    results = {
        "threshold": float(threshold),
        "num_fall_samples": len(fall_files),
        "num_no_fall_samples": len(no_fall_files),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "fall_error_mean": float(fall_scores.mean()),
        "fall_error_std": float(fall_scores.std()),
        "no_fall_error_mean": float(no_fall_scores.mean()),
        "no_fall_error_std": float(no_fall_scores.std()),
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {args.output}")
    
    # Threshold optimization
    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 70)
    
    print("\n🔍 Finding optimal threshold...")
    best_f1 = 0
    best_threshold = threshold
    best_metrics = {}
    
    # Try different thresholds
    thresholds_to_try = np.linspace(y_scores.min(), y_scores.max(), 100)
    
    for thresh in thresholds_to_try:
        y_pred_temp = (y_scores > thresh).astype(int)
        f1_temp = f1_score(y_true, y_pred_temp, zero_division=0)
        
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = thresh
            best_metrics = {
                'accuracy': accuracy_score(y_true, y_pred_temp),
                'precision': precision_score(y_true, y_pred_temp, zero_division=0),
                'recall': recall_score(y_true, y_pred_temp, zero_division=0),
                'f1': f1_temp,
            }
    
    print(f"\n📊 Current threshold: {threshold:.4f}")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    print(f"\n✨ Optimal threshold: {best_threshold:.4f}")
    print(f"   Accuracy:  {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall:    {best_metrics['recall']:.4f}")
    print(f"   F1 Score:  {best_metrics['f1']:.4f}")
    
    improvement = (best_metrics['f1'] - f1) / max(f1, 0.001) * 100
    print(f"\n   → F1 improvement: {improvement:+.1f}%")
    
    print(f"\n💡 To use optimal threshold, run:")
    print(f"   python scripts/test_kaggle_dataset.py --threshold {best_threshold:.4f}")
    
    # Add optimal results to output
    results['optimal_threshold'] = float(best_threshold)
    results['optimal_metrics'] = {k: float(v) for k, v in best_metrics.items()}
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print("=" * 70)
    
    return results


def main():
    args = parse_args()
    test_dataset(args)


if __name__ == "__main__":
    main()
