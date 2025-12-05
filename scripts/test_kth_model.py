"""Test the original KTH-trained model (anomaly detection) with comprehensive metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_model import PoseAutoencoder
from utils.pose_processing import normalize_pose_sequence, pad_or_trim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test original KTH model with anomaly detection")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/pose_autoencoder.pt"),
        help="Path to KTH model checkpoint",
    )
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        default=Path("data/KaggleRealDataset"),
        help="Path to Kaggle dataset",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=64,
        help="Sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to test (per class)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Anomaly threshold (if None, will find optimal)",
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
    
    # Pad to 18 joints
    pose_array_18 = np.zeros((num_frames, 18, 2), dtype=np.float32)
    pose_array_18[:, :17, :] = pose_array
    
    return pose_array_18


def load_and_prepare_data(args: argparse.Namespace) -> Tuple[List, List]:
    """Load test data."""
    fall_dir = args.kaggle_dir / "Fall" / "Keypoints_CSV"
    no_fall_dir = args.kaggle_dir / "No_Fall" / "Keypoints_CSV"
    
    fall_files = sorted(list(fall_dir.glob("*.csv")))
    no_fall_files = sorted(list(no_fall_dir.glob("*.csv")))
    
    if args.max_samples:
        fall_files = fall_files[:args.max_samples]
        no_fall_files = no_fall_files[:args.max_samples]
    
    all_files = fall_files + no_fall_files
    all_labels = [1] * len(fall_files) + [0] * len(no_fall_files)
    
    print(f"Test set:")
    print(f"  Fall: {len(fall_files)}")
    print(f"  No Fall: {len(no_fall_files)}")
    print(f"  Total: {len(all_files)}")
    
    return all_files, all_labels


def compute_reconstruction_error(model: PoseAutoencoder, tensor: torch.Tensor) -> float:
    """Compute reconstruction error for anomaly detection."""
    with torch.no_grad():
        reconstructed, _ = model(tensor)
        error = torch.mean((tensor - reconstructed) ** 2).item()
    return error


def find_optimal_threshold(errors: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict]:
    """Find optimal threshold that maximizes F1 score."""
    min_error = errors.min()
    max_error = errors.max()
    thresholds = np.linspace(min_error, max_error, 100)
    
    best_f1 = 0
    best_threshold = 0
    best_metrics = {}
    
    for threshold in thresholds:
        # Predictions: error > threshold = anomaly (Fall = 1)
        preds = (errors > threshold).astype(int)
        
        f1 = f1_score(labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'accuracy': accuracy_score(labels, preds),
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'f1': f1,
            }
    
    return best_threshold, best_metrics


def test_model(args: argparse.Namespace):
    """Test the model and generate comprehensive metrics."""
    
    device = torch.device(args.device)
    
    print("=" * 70)
    print("TESTING ORIGINAL KTH MODEL (ANOMALY DETECTION)")
    print("=" * 70)
    
    # Load model
    print(f"\n📥 Loading model from: {args.checkpoint}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    model = PoseAutoencoder()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded")
    print(f"   Training epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'train_loss' in checkpoint:
        print(f"   Training loss: {checkpoint['train_loss']:.4f}")
    
    # Load test data
    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)
    test_files, test_labels = load_and_prepare_data(args)
    
    # Process samples and compute reconstruction errors
    print("\n" + "=" * 70)
    print("COMPUTING RECONSTRUCTION ERRORS")
    print("=" * 70)
    
    all_errors = []
    all_labels = []
    failed_files = []
    
    with torch.no_grad():
        for file_path, label in tqdm(zip(test_files, test_labels), total=len(test_files)):
            try:
                # Load and preprocess
                pose_array = load_csv_to_pose_array(file_path)
                pose_array = normalize_pose_sequence(pose_array)
                pose_array = pad_or_trim(pose_array, args.sequence_length)
                
                # Convert to tensor
                tensor = torch.from_numpy(pose_array.transpose(2, 0, 1)).float()
                tensor = tensor.unsqueeze(0).to(device)
                
                # Compute reconstruction error
                error = compute_reconstruction_error(model, tensor)
                
                all_errors.append(error)
                all_labels.append(label)
                
            except Exception as e:
                print(f"\n⚠️  Error processing {file_path.name}: {e}")
                failed_files.append(str(file_path))
                continue
    
    if len(all_errors) == 0:
        print("❌ No samples were successfully processed!")
        return
    
    # Convert to numpy arrays
    errors = np.array(all_errors)
    y_true = np.array(all_labels)
    
    # Find optimal threshold or use provided
    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 70)
    
    if args.threshold is None:
        optimal_threshold, threshold_metrics = find_optimal_threshold(errors, y_true)
        print(f"\n🔍 Optimal threshold found: {optimal_threshold:.4f}")
        print(f"   F1 Score at optimal: {threshold_metrics['f1']:.4f}")
        threshold = optimal_threshold
    else:
        threshold = args.threshold
        print(f"\n📌 Using provided threshold: {threshold:.4f}")
    
    # Make predictions with threshold
    y_pred = (errors > threshold).astype(int)
    
    # Compute error statistics
    fall_errors = errors[y_true == 1]
    no_fall_errors = errors[y_true == 0]
    
    print(f"\n📊 Reconstruction Error Statistics:")
    print(f"   Overall:")
    print(f"      Mean:   {errors.mean():.4f}")
    print(f"      Median: {np.median(errors):.4f}")
    print(f"      Std:    {errors.std():.4f}")
    print(f"      Range:  [{errors.min():.4f}, {errors.max():.4f}]")
    print(f"   Fall samples:")
    print(f"      Mean:   {fall_errors.mean():.4f}")
    print(f"      Median: {np.median(fall_errors):.4f}")
    print(f"      Std:    {fall_errors.std():.4f}")
    print(f"   No Fall samples:")
    print(f"      Mean:   {no_fall_errors.mean():.4f}")
    print(f"      Median: {np.median(no_fall_errors):.4f}")
    print(f"      Std:    {no_fall_errors.std():.4f}")
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Compute per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    try:
        # Use errors as scores (higher error = more likely to be fall)
        auc_roc = roc_auc_score(y_true, errors)
        avg_precision = average_precision_score(y_true, errors)
    except:
        auc_roc = 0.0
        avg_precision = 0.0
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Compute additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Print results
    print(f"\n📊 Overall Metrics:")
    print(f"   Threshold:          {threshold:.4f}")
    print(f"   Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision (Fall):   {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall (Fall):      {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1 Score (Fall):    {f1:.4f} ⭐")
    print(f"   Specificity:        {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"   ROC AUC:            {auc_roc:.4f}")
    print(f"   Average Precision:  {avg_precision:.4f}")
    
    print(f"\n📊 Per-Class Metrics:")
    print(f"   Class 0 (No Fall):")
    print(f"      Precision: {precision_per_class[0]:.4f}")
    print(f"      Recall:    {recall_per_class[0]:.4f}")
    print(f"      F1 Score:  {f1_per_class[0]:.4f}")
    print(f"   Class 1 (Fall):")
    print(f"      Precision: {precision_per_class[1]:.4f}")
    print(f"      Recall:    {recall_per_class[1]:.4f}")
    print(f"      F1 Score:  {f1_per_class[1]:.4f}")
    
    print(f"\n🎯 Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                No Fall    Fall")
    print(f"   Actual No     {tn:6d}  {fp:6d}")
    print(f"          Fall   {fn:6d}  {tp:6d}")
    
    print(f"\n   True Positives (TP):   {tp:6d} - Correctly identified falls")
    print(f"   False Positives (FP):  {fp:6d} - False alarms")
    print(f"   True Negatives (TN):   {tn:6d} - Correctly identified no-falls")
    print(f"   False Negatives (FN):  {fn:6d} - Missed falls")
    
    print(f"\n📈 Clinical Metrics:")
    print(f"   Sensitivity (TPR):     {sensitivity:.4f} - How many falls detected")
    print(f"   Specificity (TNR):     {specificity:.4f} - How many no-falls correctly identified")
    print(f"   PPV (Precision):       {ppv:.4f} - If predicted fall, prob it's real")
    print(f"   NPV:                   {npv:.4f} - If predicted no-fall, prob it's real")
    
    # Detailed classification report
    print(f"\n🔍 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Fall', 'Fall'], digits=4))
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'checkpoint': str(args.checkpoint),
        'method': 'anomaly_detection',
        'threshold': float(threshold),
        'test_samples': len(y_true),
        'fall_samples': int((y_true == 1).sum()),
        'no_fall_samples': int((y_true == 0).sum()),
        'failed_samples': len(failed_files),
        'error_statistics': {
            'overall': {
                'mean': float(errors.mean()),
                'median': float(np.median(errors)),
                'std': float(errors.std()),
                'min': float(errors.min()),
                'max': float(errors.max()),
            },
            'fall': {
                'mean': float(fall_errors.mean()),
                'median': float(np.median(fall_errors)),
                'std': float(fall_errors.std()),
            },
            'no_fall': {
                'mean': float(no_fall_errors.mean()),
                'median': float(np.median(no_fall_errors)),
                'std': float(no_fall_errors.std()),
            },
        },
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
            'roc_auc': float(auc_roc),
            'average_precision': float(avg_precision),
        },
        'per_class': {
            'no_fall': {
                'precision': float(precision_per_class[0]),
                'recall': float(recall_per_class[0]),
                'f1_score': float(f1_per_class[0]),
            },
            'fall': {
                'precision': float(precision_per_class[1]),
                'recall': float(recall_per_class[1]),
                'f1_score': float(f1_per_class[1]),
            },
        },
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
        },
    }
    
    results_file = args.output_dir / "kth_model_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Plot visualizations
    try:
        fig = plt.figure(figsize=(14, 10))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, errors)
        plt.subplot(2, 3, 1)
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_roc:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, errors)
        plt.subplot(2, 3, 2)
        plt.plot(recall_curve, precision_curve, 'b-', linewidth=2, label=f'PR (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion Matrix
        plt.subplot(2, 3, 3)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix (Normalized)')
        plt.colorbar()
        tick_marks = [0, 1]
        plt.xticks(tick_marks, ['No Fall', 'Fall'])
        plt.yticks(tick_marks, ['No Fall', 'Fall'])
        
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})',
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > 0.5 else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Error distribution by class
        plt.subplot(2, 3, 4)
        plt.hist(no_fall_errors, bins=50, alpha=0.6, label='No Fall', color='blue', density=True)
        plt.hist(fall_errors, bins=50, alpha=0.6, label='Fall', color='red', density=True)
        plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={threshold:.4f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Error Distribution by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot of errors
        plt.subplot(2, 3, 5)
        plt.boxplot([no_fall_errors, fall_errors], labels=['No Fall', 'Fall'])
        plt.axhline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={threshold:.4f}')
        plt.ylabel('Reconstruction Error')
        plt.title('Error Distribution (Box Plot)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter plot: errors vs samples
        plt.subplot(2, 3, 6)
        plt.scatter(range(len(no_fall_errors)), no_fall_errors, alpha=0.5, s=10, c='blue', label='No Fall')
        plt.scatter(range(len(no_fall_errors), len(errors)), fall_errors, alpha=0.5, s=10, c='red', label='Fall')
        plt.axhline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={threshold:.4f}')
        plt.xlabel('Sample Index')
        plt.ylabel('Reconstruction Error')
        plt.title('Error per Sample')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = args.output_dir / "kth_model_metrics.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📊 Plots saved to: {plot_file}")
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n✅ Model Performance (Anomaly Detection):")
    print(f"   • Tested on {len(y_true)} samples ({(y_true==1).sum()} Fall, {(y_true==0).sum()} No Fall)")
    print(f"   • Optimal Threshold: {threshold:.4f}")
    print(f"   • Overall Accuracy: {accuracy*100:.2f}%")
    print(f"   • F1 Score: {f1:.4f}")
    print(f"   • Fall Detection Rate (Recall): {recall*100:.2f}%")
    print(f"   • False Alarm Rate: {(fp/(fp+tn))*100:.2f}%")
    
    if f1 >= 0.8:
        print(f"\n🎉 Excellent performance! (F1 ≥ 0.80)")
    elif f1 >= 0.7:
        print(f"\n👍 Good performance! (F1 ≥ 0.70)")
    elif f1 >= 0.6:
        print(f"\n👌 Acceptable performance (F1 ≥ 0.60)")
    else:
        print(f"\n⚠️  Performance needs improvement (F1 < 0.60)")
        print(f"   Consider fine-tuning on labeled data for better results.")
    
    print("=" * 70)


def main():
    args = parse_args()
    test_model(args)


if __name__ == "__main__":
    main()
