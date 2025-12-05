"""Test the fine-tuned Kaggle model with comprehensive metrics."""
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
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_model import PoseAutoencoder
from utils.pose_processing import normalize_pose_sequence, pad_or_trim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test fine-tuned model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/pose_autoencoder_kaggle_finetuned.pt"),
        help="Path to fine-tuned model checkpoint",
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


class FallClassifier(nn.Module):
    """Wrapper around PoseAutoencoder with classification head."""
    
    def __init__(self, pretrained_autoencoder: PoseAutoencoder = None):
        super().__init__()
        
        if pretrained_autoencoder is not None:
            self.autoencoder = pretrained_autoencoder
        else:
            self.autoencoder = PoseAutoencoder()
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )
    
    def forward(self, x: torch.Tensor):
        _, latent = self.autoencoder(x)
        logits = self.classifier(latent)
        return logits


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


def test_model(args: argparse.Namespace):
    """Test the model and generate comprehensive metrics."""
    
    device = torch.device(args.device)
    
    print("=" * 70)
    print("TESTING FINE-TUNED MODEL")
    print("=" * 70)
    
    # Load model
    print(f"\n📥 Loading model from: {args.checkpoint}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    model = FallClassifier()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded")
    print(f"   Training epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
    
    # Load test data
    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)
    test_files, test_labels = load_and_prepare_data(args)
    
    # Process samples
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_logits = []
    failed_files = []
    
    softmax = nn.Softmax(dim=1)
    
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
                
                # Inference
                logits = model(tensor)
                probs = softmax(logits)
                pred = torch.argmax(logits, dim=1).item()
                
                all_preds.append(pred)
                all_probs.append(probs[0, 1].item())  # Probability of Fall class
                all_labels.append(label)
                all_logits.append(logits[0].cpu().numpy())
                
            except Exception as e:
                print(f"\n⚠️  Error processing {file_path.name}: {e}")
                failed_files.append(str(file_path))
                continue
    
    if len(all_preds) == 0:
        print("❌ No samples were successfully processed!")
        return
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
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
        auc_roc = roc_auc_score(y_true, y_probs)
        avg_precision = average_precision_score(y_true, y_probs)
    except:
        auc_roc = 0.0
        avg_precision = 0.0
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Compute additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
    
    # Print results
    print(f"\n📊 Overall Metrics:")
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
        'test_samples': len(y_true),
        'fall_samples': int((y_true == 1).sum()),
        'no_fall_samples': int((y_true == 0).sum()),
        'failed_samples': len(failed_files),
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
    
    results_file = args.output_dir / "finetuned_model_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Plot ROC curve
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_roc:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
        plt.subplot(2, 2, 2)
        plt.plot(recall_curve, precision_curve, 'b-', linewidth=2, label=f'PR (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion Matrix heatmap
        plt.subplot(2, 2, 3)
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
        
        # Probability distribution
        plt.subplot(2, 2, 4)
        fall_probs = y_probs[y_true == 1]
        no_fall_probs = y_probs[y_true == 0]
        plt.hist(no_fall_probs, bins=30, alpha=0.5, label='No Fall', color='blue')
        plt.hist(fall_probs, bins=30, alpha=0.5, label='Fall', color='red')
        plt.xlabel('Predicted Probability of Fall')
        plt.ylabel('Count')
        plt.title('Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = args.output_dir / "finetuned_model_metrics.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📊 Plots saved to: {plot_file}")
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n✅ Model Performance:")
    print(f"   • Tested on {len(y_true)} samples ({(y_true==1).sum()} Fall, {(y_true==0).sum()} No Fall)")
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
        print(f"   Consider:")
        print(f"   • Training for more epochs")
        print(f"   • Using more training data")
        print(f"   • Adjusting learning rate")
    
    print("=" * 70)


def main():
    args = parse_args()
    test_model(args)


if __name__ == "__main__":
    main()
