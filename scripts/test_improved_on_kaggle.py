"""
Test the improved hybrid model (with LSTM) on Kaggle dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_model import PoseAutoencoder
from utils.pose_processing import normalize_pose_sequence, pad_or_trim


class ImprovedHybridDetectorLSTM(nn.Module):
    """
    Improved hybrid model with LSTM temporal modeling.
    Must match the architecture from training script.
    """
    
    def __init__(self, lstm_hidden: int = 128, lstm_layers: int = 2):
        super().__init__()
        
        self.autoencoder = PoseAutoencoder()
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
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
            nn.Linear(64, 2),
        )
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 2, T, J) pose tensor
            
        Returns:
            (reconstructed, logits)
        """
        B, C, T, J = x.shape
        
        # Process each time step through autoencoder
        reconstructed_frames = []
        latent_frames = []
        
        for t in range(T):
            frame = x[:, :, t:t+1, :]
            recon_frame, latent_frame = self.autoencoder(frame)
            recon_frame = recon_frame.squeeze(2)
            reconstructed_frames.append(recon_frame)
            latent_frames.append(latent_frame)
        
        # Stack reconstructed frames
        reconstructed = torch.stack(reconstructed_frames, dim=2)
        latent_sequence = torch.stack(latent_frames, dim=1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(latent_sequence)
        
        # Use last LSTM output for classification
        lstm_final = lstm_out[:, -1, :]
        logits = self.classifier(lstm_final)
        
        return reconstructed, logits


def load_csv_to_pose_array(csv_path: Path, sequence_length: int = 64) -> Optional[np.ndarray]:
    """
    Load Kaggle CSV file and convert to pose array format.
    
    Args:
        csv_path: Path to CSV file
        sequence_length: Target sequence length
        
    Returns:
        Pose array of shape (T, J, 2) or None if failed
    """
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return None
        
        # Check format - could be wide (x0, y0, x1, y1...) or long (Frame, Keypoint, X, Y)
        if 'Frame' in df.columns and 'Keypoint' in df.columns:
            # Long format - convert to wide
            keypoint_mapping = {
                'Nose': 0, 'Neck': 1,
                'Right Shoulder': 2, 'Right Elbow': 3, 'Right Wrist': 4,
                'Left Shoulder': 5, 'Left Elbow': 6, 'Left Wrist': 7,
                'Right Hip': 8, 'Right Knee': 9, 'Right Ankle': 10,
                'Left Hip': 11, 'Left Knee': 12, 'Left Ankle': 13,
                'Right Eye': 14, 'Left Eye': 15,
                'Right Ear': 16, 'Left Ear': 17
            }
            
            frames = sorted(df['Frame'].unique())
            num_frames = len(frames)
            pose_array = np.zeros((num_frames, 18, 2))
            
            for frame_idx, frame_num in enumerate(frames):
                frame_data = df[df['Frame'] == frame_num]
                # Use vectorized operations instead of iterrows
                for keypoint_name, joint_idx in keypoint_mapping.items():
                    kp_data = frame_data[frame_data['Keypoint'] == keypoint_name]
                    if not kp_data.empty:
                        pose_array[frame_idx, joint_idx, 0] = kp_data['X'].iloc[0]
                        pose_array[frame_idx, joint_idx, 1] = kp_data['Y'].iloc[0]
        else:
            # Wide format
            x_cols = [f'x{i}' for i in range(18)]
            y_cols = [f'y{i}' for i in range(18)]
            
            if not all(col in df.columns for col in x_cols + y_cols):
                return None
            
            x_coords = df[x_cols].values  # (T, 18)
            y_coords = df[y_cols].values  # (T, 18)
            
            # Stack to (T, 18, 2)
            pose_array = np.stack([x_coords, y_coords], axis=-1)
        
        # Normalize and pad/trim
        pose_array = normalize_pose_sequence(pose_array)
        pose_array = pad_or_trim(pose_array, sequence_length)
        
        return pose_array
        
    except Exception as e:
        return None


def test_model(model: ImprovedHybridDetectorLSTM,
               test_dir: Path,
               device: str,
               sequence_length: int = 64) -> dict:
    """
    Test model on Kaggle dataset.
    
    Args:
        model: Trained model
        test_dir: Path to Kaggle test data
        device: Device to run on
        sequence_length: Sequence length
        
    Returns:
        Dictionary with results
    """
    model.eval()
    
    # Find all CSV files
    fall_dir = test_dir / "Fall" / "Keypoints_CSV"
    no_fall_dir = test_dir / "No_Fall" / "Keypoints_CSV"
    
    fall_files = sorted(fall_dir.glob("*.csv"))
    no_fall_files = sorted(no_fall_dir.glob("*.csv"))
    
    print(f"Found {len(fall_files)} Fall videos and {len(no_fall_files)} No Fall videos")
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_recon_errors = []
    failed_files = []
    
    # Process Fall videos
    print("\nProcessing Fall videos...")
    for csv_file in tqdm(fall_files, desc="Fall videos"):
        pose_array = load_csv_to_pose_array(csv_file, sequence_length)
        if pose_array is None:
            failed_files.append(str(csv_file))
            continue
        
        # Convert to tensor
        tensor = torch.from_numpy(pose_array.transpose(2, 0, 1)).float()  # (2, T, J)
        tensor = tensor.unsqueeze(0).to(device)  # (1, 2, T, J)
        
        with torch.no_grad():
            reconstructed, logits = model(tensor)
            
            # Classification
            probs = torch.softmax(logits, dim=1)
            predicted = torch.argmax(logits, dim=1).item()
            prob_fall = probs[0, 1].item()
            
            # Reconstruction error
            recon_error = torch.mean((tensor - reconstructed) ** 2).item()
        
        all_labels.append(1)  # Fall = 1
        all_predictions.append(predicted)
        all_probabilities.append(prob_fall)
        all_recon_errors.append(recon_error)
    
    # Process No Fall videos
    print("\nProcessing No Fall videos...")
    for csv_file in tqdm(no_fall_files, desc="No Fall videos"):
        pose_array = load_csv_to_pose_array(csv_file, sequence_length)
        if pose_array is None:
            failed_files.append(str(csv_file))
            continue
        
        # Convert to tensor
        tensor = torch.from_numpy(pose_array.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            reconstructed, logits = model(tensor)
            
            # Classification
            probs = torch.softmax(logits, dim=1)
            predicted = torch.argmax(logits, dim=1).item()
            prob_fall = probs[0, 1].item()
            
            # Reconstruction error
            recon_error = torch.mean((tensor - reconstructed) ** 2).item()
        
        all_labels.append(0)  # No Fall = 0
        all_predictions.append(predicted)
        all_probabilities.append(prob_fall)
        all_recon_errors.append(recon_error)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_recon_errors = np.array(all_recon_errors)
    
    print(f"\n{len(failed_files)} files failed to load")
    print(f"Successfully processed {len(all_labels)} samples")
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probabilities)
    except:
        roc_auc = 0.5
    
    # Reconstruction error statistics
    fall_mask = all_labels == 1
    no_fall_mask = all_labels == 0
    
    fall_recon_mean = np.mean(all_recon_errors[fall_mask])
    fall_recon_std = np.std(all_recon_errors[fall_mask])
    no_fall_recon_mean = np.mean(all_recon_errors[no_fall_mask])
    no_fall_recon_std = np.std(all_recon_errors[no_fall_mask])
    
    results = {
        'total_samples': len(all_labels),
        'fall_samples': int(np.sum(all_labels == 1)),
        'no_fall_samples': int(np.sum(all_labels == 0)),
        'failed_files': len(failed_files),
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'false_alarm_rate': float(false_alarm_rate),
            'roc_auc': float(roc_auc),
        },
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
        },
        'reconstruction_errors': {
            'fall_mean': float(fall_recon_mean),
            'fall_std': float(fall_recon_std),
            'no_fall_mean': float(no_fall_recon_mean),
            'no_fall_std': float(no_fall_recon_std),
        },
        'predictions': all_predictions.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probabilities.tolist(),
        'recon_errors': all_recon_errors.tolist(),
    }
    
    return results


def plot_results(results: dict, output_path: Path):
    """Create visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    labels = np.array(results['labels'])
    predictions = np.array(results['predictions'])
    probabilities = np.array(results['probabilities'])
    recon_errors = np.array(results['recon_errors'])
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = results['metrics']['roc_auc']
    
    axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random (AUC = 0.5)')
    axes[0, 0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(labels, probabilities)
    
    axes[0, 1].plot(recall_vals, precision_vals, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Recall', fontsize=12)
    axes[0, 1].set_ylabel('Precision', fontsize=12)
    axes[0, 1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    im = axes[1, 0].imshow(cm, cmap='Blues', aspect='auto')
    
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['No Fall', 'Fall'], fontsize=11)
    axes[1, 0].set_yticklabels(['No Fall', 'Fall'], fontsize=11)
    axes[1, 0].set_xlabel('Predicted', fontsize=12)
    axes[1, 0].set_ylabel('Actual', fontsize=12)
    axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = axes[1, 0].text(j, i, f'{cm[i, j]}',
                                  ha="center", va="center", color="black", fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Reconstruction Error Distribution
    fall_mask = labels == 1
    no_fall_mask = labels == 0
    
    axes[1, 1].hist(recon_errors[fall_mask], bins=50, alpha=0.6, label='Fall', color='red', density=True)
    axes[1, 1].hist(recon_errors[no_fall_mask], bins=50, alpha=0.6, label='No Fall', color='blue', density=True)
    axes[1, 1].axvline(np.mean(recon_errors[fall_mask]), color='red', linestyle='--', linewidth=2, label=f'Fall Mean: {np.mean(recon_errors[fall_mask]):.4f}')
    axes[1, 1].axvline(np.mean(recon_errors[no_fall_mask]), color='blue', linestyle='--', linewidth=2, label=f'No Fall Mean: {np.mean(recon_errors[no_fall_mask]):.4f}')
    axes[1, 1].set_xlabel('Reconstruction Error', fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)
    axes[1, 1].set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test improved hybrid model on Kaggle dataset")
    
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/improved_hybrid_detector.pt"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/kaggle_fall_detection"),
        help="Path to Kaggle test data directory",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/improved_model_test_results.json"),
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("results/improved_model_metrics.png"),
        help="Path to save plots",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=64,
        help="Sequence length",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("IMPROVED HYBRID MODEL TESTING ON KAGGLE DATASET")
    print("=" * 80)
    print()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = ImprovedHybridDetectorLSTM()
    
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation accuracy during training: {checkpoint.get('val_accuracy', 'unknown'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    model.eval()
    
    print(f"Model loaded on {args.device}")
    print()
    
    # Test
    results = test_model(model, args.test_dir, args.device, args.sequence_length)
    
    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"\nTotal samples: {results['total_samples']}")
    print(f"  Fall: {results['fall_samples']}")
    print(f"  No Fall: {results['no_fall_samples']}")
    print(f"  Failed: {results['failed_files']}")
    print()
    
    metrics = results['metrics']
    print("Classification Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1 Score:          {metrics['f1_score']:.4f}")
    print(f"  Specificity:       {metrics['specificity']:.4f}")
    print(f"  False Alarm Rate:  {metrics['false_alarm_rate']:.4f} ({metrics['false_alarm_rate']*100:.2f}%)")
    print(f"  ROC AUC:           {metrics['roc_auc']:.4f}")
    print()
    
    cm = results['confusion_matrix']
    print("Confusion Matrix:")
    print(f"  True Negatives:  {cm['true_negatives']:>6}")
    print(f"  False Positives: {cm['false_positives']:>6}")
    print(f"  False Negatives: {cm['false_negatives']:>6}")
    print(f"  True Positives:  {cm['true_positives']:>6}")
    print()
    
    recon = results['reconstruction_errors']
    print("Reconstruction Errors:")
    print(f"  Fall:    {recon['fall_mean']:.4f} ± {recon['fall_std']:.4f}")
    print(f"  No Fall: {recon['no_fall_mean']:.4f} ± {recon['no_fall_std']:.4f}")
    print()
    
    # Save results
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_json}")
    
    # Create plots
    plot_results(results, args.output_plot)
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
