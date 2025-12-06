"""
Test ensemble fall detector on Kaggle dataset and compare with all previous approaches.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_model import PoseAutoencoder
from models.ensemble_detector import EnsembleFallDetector
from utils.pose_processing import normalize_pose_sequence, pad_or_trim


def load_kaggle_csv_sequence(csv_path: Path, sequence_length: int = 64) -> np.ndarray:
    """Load a sequence from Kaggle CSV format (long format)."""
    try:
        df = pd.read_csv(csv_path)
        
        # Handle different column name formats
        if 'frame' in df.columns:
            frame_col = 'frame'
        elif 'Frame' in df.columns:
            frame_col = 'Frame'
        else:
            return None
        
        if 'keypoint' in df.columns:
            keypoint_col = 'keypoint'
        elif 'Keypoint' in df.columns:
            keypoint_col = 'Keypoint'
        else:
            return None
        
        if 'x' in df.columns and 'y' in df.columns:
            x_col, y_col = 'x', 'y'
        elif 'X' in df.columns and 'Y' in df.columns:
            x_col, y_col = 'X', 'Y'
        else:
            return None
        
        # Get unique frames and keypoints
        frames = sorted(df[frame_col].unique())
        keypoints = df[keypoint_col].unique()
        
        # Accept 17 or 18 keypoints (some datasets don't have Neck)
        if len(keypoints) not in [17, 18]:
            return None
        
        # Map keypoint names to indices (OpenPose COCO format)
        keypoint_map = {
            'Nose': 0,
            'Neck': 1,
            'Right Shoulder': 2,
            'Right Elbow': 3,
            'Right Wrist': 4,
            'Left Shoulder': 5,
            'Left Elbow': 6,
            'Left Wrist': 7,
            'Right Hip': 8,
            'Right Knee': 9,
            'Right Ankle': 10,
            'Left Hip': 11,
            'Left Knee': 12,
            'Left Ankle': 13,
            'Right Eye': 14,
            'Left Eye': 15,
            'Right Ear': 16,
            'Left Ear': 17
        }
        
        # Build sequence: (num_frames, 18, 2)
        sequence = []
        for frame_id in frames:
            frame_data = df[df[frame_col] == frame_id]
            pose = np.zeros((18, 2))
            
            for _, row in frame_data.iterrows():
                kp_name = row[keypoint_col]
                if kp_name in keypoint_map:
                    kp_id = keypoint_map[kp_name]
                    pose[kp_id, 0] = row[x_col]
                    pose[kp_id, 1] = row[y_col]
            
            sequence.append(pose)
        
        sequence = np.array(sequence)
        
        # Normalize and pad/trim
        sequence = normalize_pose_sequence(sequence)
        sequence = pad_or_trim(sequence, sequence_length)
        
        return sequence
    
    except Exception as e:
        return None


def load_kaggle_dataset(data_dir: Path, sequence_length: int = 64) -> Tuple[List[np.ndarray], List[int]]:
    """Load all Kaggle sequences."""
    fall_dir = data_dir / "Fall" / "Keypoints_CSV"
    no_fall_dir = data_dir / "No_Fall" / "Keypoints_CSV"
    
    sequences = []
    labels = []
    
    print("Loading Fall sequences...")
    if fall_dir.exists():
        fall_files = sorted(fall_dir.glob("*.csv"))
        for csv_file in tqdm(fall_files, desc="Loading Falls"):
            seq = load_kaggle_csv_sequence(csv_file, sequence_length)
            if seq is not None:
                sequences.append(seq)
                labels.append(1)  # Fall
    
    print("Loading No Fall sequences...")
    if no_fall_dir.exists():
        no_fall_files = sorted(no_fall_dir.glob("*.csv"))
        for csv_file in tqdm(no_fall_files, desc="Loading No Falls"):
            seq = load_kaggle_csv_sequence(csv_file, sequence_length)
            if seq is not None:
                sequences.append(seq)
                labels.append(0)  # No fall
    
    print(f"\nLoaded {len(sequences)} sequences:")
    print(f"  Falls: {sum(labels)}")
    print(f"  No Falls: {len(labels) - sum(labels)}")
    
    return sequences, labels


def test_ensemble(ensemble: EnsembleFallDetector, sequences: List[np.ndarray], 
                  labels: List[int]) -> Dict:
    """Test ensemble detector."""
    predictions = []
    confidences = []
    all_individual_preds = defaultdict(list)
    
    print("\nTesting ensemble...")
    for seq in tqdm(sequences, desc="Predicting"):
        pred, conf, individual_preds = ensemble.predict(seq)
        predictions.append(pred)
        confidences.append(conf)
        
        # Store individual predictions
        for name, (p, c) in individual_preds.items():
            all_individual_preds[name].append(p)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Compute fall detection rate and false alarm rate
    fall_detection_rate = recall  # Same as recall for fall class
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    results = {
        'ensemble': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fall_detection_rate': fall_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions,
            'confidences': confidences,
        }
    }
    
    # Compute metrics for individual detectors
    for name, preds in all_individual_preds.items():
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1_ind = f1_score(labels, preds, zero_division=0)
        
        cm_ind = confusion_matrix(labels, preds)
        tn_ind, fp_ind, fn_ind, tp_ind = cm_ind.ravel()
        
        fdr_ind = rec
        far_ind = fp_ind / (fp_ind + tn_ind) if (fp_ind + tn_ind) > 0 else 0.0
        
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1_ind,
            'fall_detection_rate': fdr_ind,
            'false_alarm_rate': far_ind,
            'confusion_matrix': cm_ind.tolist(),
            'predictions': preds,
        }
    
    return results


def print_results(results: Dict, detector_name: str):
    """Print results for a detector."""
    metrics = results[detector_name]
    
    print(f"\n{'=' * 80}")
    print(f"{detector_name.upper()} RESULTS")
    print('=' * 80)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nFall Detection Rate: {metrics['fall_detection_rate']:.4f} ({metrics['fall_detection_rate']*100:.2f}%)")
    print(f"False Alarm Rate: {metrics['false_alarm_rate']:.4f} ({metrics['false_alarm_rate']*100:.2f}%)")
    
    cm = np.array(metrics['confusion_matrix'])
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                No Fall    Fall")
    print(f"Actual No Fall     {cm[0,0]:<7}  {cm[0,1]:<7}")
    print(f"       Fall        {cm[1,0]:<7}  {cm[1,1]:<7}")


def compare_all_results(results: Dict):
    """Print comparison of all detectors."""
    print("\n" + "=" * 80)
    print("COMPARISON OF ALL DETECTORS")
    print("=" * 80)
    
    # Table header
    print(f"\n{'Detector':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FAR':<12}")
    print("-" * 80)
    
    # Sort by accuracy
    detector_names = sorted(results.keys(), key=lambda x: results[x]['accuracy'], reverse=True)
    
    for name in detector_names:
        metrics = results[name]
        print(f"{name:<20} "
              f"{metrics['accuracy']:.4f} ({metrics['accuracy']*100:5.2f}%)  "
              f"{metrics['precision']:.4f}      "
              f"{metrics['recall']:.4f}      "
              f"{metrics['f1_score']:.4f}      "
              f"{metrics['false_alarm_rate']:.4f} ({metrics['false_alarm_rate']*100:5.2f}%)")
    
    print("\nKey Insights:")
    best_acc = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'])
    lowest_far = min(results.items(), key=lambda x: x[1]['false_alarm_rate'])
    
    print(f"  - Best Accuracy: {best_acc[0]} ({best_acc[1]['accuracy']*100:.2f}%)")
    print(f"  - Best F1 Score: {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
    print(f"  - Lowest False Alarm Rate: {lowest_far[0]} ({lowest_far[1]['false_alarm_rate']*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Test Ensemble on Kaggle Dataset")
    
    # Data
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        default=Path("data/KaggleRealDataset"),
        help="Directory with Kaggle dataset (Fall/ and No_Fall/ subdirs)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=64,
        help="Sequence length",
    )
    
    # Model
    parser.add_argument(
        "--ensemble-config",
        type=Path,
        default=Path("models/ensemble_config.json"),
        help="Path to ensemble configuration",
    )
    parser.add_argument(
        "--neural-model",
        type=Path,
        default=None,
        help="Path to trained neural model (optional)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    
    # Options
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to test (for quick testing)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ENSEMBLE FALL DETECTOR - KAGGLE DATASET TEST")
    print("=" * 80)
    print()
    
    # Load ensemble configuration
    if args.ensemble_config.exists():
        print(f"Loading ensemble configuration from {args.ensemble_config}...")
        with open(args.ensemble_config, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded:")
        print(f"  Voting strategy: {config['voting_strategy']}")
        print(f"  Detectors: {', '.join(config['detectors'])}")
        print(f"  Weights: {config['weights']}")
    else:
        print("No configuration found, using defaults")
        config = {
            'voting_strategy': 'weighted',
            'weights': {
                'physics': 0.30,
                'temporal': 0.25,
                'geometry': 0.20,
                'neural': 0.15,
                'anomaly': 0.10,
            }
        }
    
    # Load neural model if provided
    neural_model = None
    if args.neural_model and args.neural_model.exists():
        print(f"\nLoading neural model from {args.neural_model}...")
        neural_model = PoseAutoencoder()
        
        checkpoint = torch.load(args.neural_model, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            neural_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            neural_model.load_state_dict(checkpoint)
        
        neural_model = neural_model.to(args.device)
        neural_model.eval()
        print("Neural model loaded")
    elif config.get('neural_model_path'):
        neural_model_path = Path(config['neural_model_path'])
        if neural_model_path.exists():
            print(f"\nLoading neural model from config: {neural_model_path}...")
            neural_model = PoseAutoencoder()
            
            checkpoint = torch.load(neural_model_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                neural_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                neural_model.load_state_dict(checkpoint)
            
            neural_model = neural_model.to(args.device)
            neural_model.eval()
            print("Neural model loaded from config")
    
    if neural_model is None:
        print("\nNo neural model provided, using rule-based detectors only")
    
    # Create ensemble
    print("\nCreating ensemble detector...")
    ensemble = EnsembleFallDetector(
        neural_model=neural_model,
        device=args.device,
        voting_strategy=config['voting_strategy'],
        weights=config['weights']
    )
    print(f"Ensemble created with {len(ensemble.detectors)} detectors")
    
    # Load Kaggle dataset
    print("\n" + "=" * 80)
    print("LOADING KAGGLE DATASET")
    print("=" * 80)
    sequences, labels = load_kaggle_dataset(args.kaggle_dir, args.sequence_length)
    
    if len(sequences) == 0:
        print("ERROR: No sequences loaded!")
        return
    
    # Limit samples if requested
    if args.max_samples and len(sequences) > args.max_samples:
        print(f"\nLimiting to {args.max_samples} samples for quick testing...")
        indices = np.random.choice(len(sequences), args.max_samples, replace=False)
        sequences = [sequences[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    # Test ensemble
    print("\n" + "=" * 80)
    print("TESTING ENSEMBLE")
    print("=" * 80)
    results = test_ensemble(ensemble, sequences, labels)
    
    # Print ensemble results
    print_results(results, 'ensemble')
    
    # Print individual detector results
    for detector_name in sorted(ensemble.detectors.keys()):
        if detector_name in results:
            print_results(results, detector_name)
    
    # Print comparison
    compare_all_results(results)
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_to_save = {}
    for name, metrics in results.items():
        results_to_save[name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'fall_detection_rate': float(metrics['fall_detection_rate']),
            'false_alarm_rate': float(metrics['false_alarm_rate']),
            'confusion_matrix': metrics['confusion_matrix'],
        }
    
    results_file = args.output_dir / "ensemble_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Save summary report
    summary_file = args.output_dir / "ENSEMBLE_TEST_REPORT.md"
    with open(summary_file, 'w') as f:
        f.write("# Ensemble Fall Detector - Test Results\n\n")
        f.write(f"## Dataset\n\n")
        f.write(f"- Total sequences: {len(sequences)}\n")
        f.write(f"- Falls: {sum(labels)}\n")
        f.write(f"- No Falls: {len(labels) - sum(labels)}\n")
        f.write(f"- Sequence length: {args.sequence_length}\n\n")
        
        f.write(f"## Ensemble Configuration\n\n")
        f.write(f"- Voting strategy: {config['voting_strategy']}\n")
        f.write(f"- Detectors: {', '.join(config['detectors'])}\n")
        f.write(f"- Weights:\n")
        for name, weight in config['weights'].items():
            f.write(f"  - {name}: {weight:.2f}\n")
        f.write("\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Detector | Accuracy | Precision | Recall | F1 | FAR |\n")
        f.write("|----------|----------|-----------|--------|-------|-----|\n")
        
        detector_names = sorted(results.keys(), key=lambda x: results[x]['accuracy'], reverse=True)
        for name in detector_names:
            metrics = results[name]
            f.write(f"| {name} | {metrics['accuracy']*100:.2f}% | "
                   f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                   f"{metrics['f1_score']:.4f} | {metrics['false_alarm_rate']*100:.2f}% |\n")
        
        f.write("\n## Key Insights\n\n")
        best_acc = max(results.items(), key=lambda x: x[1]['accuracy'])
        best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'])
        lowest_far = min(results.items(), key=lambda x: x[1]['false_alarm_rate'])
        
        f.write(f"- **Best Accuracy**: {best_acc[0]} ({best_acc[1]['accuracy']*100:.2f}%)\n")
        f.write(f"- **Best F1 Score**: {best_f1[0]} ({best_f1[1]['f1_score']:.4f})\n")
        f.write(f"- **Lowest False Alarm Rate**: {lowest_far[0]} ({lowest_far[1]['false_alarm_rate']*100:.2f}%)\n\n")
        
        f.write("## Confusion Matrices\n\n")
        for name in detector_names:
            cm = np.array(results[name]['confusion_matrix'])
            f.write(f"### {name}\n\n")
            f.write("```\n")
            f.write("                Predicted\n")
            f.write("              No Fall    Fall\n")
            f.write(f"Actual No Fall  {cm[0,0]:<7}  {cm[0,1]:<7}\n")
            f.write(f"       Fall     {cm[1,0]:<7}  {cm[1,1]:<7}\n")
            f.write("```\n\n")
    
    print(f"Summary report saved to {summary_file}")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
