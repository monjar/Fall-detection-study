"""
Test ensemble detector with neural network component included.

This tests the enhanced ensemble mode with 5 detectors:
- Physics-based
- Temporal pattern
- Pose geometry
- Neural network (CNN/LSTM)
- Anomaly detector

Compared to the basic ensemble (3 detectors only).
"""
import sys
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ensemble_detector import EnsembleFallDetector
from models.cnn_model import PoseAutoencoder
from utils.pose_processing import normalize_pose_sequence, pad_or_trim
import pandas as pd


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


def load_neural_model(model_path: Path, device: str = 'cuda'):
    """Load trained neural model for ensemble."""
    print(f"Loading neural model from {model_path}...")
    
    # Try to load as improved hybrid model first
    try:
        # Import hybrid model class
        class ImprovedHybridDetectorLSTM(torch.nn.Module):
            def __init__(self, pretrained_autoencoder=None, lstm_hidden=128, lstm_layers=2):
                super().__init__()
                if pretrained_autoencoder is not None:
                    self.autoencoder = pretrained_autoencoder
                else:
                    self.autoencoder = PoseAutoencoder()
                
                self.lstm = torch.nn.LSTM(
                    input_size=128,
                    hidden_size=lstm_hidden,
                    num_layers=lstm_layers,
                    batch_first=True,
                    dropout=0.3 if lstm_layers > 1 else 0,
                    bidirectional=True
                )
                
                lstm_output_size = lstm_hidden * 2
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(lstm_output_size, 128),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.4),
                    torch.nn.Linear(128, 64),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.4),
                    torch.nn.Linear(64, 2),
                )
            
            def forward(self, x):
                B, C, T, J = x.shape
                reconstructed_frames = []
                latent_frames = []
                
                for t in range(T):
                    frame = x[:, :, t:t+1, :]
                    recon_frame, latent_frame = self.autoencoder(frame)
                    recon_frame = recon_frame.squeeze(2)
                    reconstructed_frames.append(recon_frame)
                    latent_frames.append(latent_frame)
                
                reconstructed = torch.stack(reconstructed_frames, dim=2)
                latent_sequence = torch.stack(latent_frames, dim=1)
                
                lstm_out, _ = self.lstm(latent_sequence)
                lstm_final = lstm_out[:, -1, :]
                logits = self.classifier(lstm_final)
                
                return reconstructed, logits
        
        model = ImprovedHybridDetectorLSTM()
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if checkpoint is a dict with model_state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        print(f"Loaded ImprovedHybridDetectorLSTM model")
        
    except Exception as e:
        print(f"Failed to load as hybrid model: {e}")
        print("Trying to load as basic autoencoder...")
        
        # Try basic autoencoder
        model = PoseAutoencoder()
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if checkpoint is a dict with model_state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print(f"Loaded PoseAutoencoder model")
    
    model = model.to(device)
    model.eval()
    return model


def extract_poses_from_video(video_path: Path, max_frames: int = 64):
    """Extract pose keypoints from video using OpenCV OpenPose."""
    # Check for OpenPose models
    model_dir = PROJECT_ROOT / "models" / "opencv_pose"
    prototxt = model_dir / "pose_deploy_linevec.prototxt"
    weights = model_dir / "pose_iter_440000.caffemodel"
    
    if not prototxt.exists() or not weights.exists():
        raise FileNotFoundError(f"OpenPose models not found in {model_dir}")
    
    # Load OpenPose
    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    keypoints_list = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        
        # Prepare input for OpenPose
        h, w = frame.shape[:2]
        inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (192, 192),
                                    (0, 0, 0), swapRB=False, crop=False)
        
        net.setInput(inp)
        output = net.forward()
        
        # Extract keypoints
        points = []
        for i in range(18):  # 18 keypoints in COCO model
            prob_map = output[0, i, :, :]
            _, prob, _, point = cv2.minMaxLoc(prob_map)
            
            if prob > 0.1:
                x = int(point[0] * w / output.shape[3])
                y = int(point[1] * h / output.shape[2])
                points.append([x, y])
            else:
                points.append([0, 0])
        
        keypoints_list.append(points)
        frame_count += 1
    
    cap.release()
    
    if len(keypoints_list) == 0:
        raise RuntimeError(f"No keypoints extracted from {video_path}")
    
    return np.array(keypoints_list)


def test_on_kaggle_dataset(ensemble: EnsembleFallDetector, 
                          dataset_root: Path,
                          num_samples: int = 500):
    """Test ensemble on Kaggle dataset."""
    print(f"\nTesting on Kaggle dataset (up to {num_samples} samples per class)...")
    
    # Collect CSV paths
    fall_dir = dataset_root / "Fall" / "Keypoints_CSV"
    no_fall_dir = dataset_root / "No_Fall" / "Keypoints_CSV"
    
    fall_csvs = list(fall_dir.glob("*.csv"))[:num_samples]
    no_fall_csvs = list(no_fall_dir.glob("*.csv"))[:num_samples]
    
    print(f"Found {len(fall_csvs)} fall sequences")
    print(f"Found {len(no_fall_csvs)} no-fall sequences")
    
    all_files = [(v, 1) for v in fall_csvs] + [(v, 0) for v in no_fall_csvs]
    
    # Run predictions
    predictions = []
    ground_truth = []
    individual_results = {
        'physics': {'predictions': [], 'confidences': []},
        'temporal': {'predictions': [], 'confidences': []},
        'geometry': {'predictions': [], 'confidences': []},
        'neural': {'predictions': [], 'confidences': []},
        'anomaly': {'predictions': [], 'confidences': []},
    }
    
    print("\nProcessing sequences...")
    for csv_path, label in tqdm(all_files, desc="Testing"):
        try:
            # Load keypoints from CSV
            keypoints_padded = load_kaggle_csv_sequence(csv_path, 64)
            
            if keypoints_padded is None:
                continue
            
            # Predict with ensemble
            pred, conf, individual_preds = ensemble.predict(keypoints_padded)
            
            predictions.append(pred)
            ground_truth.append(label)
            
            # Store individual predictions
            for detector_name, (det_pred, det_conf) in individual_preds.items():
                if detector_name in individual_results:
                    individual_results[detector_name]['predictions'].append(det_pred)
                    individual_results[detector_name]['confidences'].append(det_conf)
            
        except Exception as e:
            print(f"\nError processing {csv_path.name}: {e}")
            continue
    
    # Calculate metrics
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Ensemble metrics
    accuracy = np.mean(predictions == ground_truth)
    
    # Fall detection rate (recall)
    fall_mask = ground_truth == 1
    fall_detection_rate = np.mean(predictions[fall_mask] == 1) if np.any(fall_mask) else 0
    
    # False alarm rate
    no_fall_mask = ground_truth == 0
    false_alarm_rate = np.mean(predictions[no_fall_mask] == 1) if np.any(no_fall_mask) else 0
    
    # Precision
    pred_fall_mask = predictions == 1
    precision = np.mean(ground_truth[pred_fall_mask] == 1) if np.any(pred_fall_mask) else 0
    
    # F1 score
    if precision + fall_detection_rate > 0:
        f1_score = 2 * (precision * fall_detection_rate) / (precision + fall_detection_rate)
    else:
        f1_score = 0
    
    # Confusion matrix
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    tn = np.sum((predictions == 0) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    
    ensemble_results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(fall_detection_rate),
        'f1_score': float(f1_score),
        'fall_detection_rate': float(fall_detection_rate),
        'false_alarm_rate': float(false_alarm_rate),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'num_samples': len(predictions),
    }
    
    # Calculate metrics for individual detectors
    results = {'ensemble_with_neural': ensemble_results}
    
    for detector_name, data in individual_results.items():
        if len(data['predictions']) > 0:
            det_preds = np.array(data['predictions'])
            det_acc = np.mean(det_preds == ground_truth)
            det_recall = np.mean(det_preds[fall_mask] == 1) if np.any(fall_mask) else 0
            det_fpr = np.mean(det_preds[no_fall_mask] == 1) if np.any(no_fall_mask) else 0
            det_pred_fall = det_preds == 1
            det_precision = np.mean(ground_truth[det_pred_fall] == 1) if np.any(det_pred_fall) else 0
            
            if det_precision + det_recall > 0:
                det_f1 = 2 * (det_precision * det_recall) / (det_precision + det_recall)
            else:
                det_f1 = 0
            
            tp = np.sum((det_preds == 1) & (ground_truth == 1))
            fp = np.sum((det_preds == 1) & (ground_truth == 0))
            tn = np.sum((det_preds == 0) & (ground_truth == 0))
            fn = np.sum((det_preds == 0) & (ground_truth == 1))
            
            results[detector_name] = {
                'accuracy': float(det_acc),
                'precision': float(det_precision),
                'recall': float(det_recall),
                'f1_score': float(det_f1),
                'fall_detection_rate': float(det_recall),
                'false_alarm_rate': float(det_fpr),
                'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
            }
    
    return results


def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description='Test ensemble with neural network')
    parser.add_argument('--model', type=Path, 
                       default=PROJECT_ROOT / 'models' / 'improved_hybrid_detector.pt',
                       help='Path to neural model')
    parser.add_argument('--kaggle-root', type=Path,
                       default=PROJECT_ROOT / 'data' / 'KaggleRealDataset',
                       help='Path to Kaggle dataset')
    parser.add_argument('--num-samples', type=int, default=500,
                       help='Number of samples per class to test')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output', type=Path,
                       default=PROJECT_ROOT / 'results' / 'ensemble_with_neural_test_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TESTING ENSEMBLE WITH NEURAL NETWORK")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.kaggle_root}")
    print()
    
    # Load neural model
    neural_model = load_neural_model(args.model, args.device)
    
    # Create ensemble with neural network
    print("\nCreating ensemble detector with neural network...")
    ensemble = EnsembleFallDetector(
        neural_model=neural_model,
        device=args.device,
        voting_strategy='weighted',
        weights={
            'physics': 0.25,
            'temporal': 0.20,
            'geometry': 0.20,
            'neural': 0.20,
            'anomaly': 0.15,
        }
    )
    
    print(f"Ensemble created with {len(ensemble.detectors)} detectors:")
    for name in ensemble.detectors.keys():
        print(f"  - {name} (weight: {ensemble.weights.get(name, 0.0):.2f})")
    
    # Test on Kaggle dataset
    results = test_on_kaggle_dataset(ensemble, args.kaggle_root, args.num_samples)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for detector_name, metrics in results.items():
        print(f"\n{detector_name.upper()}:")
        print(f"  Accuracy:           {metrics['accuracy']:.2%}")
        print(f"  Precision:          {metrics['precision']:.2%}")
        print(f"  Recall:             {metrics['recall']:.2%}")
        print(f"  F1 Score:           {metrics['f1_score']:.4f}")
        print(f"  Fall Detection:     {metrics['fall_detection_rate']:.2%}")
        print(f"  False Alarm Rate:   {metrics['false_alarm_rate']:.2%}")
        print(f"  Confusion Matrix:")
        print(f"    [[{metrics['confusion_matrix'][0][0]:4d}, {metrics['confusion_matrix'][0][1]:4d}]")
        print(f"     [{metrics['confusion_matrix'][1][0]:4d}, {metrics['confusion_matrix'][1][1]:4d}]]")
    
    # Save results
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()
