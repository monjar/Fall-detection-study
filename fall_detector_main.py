"""
Main script for end-to-end fall detection from video.

Usage:
    # Using ensemble detector (no model needed)
    python fall_detector_main.py --video path/to/video.mp4 --detector ensemble
    
    # Using trained neural network model
    python fall_detector_main.py --video path/to/video.mp4 --detector neural --model models/best_model.pth
    
    # Using specific ensemble component
    python fall_detector_main.py --video path/to/video.mp4 --detector physics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_model import PoseAutoencoder
from models.ensemble_detector import (
    EnsembleFallDetector,
    PhysicsBasedDetector,
    TemporalPatternDetector,
    PoseGeometryDetector,
)
from utils.pose_processing import normalize_pose_sequence, pad_or_trim


# Import hybrid model classes
class ImprovedHybridDetectorLSTM(nn.Module):
    """Improved hybrid model with LSTM temporal modeling."""
    
    def __init__(self, pretrained_autoencoder: Optional[PoseAutoencoder] = None,
                 lstm_hidden: int = 128, lstm_layers: int = 2):
        super().__init__()
        
        if pretrained_autoencoder is not None:
            self.autoencoder = pretrained_autoencoder
        else:
            self.autoencoder = PoseAutoencoder()
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_size = lstm_hidden * 2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 2),
        )
        
    def forward(self, x: torch.Tensor):
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


class OpenPoseExtractor:
    """Extract pose keypoints from video using OpenCV OpenPose."""
    
    def __init__(self, model_dir: Path = None):
        """Initialize OpenPose model."""
        if model_dir is None:
            model_dir = PROJECT_ROOT / "models" / "opencv_pose"
        
        self.prototxt = model_dir / "pose_deploy_linevec.prototxt"
        self.weights = model_dir / "pose_iter_440000.caffemodel"
        
        if not self.prototxt.exists() or not self.weights.exists():
            raise FileNotFoundError(
                f"OpenPose models not found in {model_dir}. "
                f"Run: python scripts/download_pose_models.py"
            )
        
        print(f"Loading OpenPose model from {model_dir}...")
        self.net = cv2.dnn.readNetFromCaffe(str(self.prototxt), str(self.weights))
        
        # Use GPU if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using CUDA for OpenPose")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU for OpenPose")
        
        # OpenPose COCO model has 18 keypoints
        self.n_points = 18
        self.threshold = 0.1
    
    def extract_from_video(self, video_path: Path, 
                          max_frames: Optional[int] = None,
                          skip_frames: int = 1) -> np.ndarray:
        """
        Extract pose keypoints from video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (None = all)
            skip_frames: Process every Nth frame (1 = all frames, 2 = every other, etc.)
        
        Returns:
            np.ndarray: Shape (num_frames, 18, 2) with normalized keypoints
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nVideo info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f}s")
        
        if max_frames:
            frames_to_process = min(max_frames, total_frames)
        else:
            frames_to_process = total_frames
        
        print(f"  Processing: {frames_to_process // skip_frames} frames (skip={skip_frames})")
        
        keypoints_sequence = []
        frame_idx = 0
        processed = 0
        
        print("\nExtracting poses...")
        while cap.isOpened() and (max_frames is None or processed < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if requested
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue
            
            # Extract keypoints from this frame
            keypoints = self._extract_from_frame(frame, width, height)
            keypoints_sequence.append(keypoints)
            
            processed += 1
            if processed % 10 == 0:
                print(f"  Processed {processed}/{frames_to_process // skip_frames} frames", end='\r')
            
            frame_idx += 1
        
        cap.release()
        print(f"\n  Extracted {len(keypoints_sequence)} frames")
        
        if len(keypoints_sequence) == 0:
            raise RuntimeError("No keypoints extracted from video")
        
        return np.array(keypoints_sequence)
    
    def _extract_from_frame(self, frame: np.ndarray, 
                           width: int, height: int) -> np.ndarray:
        """Extract keypoints from a single frame."""
        # Prepare input blob
        input_blob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False
        )
        self.net.setInput(input_blob)
        
        # Forward pass
        output = self.net.forward()
        
        # Extract keypoints
        keypoints = np.zeros((self.n_points, 2))
        
        for i in range(self.n_points):
            # Get heatmap for this keypoint
            prob_map = output[0, i, :, :]
            
            # Find maximum
            _, prob, _, point = cv2.minMaxLoc(prob_map)
            
            # Scale to original image size
            x = (width * point[0]) / output.shape[3]
            y = (height * point[1]) / output.shape[2]
            
            if prob > self.threshold:
                keypoints[i] = [x, y]
            else:
                keypoints[i] = [0, 0]  # Not detected
        
        return keypoints


class FallDetectorMain:
    """Main fall detector that coordinates video processing and inference."""
    
    def __init__(self, detector_type: str = 'ensemble',
                 model_path: Optional[Path] = None,
                 device: str = 'cpu',
                 sequence_length: int = 64,
                 ensemble_config: Optional[Path] = None):
        """
        Initialize fall detector.
        
        Args:
            detector_type: Type of detector ('ensemble', 'neural', 'physics', 'temporal', 'geometry')
            model_path: Path to trained model (for neural detector)
            device: Device to run on ('cpu' or 'cuda')
            sequence_length: Number of frames to use for detection
            ensemble_config: Path to ensemble configuration JSON
        """
        self.detector_type = detector_type
        self.device = device
        self.sequence_length = sequence_length
        
        print(f"\nInitializing {detector_type} detector...")
        
        # Initialize OpenPose
        self.pose_extractor = OpenPoseExtractor()
        
        # Initialize detector
        if detector_type == 'neural':
            if model_path is None:
                raise ValueError("model_path required for neural detector")
            self.detector = self._load_neural_model(model_path)
            
        elif detector_type == 'ensemble':
            self.detector = self._load_ensemble_detector(model_path, ensemble_config)
            
        elif detector_type == 'physics':
            self.detector = PhysicsBasedDetector()
            
        elif detector_type == 'temporal':
            self.detector = TemporalPatternDetector()
            
        elif detector_type == 'geometry':
            self.detector = PoseGeometryDetector()
            
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        print("Detector initialized successfully!")
    
    def _load_neural_model(self, model_path: Path) -> torch.nn.Module:
        """Load trained neural network model."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading neural model from {model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Detect model type by checking state_dict keys or config
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
        else:
            state_dict = checkpoint
            config = {}
        
        # Check if it's an LSTM hybrid model
        is_lstm_model = any('lstm' in key for key in state_dict.keys())
        
        # Check if keys are prefixed (e.g., "autoencoder.encoder.0.weight")
        has_autoencoder_prefix = any(key.startswith('autoencoder.') for key in state_dict.keys())
        has_classifier = any(key.startswith('classifier.') for key in state_dict.keys())
        
        if is_lstm_model:
            print("Detected LSTM hybrid model")
            lstm_hidden = config.get('lstm_hidden', 128)
            lstm_layers = config.get('lstm_layers', 2)
            model = ImprovedHybridDetectorLSTM(
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers
            )
        elif has_autoencoder_prefix or has_classifier:
            print("Detected hybrid model with prefixed keys")
            # This is a hybrid model, we need to extract only the autoencoder part
            model = PoseAutoencoder()
            
            # Remove prefixes and classifier keys to get just the autoencoder
            clean_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('autoencoder.'):
                    # Remove 'autoencoder.' prefix
                    new_key = key[len('autoencoder.'):]
                    clean_state_dict[new_key] = value
                elif not key.startswith('classifier.'):
                    # Keep keys that aren't classifier keys
                    clean_state_dict[key] = value
            
            state_dict = clean_state_dict
        else:
            print("Detected simple autoencoder model")
            model = PoseAutoencoder()
        
        # Load weights
        try:
            if 'model_state_dict' in checkpoint and not has_autoencoder_prefix:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Warning: Could not load model directly: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(state_dict, strict=False)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def _load_ensemble_detector(self, model_path: Optional[Path],
                                ensemble_config: Optional[Path]) -> EnsembleFallDetector:
        """Load ensemble detector."""
        # Load configuration if provided
        if ensemble_config and ensemble_config.exists():
            print(f"Loading ensemble config from {ensemble_config}...")
            with open(ensemble_config, 'r') as f:
                config = json.load(f)
            weights = config.get('weights', {})
            voting_strategy = config.get('voting_strategy', 'weighted')
        else:
            print("Using default ensemble configuration...")
            weights = {
                'physics': 0.30,
                'temporal': 0.25,
                'geometry': 0.20,
            }
            voting_strategy = 'weighted'
        
        # Load neural model if provided
        neural_model = None
        if model_path and model_path.exists():
            neural_model = self._load_neural_model(model_path)
        
        ensemble = EnsembleFallDetector(
            neural_model=neural_model,
            device=self.device,
            voting_strategy=voting_strategy,
            weights=weights
        )
        
        print(f"Ensemble created with {len(ensemble.detectors)} detectors")
        return ensemble
    
    def process_video(self, video_path: Path,
                     max_frames: Optional[int] = None,
                     skip_frames: int = 1) -> Dict:
        """
        Process video and detect falls.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            skip_frames: Process every Nth frame
        
        Returns:
            Dict with results:
                - prediction: 0 (no fall) or 1 (fall)
                - probability: Confidence score [0-1]
                - keypoints: Extracted pose sequence
                - individual_predictions: Per-detector results (if ensemble)
        """
        print("=" * 80)
        print(f"PROCESSING VIDEO: {video_path.name}")
        print("=" * 80)
        
        # Extract pose keypoints
        keypoints = self.pose_extractor.extract_from_video(
            video_path, max_frames, skip_frames
        )
        
        # Normalize and prepare sequence
        print("\nPreparing sequence...")
        keypoints_normalized = normalize_pose_sequence(keypoints)
        keypoints_padded = pad_or_trim(keypoints_normalized, self.sequence_length)
        
        print(f"  Input shape: {keypoints_padded.shape}")
        
        # Run inference
        print(f"\nRunning {self.detector_type} detector...")
        result = self._run_inference(keypoints_padded)
        
        # Print results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Prediction: {'FALL' if result['prediction'] == 1 else 'NO FALL'}")
        print(f"Confidence: {result['probability']:.2%}")
        
        if 'individual_predictions' in result:
            print("\nIndividual detector predictions:")
            for name, (pred, conf) in result['individual_predictions'].items():
                status = 'FALL' if pred == 1 else 'NO FALL'
                print(f"  {name:12s}: {status:8s} (confidence: {conf:.2%})")
        
        print("=" * 80)
        
        return result
    
    def _run_inference(self, keypoints: np.ndarray) -> Dict:
        """Run inference on keypoints."""
        if self.detector_type == 'neural':
            return self._run_neural_inference(keypoints)
        elif self.detector_type == 'ensemble':
            return self._run_ensemble_inference(keypoints)
        else:
            return self._run_rule_based_inference(keypoints)
    
    def _run_neural_inference(self, keypoints: np.ndarray) -> Dict:
        """Run neural network inference."""
        # Convert to tensor: keypoints is (T, J, 2)
        # Model expects (B, 2, T, J) for LSTM or (B, 2, T, J) for autoencoder
        x = torch.FloatTensor(keypoints).unsqueeze(0)  # (1, T, J, 2)
        x = x.permute(0, 3, 1, 2)  # (1, 2, T, J)
        x = x.to(self.device)
        
        with torch.no_grad():
            # Check if it's LSTM model (has lstm attribute)
            if hasattr(self.detector, 'lstm'):
                # LSTM model returns (reconstructed, logits)
                reconstructed, logits = self.detector(x)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                prob = probs[0, 1].item()  # Probability of fall class
            else:
                # Simple autoencoder - use reconstruction error
                reconstructed, latent = self.detector(x)
                recon_error = torch.mean((x - reconstructed) ** 2, dim=[1, 2, 3]).item()
                
                # Load calibrated threshold if available
                threshold_file = PROJECT_ROOT / "results" / "anomaly_threshold.json"
                if threshold_file.exists():
                    with open(threshold_file, 'r') as f:
                        threshold_data = json.load(f)
                    threshold = threshold_data['threshold']  # ~4.93
                else:
                    # Fallback threshold (calibrated for typical autoencoder)
                    threshold = 5.0
                
                # Predict: high reconstruction error = fall (anomaly)
                pred = 1 if recon_error > threshold else 0
                
                # Calculate confidence (how far from threshold)
                if pred == 1:
                    # Fall: confidence based on how much above threshold
                    prob = min(0.5 + (recon_error - threshold) / (threshold * 2), 1.0)
                else:
                    # No fall: confidence based on how much below threshold
                    prob = max(0.5 - (threshold - recon_error) / (threshold * 2), 0.0)
        
        return {
            'prediction': pred,
            'probability': prob,
            'keypoints': keypoints,
        }
    
    def _run_ensemble_inference(self, keypoints: np.ndarray) -> Dict:
        """Run ensemble inference."""
        prediction, confidence, individual_preds = self.detector.predict(keypoints)
        
        return {
            'prediction': prediction,
            'probability': confidence,
            'keypoints': keypoints,
            'individual_predictions': individual_preds,
        }
    
    def _run_rule_based_inference(self, keypoints: np.ndarray) -> Dict:
        """Run rule-based detector inference."""
        prediction, confidence = self.detector.predict(keypoints)
        
        return {
            'prediction': prediction,
            'probability': confidence,
            'keypoints': keypoints,
        }


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end fall detection from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using ensemble detector (recommended, no training needed)
  python fall_detector_main.py --video data/test_video.mp4 --detector ensemble
  
  # Using physics detector only (high recall)
  python fall_detector_main.py --video data/test_video.mp4 --detector physics
  
  # Using trained neural network
  python fall_detector_main.py --video data/test_video.mp4 --detector neural \\
      --model models/best_model.pth
  
  # Process subset of frames (faster)
  python fall_detector_main.py --video data/test_video.mp4 --detector ensemble \\
      --max-frames 64 --skip-frames 2
  
  # Save results to file
  python fall_detector_main.py --video data/test_video.mp4 --detector ensemble \\
      --output results/prediction.json
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--video',
        type=Path,
        required=True,
        help='Path to input video file'
    )
    
    # Detector configuration
    parser.add_argument(
        '--detector',
        choices=['ensemble', 'neural', 'physics', 'temporal', 'geometry'],
        default='ensemble',
        help='Type of detector to use (default: ensemble)'
    )
    parser.add_argument(
        '--model',
        type=Path,
        help='Path to trained model (required for neural detector)'
    )
    parser.add_argument(
        '--ensemble-config',
        type=Path,
        default=Path('models/ensemble_config.json'),
        help='Path to ensemble configuration JSON'
    )
    
    # Processing options
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=64,
        help='Number of frames for detection (default: 64)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames to process (default: all)'
    )
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=1,
        help='Process every Nth frame (default: 1 = all frames)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (default: auto-detect)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=Path,
        help='Path to save results JSON (optional)'
    )
    parser.add_argument(
        '--save-keypoints',
        type=Path,
        help='Path to save extracted keypoints NPZ (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.detector == 'neural' and args.model is None:
        parser.error("--model required when using neural detector")
    
    if not args.video.exists():
        parser.error(f"Video file not found: {args.video}")
    
    try:
        # Initialize detector
        detector = FallDetectorMain(
            detector_type=args.detector,
            model_path=args.model,
            device=args.device,
            sequence_length=args.sequence_length,
            ensemble_config=args.ensemble_config
        )
        
        # Process video
        result = detector.process_video(
            video_path=args.video,
            max_frames=args.max_frames,
            skip_frames=args.skip_frames
        )
        
        # Save results if requested
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare serializable result
            output_data = {
                'video': str(args.video),
                'detector': args.detector,
                'prediction': int(result['prediction']),
                'probability': float(result['probability']),
                'label': 'FALL' if result['prediction'] == 1 else 'NO FALL',
            }
            
            if 'individual_predictions' in result:
                output_data['individual_predictions'] = {
                    name: {'prediction': int(pred), 'confidence': float(conf)}
                    for name, (pred, conf) in result['individual_predictions'].items()
                }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\nResults saved to: {args.output}")
        
        # Save keypoints if requested
        if args.save_keypoints:
            args.save_keypoints.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_keypoints, keypoints=result['keypoints'])
            print(f"Keypoints saved to: {args.save_keypoints}")
        
        # Return exit code based on prediction
        sys.exit(0 if result['prediction'] == 0 else 1)
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
