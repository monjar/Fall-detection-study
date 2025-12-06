"""
Train and evaluate ensemble fall detection system.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_model import PoseAutoencoder
from models.ensemble_detector import EnsembleFallDetector
from utils.pose_processing import normalize_pose_sequence, pad_or_trim


def load_kth_sequences(data_dir: Path, sequence_length: int = 64,
                       max_sequences: int = None) -> List[np.ndarray]:
    """Load KTH normal sequences for calibration."""
    npz_files = sorted(data_dir.glob("*.npz"))
    
    if max_sequences:
        npz_files = npz_files[:max_sequences]
    
    sequences = []
    print(f"Loading {len(npz_files)} KTH sequences...")
    
    for npz_file in tqdm(npz_files, desc="Loading KTH"):
        try:
            data = np.load(npz_file)
            
            if 'poses' in data:
                pose_seq = data['poses']
            elif 'keypoints' in data:
                pose_seq = data['keypoints']
            elif 'pose_sequence' in data:
                pose_seq = data['pose_sequence']
            else:
                continue
            
            pose_seq = normalize_pose_sequence(pose_seq)
            pose_seq = pad_or_trim(pose_seq, sequence_length)
            sequences.append(pose_seq)
            
        except Exception as e:
            continue
    
    print(f"Loaded {len(sequences)} sequences")
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Train Ensemble Fall Detector")
    
    # Data
    parser.add_argument(
        "--kth-dir",
        type=Path,
        default=Path("data/pose_keypoints/npz"),
        help="Directory with KTH NPZ files",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=64,
        help="Sequence length",
    )
    parser.add_argument(
        "--max-kth-sequences",
        type=int,
        default=200,
        help="Max KTH sequences to use for calibration",
    )
    
    # Model
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
    
    # Ensemble
    parser.add_argument(
        "--voting-strategy",
        choices=['weighted', 'majority', 'soft'],
        default='weighted',
        help="Voting strategy for ensemble",
    )
    parser.add_argument(
        "--physics-weight",
        type=float,
        default=0.30,
        help="Weight for physics detector",
    )
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=0.25,
        help="Weight for temporal detector",
    )
    parser.add_argument(
        "--geometry-weight",
        type=float,
        default=0.20,
        help="Weight for geometry detector",
    )
    parser.add_argument(
        "--neural-weight",
        type=float,
        default=0.15,
        help="Weight for neural detector",
    )
    parser.add_argument(
        "--anomaly-weight",
        type=float,
        default=0.10,
        help="Weight for anomaly detector",
    )
    
    # Output
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("models/ensemble_config.json"),
        help="Path to save ensemble configuration",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ENSEMBLE FALL DETECTOR SETUP")
    print("=" * 80)
    print()
    
    # Load neural model if provided
    neural_model = None
    if args.neural_model and args.neural_model.exists():
        print(f"Loading neural model from {args.neural_model}...")
        neural_model = PoseAutoencoder()
        
        checkpoint = torch.load(args.neural_model, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            neural_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            neural_model.load_state_dict(checkpoint)
        
        neural_model = neural_model.to(args.device)
        neural_model.eval()
        print("Neural model loaded")
    else:
        print("No neural model provided, using rule-based detectors only")
    
    # Create ensemble
    print("\nCreating ensemble detector...")
    weights = {
        'physics': args.physics_weight,
        'temporal': args.temporal_weight,
        'geometry': args.geometry_weight,
        'neural': args.neural_weight,
        'anomaly': args.anomaly_weight,
    }
    
    ensemble = EnsembleFallDetector(
        neural_model=neural_model,
        device=args.device,
        voting_strategy=args.voting_strategy,
        weights=weights
    )
    
    print(f"Ensemble created with {len(ensemble.detectors)} detectors:")
    for name in ensemble.detectors.keys():
        weight = weights.get(name, 0.0)
        print(f"  - {name}: weight={weight:.2f}")
    
    # Load KTH sequences for calibration
    print("\nLoading KTH sequences for calibration...")
    normal_sequences = load_kth_sequences(
        args.kth_dir,
        sequence_length=args.sequence_length,
        max_sequences=args.max_kth_sequences
    )
    
    # Calibrate anomaly detector
    if 'anomaly' in ensemble.detectors:
        print("\nCalibrating anomaly detector...")
        ensemble.calibrate_anomaly_detector(normal_sequences)
    
    # Save configuration
    config = {
        'voting_strategy': args.voting_strategy,
        'weights': weights,
        'sequence_length': args.sequence_length,
        'neural_model_path': str(args.neural_model) if args.neural_model else None,
        'detectors': list(ensemble.detectors.keys()),
    }
    
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nEnsemble configuration saved to {args.output_config}")
    
    # Test on a sample sequence
    print("\nTesting ensemble on sample sequence...")
    if len(normal_sequences) > 0:
        sample_seq = normal_sequences[0]
        prediction, confidence, individual_preds = ensemble.predict(sample_seq)
        
        print(f"\nSample prediction (normal sequence):")
        print(f"  Ensemble: prediction={prediction}, confidence={confidence:.3f}")
        print(f"  Individual detectors:")
        for name, (pred, conf) in individual_preds.items():
            print(f"    - {name}: prediction={pred}, confidence={conf:.3f}")
    
    print("\n" + "=" * 80)
    print("ENSEMBLE SETUP COMPLETE")
    print("=" * 80)
    print(f"\nConfiguration saved to: {args.output_config}")
    print(f"Detectors: {', '.join(ensemble.detectors.keys())}")
    print(f"Voting strategy: {args.voting_strategy}")
    print("\nReady to test on Kaggle dataset!")


if __name__ == "__main__":
    main()
