"""
Debug script to test fall detection on sample videos and understand why
all videos are being classified as falls.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.cwd()))

from fall_detector_main import FallDetectorMain
from models.ensemble_detector import PhysicsBasedDetector

# Test with a simple synthetic "normal" video
print("=" * 80)
print("DEBUG: Testing Fall Detection Logic")
print("=" * 80)

# Create synthetic "normal walking" keypoints
# Person standing/walking - minimal vertical movement
print("\n1. Testing Physics Detector with Synthetic Data")
print("-" * 80)

T = 64  # frames
J = 18  # joints

# Normal walking: person upright, small movements
normal_keypoints = np.random.randn(T, J, 2) * 0.05  # Small random noise
normal_keypoints[:, :, 0] += 0.5  # Center X around 0.5
normal_keypoints[:, :, 1] += 0.5  # Center Y around 0.5

# Add slight up-down motion for walking
normal_keypoints[:, :, 1] += 0.02 * np.sin(np.linspace(0, 4*np.pi, T))[:, np.newaxis]

physics = PhysicsBasedDetector()
features = physics.extract_features(normal_keypoints)

print(f"Normal Walking Features:")
for key, value in features.items():
    print(f"  {key:20s}: {value:.4f}")

pred, conf = physics.predict(normal_keypoints)
print(f"\nPrediction: {'FALL' if pred == 1 else 'NO FALL'}")
print(f"Confidence: {conf:.2%}")

# Fall scenario: large downward movement
print("\n" + "=" * 80)
print("2. Testing with Synthetic Fall")
print("-" * 80)

fall_keypoints = np.copy(normal_keypoints)
# Simulate fall: rapid downward motion in second half
fall_start = T // 2
fall_keypoints[fall_start:, :, 1] += np.linspace(0, 0.5, T - fall_start)[:, np.newaxis]

features_fall = physics.extract_features(fall_keypoints)

print(f"Fall Features:")
for key, value in features_fall.items():
    print(f"  {key:20s}: {value:.4f}")

pred_fall, conf_fall = physics.predict(fall_keypoints)
print(f"\nPrediction: {'FALL' if pred_fall == 1 else 'NO FALL'}")
print(f"Confidence: {conf_fall:.2%}")

# Test ensemble
print("\n" + "=" * 80)
print("3. Testing Ensemble Detector")
print("-" * 80)

try:
    from models.ensemble_detector import EnsembleFallDetector
    
    ensemble = EnsembleFallDetector(
        neural_model=None,
        device='cpu',
        voting_strategy='weighted',
        weights={'physics': 0.30, 'temporal': 0.25, 'geometry': 0.20}
    )
    
    print("\nNormal Walking:")
    pred_ens, conf_ens, individual = ensemble.predict(normal_keypoints)
    print(f"Prediction: {'FALL' if pred_ens == 1 else 'NO FALL'} (confidence: {conf_ens:.2%})")
    print(f"Individual predictions:")
    for name, (p, c) in individual.items():
        print(f"  {name:12s}: {'FALL' if p == 1 else 'NO FALL'} ({c:.2%})")
    
    print("\nFall:")
    pred_ens_fall, conf_ens_fall, individual_fall = ensemble.predict(fall_keypoints)
    print(f"Prediction: {'FALL' if pred_ens_fall == 1 else 'NO FALL'} (confidence: {conf_ens_fall:.2%})")
    print(f"Individual predictions:")
    for name, (p, c) in individual_fall.items():
        print(f"  {name:12s}: {'FALL' if p == 1 else 'NO FALL'} ({c:.2%})")
    
except Exception as e:
    print(f"Error testing ensemble: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("4. Diagnosis")
print("=" * 80)
print("""
If both normal and fall scenarios predict FALL, the issue could be:
1. Thresholds are too sensitive (too low)
2. Normalization is incorrect (keypoints not in expected range)
3. OpenPose extraction producing unusual keypoint distributions
4. The "normal" videos you're testing have significant motion

To fix:
- Check your video keypoint distributions
- Adjust thresholds in PhysicsBasedDetector
- Ensure keypoints are properly normalized
""")
