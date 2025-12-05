"""Generate synthetic fall sequences from normal KTH activities."""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
import random


def extract_fall_features(pose_sequence: np.ndarray) -> dict:
    """
    Extract physics-based features that indicate falling.
    
    Args:
        pose_sequence: (T, J, 2) pose array
        
    Returns:
        Dictionary of fall-related features
    """
    T, J, _ = pose_sequence.shape
    
    # Handle missing data (zeros)
    pose_sequence = pose_sequence.copy()
    pose_sequence[pose_sequence == 0] = np.nan
    
    # Extract key joints
    # Assuming joint order: 0=Nose, 5=LShoulder, 6=RShoulder, 11=LHip, 12=RHip, 15=LAnkle, 16=RAnkle
    
    # Hip center (average of left and right hips)
    hip_center = np.nanmean(pose_sequence[:, [11, 12], :], axis=1)  # (T, 2)
    
    # Shoulder center
    shoulder_center = np.nanmean(pose_sequence[:, [5, 6], :], axis=1)  # (T, 2)
    
    # Ankle center
    ankle_center = np.nanmean(pose_sequence[:, [15, 16], :], axis=1)  # (T, 2)
    
    # 1. Vertical velocity (hip movement)
    hip_height = hip_center[:, 1]
    valid_frames = ~np.isnan(hip_height)
    
    if valid_frames.sum() < 2:
        return {
            'min_velocity': 0.0,
            'max_acceleration': 0.0,
            'body_angle': 90.0,
            'height_drop': 0.0,
            'aspect_ratio': 1.0,
            'is_valid': False
        }
    
    vertical_velocity = np.diff(hip_height[valid_frames])
    acceleration = np.diff(vertical_velocity) if len(vertical_velocity) > 1 else np.array([0.0])
    
    # 2. Body orientation angle (shoulder-hip line)
    body_vector = shoulder_center - hip_center  # (T, 2)
    body_angle = np.arctan2(body_vector[:, 1], body_vector[:, 0]) * 180 / np.pi  # Degrees
    body_angle = np.abs(body_angle)  # Absolute angle from horizontal
    
    # 3. Height drop (first to last)
    valid_indices = np.where(valid_frames)[0]
    if len(valid_indices) > 0:
        height_drop = hip_height[valid_indices[0]] - hip_height[valid_indices[-1]]
    else:
        height_drop = 0.0
    
    # 4. Bounding box aspect ratio
    all_points = pose_sequence.reshape(-1, 2)
    valid_points = all_points[~np.isnan(all_points).any(axis=1)]
    
    if len(valid_points) > 0:
        min_x, min_y = np.nanmin(valid_points, axis=0)
        max_x, max_y = np.nanmax(valid_points, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = height / width if width > 0 else 1.0
    else:
        aspect_ratio = 1.0
    
    return {
        'min_velocity': float(np.nanmin(vertical_velocity)) if len(vertical_velocity) > 0 else 0.0,
        'max_acceleration': float(np.nanmax(np.abs(acceleration))) if len(acceleration) > 0 else 0.0,
        'body_angle': float(np.nanmean(body_angle[-10:])) if len(body_angle) > 0 else 90.0,  # Last 10 frames
        'height_drop': float(height_drop),
        'aspect_ratio': float(aspect_ratio),
        'is_valid': True
    }


def is_fall_by_rules(features: dict, thresholds: Optional[dict] = None) -> bool:
    """
    Rule-based fall detection using physics features.
    
    Args:
        features: Dictionary from extract_fall_features()
        thresholds: Custom thresholds (optional)
        
    Returns:
        True if features indicate a fall
    """
    if not features['is_valid']:
        return False
    
    if thresholds is None:
        thresholds = {
            'min_velocity': -0.3,      # Fast downward movement
            'body_angle': 60.0,        # Body close to horizontal (< 30° from horizontal)
            'height_drop': 0.2,        # Significant vertical drop
            'aspect_ratio': 0.6,       # Wide bounding box (lying down)
        }
    
    # Fall indicators
    rapid_descent = features['min_velocity'] < thresholds['min_velocity']
    lying_down = features['body_angle'] < thresholds['body_angle']
    large_drop = features['height_drop'] > thresholds['height_drop']
    horizontal_pose = features['aspect_ratio'] < thresholds['aspect_ratio']
    
    # Fall = (rapid descent AND large drop) OR (lying down AND horizontal pose)
    is_fall = (rapid_descent and large_drop) or (lying_down and horizontal_pose)
    
    return is_fall


def generate_fall_from_standing(pose_sequence: np.ndarray, 
                                fall_duration: int = 20,
                                fall_start_frame: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Generate a synthetic fall sequence from a standing/walking pose sequence.
    
    Strategy:
    1. Find frames where person is upright
    2. Simulate gradual descent (lowering vertical positions)
    3. Rotate body toward horizontal orientation
    4. Add motion blur/instability
    
    Args:
        pose_sequence: (T, J, 2) original sequence
        fall_duration: Number of frames for the fall
        fall_start_frame: Which frame to start the fall (None = random)
        
    Returns:
        synthetic_fall: (T, J, 2) modified sequence with fall
        metadata: Information about the synthetic fall
    """
    T, J, _ = pose_sequence.shape
    synthetic_fall = pose_sequence.copy()
    
    if fall_start_frame is None:
        # Start fall somewhere in first half
        fall_start_frame = random.randint(T // 4, T // 2)
    
    fall_end_frame = min(fall_start_frame + fall_duration, T)
    
    # Extract hip center for reference
    hip_center = np.nanmean(pose_sequence[:, [11, 12], :], axis=1)
    
    # Create fall trajectory
    for t in range(fall_start_frame, fall_end_frame):
        progress = (t - fall_start_frame) / fall_duration  # 0 to 1
        
        # 1. Lower vertical positions (simulate falling down)
        vertical_drop = progress ** 2  # Accelerating descent
        synthetic_fall[t, :, 1] += vertical_drop * 0.5  # Move down
        
        # 2. Rotate body toward horizontal
        center = np.nanmean(synthetic_fall[t, :, :], axis=0)
        centered_pose = synthetic_fall[t, :, :] - center
        
        # Rotation angle (gradually tilt)
        rotation_angle = progress * np.pi / 4  # Up to 45 degrees
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_pose = centered_pose @ rotation_matrix.T
        synthetic_fall[t, :, :] = rotated_pose + center
        
        # 3. Add instability (random jitter)
        noise_scale = progress * 0.02
        synthetic_fall[t, :, :] += np.random.randn(J, 2) * noise_scale
    
    # Post-fall: lying down state
    for t in range(fall_end_frame, T):
        # Keep the fallen position (horizontal)
        if t > fall_end_frame:
            synthetic_fall[t, :, :] = synthetic_fall[fall_end_frame - 1, :, :]
            # Small random movements (struggling)
            synthetic_fall[t, :, :] += np.random.randn(J, 2) * 0.01
    
    metadata = {
        'fall_start_frame': fall_start_frame,
        'fall_end_frame': fall_end_frame,
        'fall_duration': fall_end_frame - fall_start_frame,
        'method': 'standing_to_falling'
    }
    
    return synthetic_fall, metadata


def generate_fall_by_rotation(pose_sequence: np.ndarray,
                              rotation_angle: float = 90.0) -> Tuple[np.ndarray, dict]:
    """
    Generate fall by rotating the entire pose sequence.
    
    Simpler approach: just rotate to horizontal orientation.
    
    Args:
        pose_sequence: (T, J, 2) original sequence
        rotation_angle: Degrees to rotate (90 = horizontal)
        
    Returns:
        synthetic_fall: Rotated sequence
        metadata: Information about transformation
    """
    T, J, _ = pose_sequence.shape
    synthetic_fall = pose_sequence.copy()
    
    angle_rad = rotation_angle * np.pi / 180
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # Rotate each frame around its center
    for t in range(T):
        frame = synthetic_fall[t, :, :]
        center = np.nanmean(frame, axis=0)
        centered = frame - center
        rotated = centered @ rotation_matrix.T
        synthetic_fall[t, :, :] = rotated + center
    
    metadata = {
        'rotation_angle': rotation_angle,
        'method': 'rotation'
    }
    
    return synthetic_fall, metadata


def generate_forward_fall(pose_sequence: np.ndarray, fall_start: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Generate forward fall (like tripping).
    Body tilts forward, arms reach out, then impact.
    """
    T, J, _ = pose_sequence.shape
    synthetic_fall = pose_sequence.copy()
    
    if fall_start is None:
        fall_start = random.randint(T // 4, T // 2)
    
    fall_duration = 20
    fall_end = min(fall_start + fall_duration, T)
    
    for t in range(fall_start, T):
        if t < fall_end:
            progress = (t - fall_start) / fall_duration
            
            # Forward tilt
            synthetic_fall[t, :, 1] += progress ** 2 * 0.4  # Move down
            synthetic_fall[t, :, 0] += progress * 0.2  # Move forward
            
            # Arms reach forward (joints 7-10 = arms)
            synthetic_fall[t, 7:11, 0] += progress * 0.3
            synthetic_fall[t, 7:11, 1] -= progress * 0.1
            
            # Legs stay back
            synthetic_fall[t, 13:17, 0] -= progress * 0.1
        else:
            # Post-fall lying position
            synthetic_fall[t, :, :] = synthetic_fall[fall_end - 1, :, :]
    
    return synthetic_fall, {'method': 'forward_fall', 'fall_start': fall_start}


def generate_backward_fall(pose_sequence: np.ndarray, fall_start: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Generate backward fall (like slipping).
    Body tilts backward, arms flail, legs go up.
    """
    T, J, _ = pose_sequence.shape
    synthetic_fall = pose_sequence.copy()
    
    if fall_start is None:
        fall_start = random.randint(T // 4, T // 2)
    
    fall_duration = 20
    fall_end = min(fall_start + fall_duration, T)
    
    for t in range(fall_start, T):
        if t < fall_end:
            progress = (t - fall_start) / fall_duration
            
            # Backward tilt
            synthetic_fall[t, :, 1] += progress ** 2 * 0.5  # Move down
            synthetic_fall[t, :, 0] -= progress * 0.2  # Move backward
            
            # Arms flail upward
            synthetic_fall[t, 7:11, 1] -= progress * 0.3
            
            # Legs may lift slightly
            synthetic_fall[t, 13:17, 1] -= progress * 0.15
        else:
            # Post-fall lying on back
            synthetic_fall[t, :, :] = synthetic_fall[fall_end - 1, :, :]
    
    return synthetic_fall, {'method': 'backward_fall', 'fall_start': fall_start}


def generate_sideways_fall(pose_sequence: np.ndarray, direction: str = 'left') -> Tuple[np.ndarray, dict]:
    """
    Generate sideways fall (loss of balance to side).
    """
    T, J, _ = pose_sequence.shape
    synthetic_fall = pose_sequence.copy()
    
    fall_start = random.randint(T // 4, T // 2)
    fall_duration = 20
    fall_end = min(fall_start + fall_duration, T)
    
    side_multiplier = -1 if direction == 'left' else 1
    
    for t in range(fall_start, T):
        if t < fall_end:
            progress = (t - fall_start) / fall_duration
            
            # Sideways tilt
            synthetic_fall[t, :, 0] += side_multiplier * progress ** 2 * 0.4
            synthetic_fall[t, :, 1] += progress ** 2 * 0.3
            
            # One arm reaches out
            arm_indices = [7, 9] if direction == 'left' else [8, 10]
            synthetic_fall[t, arm_indices, 0] += side_multiplier * progress * 0.4
        else:
            synthetic_fall[t, :, :] = synthetic_fall[fall_end - 1, :, :]
    
    return synthetic_fall, {'method': 'sideways_fall', 'direction': direction, 'fall_start': fall_start}


def generate_collapse_fall(pose_sequence: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Generate collapse fall (legs give out, medical emergency).
    Sudden vertical drop without forward/backward motion.
    """
    T, J, _ = pose_sequence.shape
    synthetic_fall = pose_sequence.copy()
    
    fall_start = random.randint(T // 4, T // 2)
    fall_duration = 15  # Faster collapse
    fall_end = min(fall_start + fall_duration, T)
    
    for t in range(fall_start, T):
        if t < fall_end:
            progress = (t - fall_start) / fall_duration
            
            # Pure vertical drop
            synthetic_fall[t, :, 1] += progress ** 3 * 0.6  # Accelerating
            
            # Legs collapse first
            synthetic_fall[t, 11:17, 1] += progress ** 2 * 0.3  # Hips and legs
            
            # Upper body follows
            synthetic_fall[t, 0:11, 1] += progress ** 2.5 * 0.4
        else:
            synthetic_fall[t, :, :] = synthetic_fall[fall_end - 1, :, :]
    
    return synthetic_fall, {'method': 'collapse_fall', 'fall_start': fall_start}


def generate_stumble_fall(pose_sequence: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Generate stumble fall (try to catch balance, then fall).
    Two-stage: stumble + recovery attempt, then fall.
    """
    T, J, _ = pose_sequence.shape
    synthetic_fall = pose_sequence.copy()
    
    stumble_start = random.randint(T // 4, T // 3)
    stumble_duration = 10
    fall_start = stumble_start + stumble_duration
    fall_duration = 20
    fall_end = min(fall_start + fall_duration, T)
    
    # Stage 1: Stumble (instability)
    for t in range(stumble_start, fall_start):
        progress = (t - stumble_start) / stumble_duration
        # Wobbling motion
        wobble = np.sin(progress * np.pi * 3) * 0.1
        synthetic_fall[t, :, 0] += wobble
        synthetic_fall[t, :, 1] += abs(wobble) * 0.5
    
    # Stage 2: Fall
    for t in range(fall_start, T):
        if t < fall_end:
            progress = (t - fall_start) / fall_duration
            synthetic_fall[t, :, 1] += progress ** 2 * 0.5
            synthetic_fall[t, :, 0] += progress * random.choice([-0.2, 0.2])
        else:
            synthetic_fall[t, :, :] = synthetic_fall[fall_end - 1, :, :]
    
    return synthetic_fall, {'method': 'stumble_fall', 'stumble_start': stumble_start, 'fall_start': fall_start}


def generate_fall_by_interpolation(standing_pose: np.ndarray,
                                   lying_pose: np.ndarray,
                                   num_frames: int = 20) -> Tuple[np.ndarray, dict]:
    """
    Generate fall by interpolating between standing and lying poses.
    
    Args:
        standing_pose: (J, 2) standing pose
        lying_pose: (J, 2) lying pose  
        num_frames: Number of interpolation frames
        
    Returns:
        fall_sequence: (T, J, 2) interpolated fall
        metadata: Information about generation
    """
    J = standing_pose.shape[0]
    fall_sequence = np.zeros((num_frames, J, 2), dtype=np.float32)
    
    for t in range(num_frames):
        alpha = t / (num_frames - 1)  # 0 to 1
        # Use cubic easing for more realistic motion
        alpha_cubic = alpha ** 3
        fall_sequence[t, :, :] = (1 - alpha_cubic) * standing_pose + alpha_cubic * lying_pose
        
        # Add noise
        fall_sequence[t, :, :] += np.random.randn(J, 2) * 0.01 * alpha
    
    metadata = {
        'num_frames': num_frames,
        'method': 'interpolation'
    }
    
    return fall_sequence, metadata


def augment_fall_sequence(pose_sequence: np.ndarray,
                          add_noise: bool = True,
                          time_warp: bool = True,
                          random_shift: bool = True) -> np.ndarray:
    """
    Apply data augmentation to fall sequences.
    
    Args:
        pose_sequence: (T, J, 2) sequence
        add_noise: Add Gaussian noise
        time_warp: Warp time (speed up/slow down)
        random_shift: Random spatial shift
        
    Returns:
        Augmented sequence
    """
    augmented = pose_sequence.copy()
    
    # 1. Add noise
    if add_noise:
        noise = np.random.randn(*augmented.shape) * 0.02
        augmented += noise
    
    # 2. Time warping (resample at different speeds)
    if time_warp:
        T = augmented.shape[0]
        speed_factor = random.uniform(0.8, 1.2)
        new_T = int(T * speed_factor)
        indices = np.linspace(0, T - 1, new_T).astype(int)
        augmented = augmented[indices, :, :]
        
        # Pad or trim to original length
        if new_T < T:
            padding = np.repeat(augmented[-1:, :, :], T - new_T, axis=0)
            augmented = np.concatenate([augmented, padding], axis=0)
        elif new_T > T:
            augmented = augmented[:T, :, :]
    
    # 3. Random spatial shift
    if random_shift:
        shift = np.random.randn(2) * 0.1
        augmented += shift
    
    return augmented


def create_synthetic_fall_dataset(normal_sequences: list,
                                  num_falls_per_sequence: int = 2,
                                  methods: list = None) -> Tuple[list, list]:
    """
    Create a dataset of synthetic falls from normal sequences.
    
    Args:
        normal_sequences: List of (T, J, 2) normal pose sequences
        num_falls_per_sequence: How many falls to generate per normal sequence
        methods: Which generation methods to use. Options:
                ['forward', 'backward', 'sideways_left', 'sideways_right', 
                 'collapse', 'stumble', 'rotation', 'standing']
                If None, uses all 8 types.
        
    Returns:
        fall_sequences: List of synthetic fall sequences
        metadata_list: List of metadata dicts
    """
    if methods is None:
        # Use all 8 fall types by default
        methods = ['forward', 'backward', 'sideways_left', 'sideways_right', 
                   'collapse', 'stumble', 'rotation', 'standing']
    
    fall_generators = {
        'forward': lambda seq: generate_forward_fall(seq),
        'backward': lambda seq: generate_backward_fall(seq),
        'sideways_left': lambda seq: generate_sideways_fall(seq, direction='left'),
        'sideways_right': lambda seq: generate_sideways_fall(seq, direction='right'),
        'collapse': lambda seq: generate_collapse_fall(seq),
        'stumble': lambda seq: generate_stumble_fall(seq),
        'rotation': lambda seq: generate_fall_by_rotation(seq, rotation_angle=random.uniform(60, 120)),
        'standing': lambda seq: generate_fall_from_standing(seq)
    }
    
    fall_sequences = []
    metadata_list = []
    
    for seq in normal_sequences:
        for _ in range(num_falls_per_sequence):
            method = random.choice(methods)
            
            if method in fall_generators:
                fall_seq, metadata = fall_generators[method](seq)
            else:
                continue
            
            # Apply augmentation
            fall_seq = augment_fall_sequence(fall_seq)
            
            fall_sequences.append(fall_seq)
            metadata_list.append(metadata)
    
    return fall_sequences, metadata_list


if __name__ == "__main__":
    # Test the synthetic fall generation
    print("Testing synthetic fall generation...")
    
    # Create a simple standing pose
    T, J = 64, 18
    test_pose = np.random.randn(T, J, 2) * 0.1
    test_pose[:, 11:13, 1] = 0.5  # Hips at height 0.5
    test_pose[:, 5:7, 1] = 0.3    # Shoulders at height 0.3
    
    # Generate synthetic fall
    fall_seq, metadata = generate_fall_from_standing(test_pose, fall_duration=20)
    
    # Extract features
    normal_features = extract_fall_features(test_pose)
    fall_features = extract_fall_features(fall_seq)
    
    print("\nNormal sequence features:")
    for k, v in normal_features.items():
        print(f"  {k}: {v}")
    
    print("\nFall sequence features:")
    for k, v in fall_features.items():
        print(f"  {k}: {v}")
    
    print("\nRule-based detection:")
    print(f"  Normal is fall? {is_fall_by_rules(normal_features)}")
    print(f"  Synthetic fall is fall? {is_fall_by_rules(fall_features)}")
    
    print("\n✅ Synthetic fall generation working!")
