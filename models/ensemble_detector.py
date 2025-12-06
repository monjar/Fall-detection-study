"""
Ensemble Fall Detection System

Combines multiple detection methods:
1. Neural Network (Autoencoder + Classifier)
2. Physics-based Rules (velocity, angle, height drop)
3. Statistical Anomaly Detection (reconstruction error)
4. Temporal Pattern Matching (motion dynamics)
5. Pose-based Features (body geometry)

Ensemble voting strategies:
- Weighted voting
- Soft voting (probability averaging)
- Stacking (meta-learner)
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
from scipy import signal
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression


class PhysicsBasedDetector:
    """
    Physics-based fall detection using biomechanical rules.
    No learning - pure physics thresholds.
    """
    
    def __init__(self, 
                 velocity_threshold: float = -0.3,
                 angle_threshold: float = 45.0,
                 height_drop_threshold: float = 0.2,
                 aspect_ratio_threshold: float = 2.0):
        """
        Args:
            velocity_threshold: Max vertical velocity (negative = downward)
            angle_threshold: Max body angle from vertical (degrees)
            height_drop_threshold: Min height drop ratio
            aspect_ratio_threshold: Min width/height ratio
        """
        self.velocity_threshold = velocity_threshold
        self.angle_threshold = angle_threshold
        self.height_drop_threshold = height_drop_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        
    def extract_features(self, pose_sequence: np.ndarray) -> Dict[str, float]:
        """
        Extract physics-based features from pose sequence.
        
        Args:
            pose_sequence: (T, J, 2) array of poses
            
        Returns:
            Dictionary of features
        """
        T, J, _ = pose_sequence.shape
        
        # Get key points
        neck = pose_sequence[:, 1, :]  # Joint 1 = Neck
        hips_center = (pose_sequence[:, 8, :] + pose_sequence[:, 11, :]) / 2  # Hip center
        head = pose_sequence[:, 0, :]  # Joint 0 = Nose
        feet = (pose_sequence[:, 10, :] + pose_sequence[:, 13, :]) / 2  # Feet center
        
        # 1. Vertical velocity (downward motion)
        vertical_positions = neck[:, 1]  # Y coordinates (higher = lower in image)
        velocities = np.diff(vertical_positions)
        max_velocity = np.max(velocities) if len(velocities) > 0 else 0
        mean_velocity = np.mean(velocities) if len(velocities) > 0 else 0
        
        # 2. Body angle (deviation from vertical)
        body_vectors = neck - hips_center
        angles = []
        for vec in body_vectors:
            if np.linalg.norm(vec) > 0:
                angle = np.abs(np.arctan2(vec[0], vec[1]) * 180 / np.pi)
                angles.append(angle)
        max_angle = np.max(angles) if angles else 0
        mean_angle = np.mean(angles) if angles else 0
        
        # 3. Height drop (change in vertical extent)
        heights = np.abs(head[:, 1] - feet[:, 1])
        initial_height = np.mean(heights[:5]) if T > 5 else np.mean(heights)
        final_height = np.mean(heights[-5:]) if T > 5 else np.mean(heights)
        height_drop_ratio = (initial_height - final_height) / (initial_height + 1e-6)
        
        # 4. Aspect ratio (width/height - increases during fall)
        widths = []
        for t in range(T):
            x_coords = pose_sequence[t, :, 0]
            width = np.max(x_coords) - np.min(x_coords)
            widths.append(width)
        aspect_ratios = np.array(widths) / (heights + 1e-6)
        max_aspect_ratio = np.max(aspect_ratios)
        
        # 5. Acceleration (sudden changes)
        if len(velocities) > 1:
            accelerations = np.diff(velocities)
            max_acceleration = np.max(np.abs(accelerations))
        else:
            max_acceleration = 0
            
        # 6. Centroid drop
        centroids_y = np.mean(pose_sequence[:, :, 1], axis=1)
        centroid_drop = np.max(centroids_y) - np.min(centroids_y)
        
        return {
            'max_velocity': max_velocity,
            'mean_velocity': mean_velocity,
            'max_angle': max_angle,
            'mean_angle': mean_angle,
            'height_drop_ratio': height_drop_ratio,
            'max_aspect_ratio': max_aspect_ratio,
            'max_acceleration': max_acceleration,
            'centroid_drop': centroid_drop,
        }
    
    def predict(self, pose_sequence: np.ndarray) -> Tuple[int, float]:
        """
        Predict if sequence contains a fall.
        
        Args:
            pose_sequence: (T, J, 2) array of poses
            
        Returns:
            (prediction, confidence) where prediction is 0 (no fall) or 1 (fall)
        """
        features = self.extract_features(pose_sequence)
        
        # Rule-based decision with confidence
        fall_indicators = 0
        confidence_scores = []
        
        # Check velocity
        if features['max_velocity'] > -self.velocity_threshold:
            fall_indicators += 1
            confidence_scores.append(min(features['max_velocity'] / (-self.velocity_threshold), 1.0))
        
        # Check angle
        if features['max_angle'] > self.angle_threshold:
            fall_indicators += 1
            confidence_scores.append(min(features['max_angle'] / self.angle_threshold, 1.0))
        
        # Check height drop
        if features['height_drop_ratio'] > self.height_drop_threshold:
            fall_indicators += 1
            confidence_scores.append(min(features['height_drop_ratio'] / self.height_drop_threshold, 1.0))
        
        # Check aspect ratio
        if features['max_aspect_ratio'] > self.aspect_ratio_threshold:
            fall_indicators += 1
            confidence_scores.append(min(features['max_aspect_ratio'] / self.aspect_ratio_threshold, 1.0))
        
        # Check acceleration
        if features['max_acceleration'] > 0.1:
            fall_indicators += 1
            confidence_scores.append(min(features['max_acceleration'] / 0.1, 1.0))
        
        # Decision: if at least 2 indicators triggered
        prediction = 1 if fall_indicators >= 2 else 0
        confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return prediction, confidence


class TemporalPatternDetector:
    """
    Detect falls based on temporal motion patterns.
    Uses signal processing techniques.
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        
    def compute_motion_energy(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Compute motion energy over time."""
        T = pose_sequence.shape[0]
        motion_energy = np.zeros(T - 1)
        
        for t in range(T - 1):
            diff = pose_sequence[t + 1] - pose_sequence[t]
            motion_energy[t] = np.sum(diff ** 2)
        
        return motion_energy
    
    def detect_sudden_changes(self, signal_data: np.ndarray) -> Tuple[int, float]:
        """
        Detect sudden changes in signal (characteristic of falls).
        
        Returns:
            (prediction, confidence)
        """
        if len(signal_data) < 5:
            return 0, 0.0
        
        # Compute derivatives (rate of change)
        derivatives = np.diff(signal_data)
        
        # Find peaks (sudden increases)
        peaks, properties = signal.find_peaks(derivatives, 
                                              height=np.std(derivatives) * 2,
                                              distance=5)
        
        # Falls typically have sharp peaks in motion energy
        if len(peaks) > 0:
            peak_heights = properties['peak_heights']
            max_peak = np.max(peak_heights)
            
            # High peak + followed by low values = fall pattern
            avg_after_peak = np.mean(signal_data[-5:]) if len(signal_data) > 5 else 0
            
            if max_peak > np.mean(signal_data) * 2 and avg_after_peak < np.mean(signal_data):
                return 1, min(max_peak / (np.mean(signal_data) + 1e-6) / 5, 1.0)
        
        return 0, 0.0
    
    def predict(self, pose_sequence: np.ndarray) -> Tuple[int, float]:
        """
        Predict based on temporal patterns.
        
        Args:
            pose_sequence: (T, J, 2) array of poses
            
        Returns:
            (prediction, confidence)
        """
        # Compute motion energy
        motion_energy = self.compute_motion_energy(pose_sequence)
        
        # Detect sudden changes
        prediction, confidence = self.detect_sudden_changes(motion_energy)
        
        return prediction, confidence


class PoseGeometryDetector:
    """
    Detect falls based on pose geometry changes.
    Analyzes body shape and configuration.
    """
    
    def __init__(self):
        pass
    
    def compute_pose_compactness(self, pose: np.ndarray) -> float:
        """
        Compute how compact the pose is (falls = more compact).
        
        Args:
            pose: (J, 2) single pose
            
        Returns:
            Compactness score (higher = more compact)
        """
        # Compute bounding box
        x_coords = pose[:, 0]
        y_coords = pose[:, 1]
        
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        
        # Area of bounding box
        area = width * height
        
        # Distance from centroid
        centroid = np.mean(pose, axis=0)
        distances = np.linalg.norm(pose - centroid, axis=1)
        avg_distance = np.mean(distances)
        
        # Compactness: inverse of area * avg distance
        compactness = 1.0 / (area * avg_distance + 1e-6)
        
        return compactness
    
    def compute_vertical_spread(self, pose: np.ndarray) -> float:
        """Compute vertical spread of joints (falls = less spread)."""
        y_coords = pose[:, 1]
        return np.max(y_coords) - np.min(y_coords)
    
    def predict(self, pose_sequence: np.ndarray) -> Tuple[int, float]:
        """
        Predict based on geometry changes.
        
        Args:
            pose_sequence: (T, J, 2) array of poses
            
        Returns:
            (prediction, confidence)
        """
        T = pose_sequence.shape[0]
        
        # Compute metrics over time
        compactness_scores = []
        vertical_spreads = []
        
        for t in range(T):
            compactness = self.compute_pose_compactness(pose_sequence[t])
            v_spread = self.compute_vertical_spread(pose_sequence[t])
            
            compactness_scores.append(compactness)
            vertical_spreads.append(v_spread)
        
        compactness_scores = np.array(compactness_scores)
        vertical_spreads = np.array(vertical_spreads)
        
        # Falls: increasing compactness, decreasing vertical spread
        initial_compactness = np.mean(compactness_scores[:T//3])
        final_compactness = np.mean(compactness_scores[-T//3:])
        compactness_increase = final_compactness / (initial_compactness + 1e-6)
        
        initial_spread = np.mean(vertical_spreads[:T//3])
        final_spread = np.mean(vertical_spreads[-T//3:])
        spread_decrease = initial_spread / (final_spread + 1e-6)
        
        # Decision
        if compactness_increase > 1.5 and spread_decrease > 1.3:
            return 1, min((compactness_increase + spread_decrease) / 5, 1.0)
        
        return 0, 0.0


class NeuralNetworkDetector:
    """
    Wrapper for neural network-based detection.
    Uses the trained autoencoder + classifier.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict(self, pose_sequence: np.ndarray) -> Tuple[int, float]:
        """
        Predict using neural network.
        
        Args:
            pose_sequence: (T, J, 2) array of poses
            
        Returns:
            (prediction, confidence)
        """
        # Convert to tensor
        tensor = torch.from_numpy(pose_sequence.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, 2, T, J)
        
        with torch.no_grad():
            reconstructed, logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probs[0, prediction].item()
        
        return prediction, confidence


class AnomalyDetector:
    """
    Statistical anomaly detection based on reconstruction error.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu', 
                 threshold_percentile: float = 90):
        self.model = model
        self.device = device
        self.threshold_percentile = threshold_percentile
        self.threshold = None  # Set during calibration
    
    def compute_reconstruction_error(self, pose_sequence: np.ndarray) -> float:
        """Compute reconstruction error."""
        tensor = torch.from_numpy(pose_sequence.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            reconstructed, _ = self.model(tensor)
            error = torch.mean((tensor - reconstructed) ** 2).item()
        
        return error
    
    def calibrate(self, normal_sequences: list):
        """
        Calibrate threshold on normal sequences.
        
        Args:
            normal_sequences: List of (T, J, 2) normal pose sequences
        """
        errors = []
        for seq in normal_sequences:
            error = self.compute_reconstruction_error(seq)
            errors.append(error)
        
        self.threshold = np.percentile(errors, self.threshold_percentile)
        print(f"Anomaly detector calibrated: threshold={self.threshold:.4f}")
    
    def predict(self, pose_sequence: np.ndarray) -> Tuple[int, float]:
        """
        Predict based on reconstruction error.
        
        Args:
            pose_sequence: (T, J, 2) array of poses
            
        Returns:
            (prediction, confidence)
        """
        error = self.compute_reconstruction_error(pose_sequence)
        
        if self.threshold is None:
            # Default threshold if not calibrated
            self.threshold = 1.0
        
        # High error = anomaly = fall
        if error > self.threshold:
            confidence = min((error - self.threshold) / self.threshold, 1.0)
            return 1, confidence
        else:
            confidence = 1.0 - error / self.threshold
            return 0, confidence


class EnsembleFallDetector:
    """
    Ensemble of multiple fall detection methods.
    Combines predictions using voting or stacking.
    """
    
    def __init__(self, 
                 neural_model: Optional[nn.Module] = None,
                 device: str = 'cpu',
                 voting_strategy: str = 'weighted',
                 weights: Optional[Dict[str, float]] = None):
        """
        Args:
            neural_model: Trained neural network (optional)
            device: Device for neural network
            voting_strategy: 'weighted', 'majority', 'soft', or 'stacking'
            weights: Weights for each detector (if weighted voting)
        """
        self.device = device
        self.voting_strategy = voting_strategy
        
        # Initialize detectors
        self.detectors = {
            'physics': PhysicsBasedDetector(),
            'temporal': TemporalPatternDetector(),
            'geometry': PoseGeometryDetector(),
        }
        
        if neural_model is not None:
            self.detectors['neural'] = NeuralNetworkDetector(neural_model, device)
            self.detectors['anomaly'] = AnomalyDetector(neural_model, device)
        
        # Set weights
        if weights is None:
            # Default weights (physics and temporal more reliable)
            self.weights = {
                'physics': 0.30,
                'temporal': 0.25,
                'geometry': 0.20,
                'neural': 0.15,
                'anomaly': 0.10,
            }
        else:
            self.weights = weights
        
        # Meta-learner for stacking (trained later)
        self.meta_learner = None
    
    def predict_all(self, pose_sequence: np.ndarray) -> Dict[str, Tuple[int, float]]:
        """
        Get predictions from all detectors.
        
        Args:
            pose_sequence: (T, J, 2) array of poses
            
        Returns:
            Dictionary mapping detector name to (prediction, confidence)
        """
        predictions = {}
        
        for name, detector in self.detectors.items():
            try:
                pred, conf = detector.predict(pose_sequence)
                predictions[name] = (pred, conf)
            except Exception as e:
                print(f"Warning: {name} detector failed: {e}")
                predictions[name] = (0, 0.0)
        
        return predictions
    
    def predict_weighted(self, predictions: Dict[str, Tuple[int, float]]) -> Tuple[int, float]:
        """Weighted voting."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, (pred, conf) in predictions.items():
            weight = self.weights.get(name, 0.0)
            weighted_sum += pred * conf * weight
            total_weight += weight
        
        # Average weighted score
        score = weighted_sum / (total_weight + 1e-6)
        
        # Threshold at 0.5
        prediction = 1 if score >= 0.5 else 0
        confidence = score if prediction == 1 else (1 - score)
        
        return prediction, confidence
    
    def predict_majority(self, predictions: Dict[str, Tuple[int, float]]) -> Tuple[int, float]:
        """Majority voting."""
        votes = [pred for pred, _ in predictions.values()]
        fall_votes = sum(votes)
        total_votes = len(votes)
        
        prediction = 1 if fall_votes > total_votes / 2 else 0
        confidence = fall_votes / total_votes if prediction == 1 else 1 - fall_votes / total_votes
        
        return prediction, confidence
    
    def predict_soft(self, predictions: Dict[str, Tuple[int, float]]) -> Tuple[int, float]:
        """Soft voting (average probabilities)."""
        # Convert predictions to probabilities
        probs_fall = []
        for pred, conf in predictions.values():
            prob_fall = conf if pred == 1 else (1 - conf)
            probs_fall.append(prob_fall)
        
        avg_prob_fall = np.mean(probs_fall)
        prediction = 1 if avg_prob_fall >= 0.5 else 0
        confidence = avg_prob_fall if prediction == 1 else (1 - avg_prob_fall)
        
        return prediction, confidence
    
    def predict(self, pose_sequence: np.ndarray) -> Tuple[int, float, Dict]:
        """
        Predict using ensemble.
        
        Args:
            pose_sequence: (T, J, 2) array of poses
            
        Returns:
            (prediction, confidence, individual_predictions)
        """
        # Get predictions from all detectors
        individual_preds = self.predict_all(pose_sequence)
        
        # Combine predictions
        if self.voting_strategy == 'weighted':
            prediction, confidence = self.predict_weighted(individual_preds)
        elif self.voting_strategy == 'majority':
            prediction, confidence = self.predict_majority(individual_preds)
        elif self.voting_strategy == 'soft':
            prediction, confidence = self.predict_soft(individual_preds)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        
        return prediction, confidence, individual_preds
    
    def calibrate_anomaly_detector(self, normal_sequences: list):
        """Calibrate the anomaly detector on normal sequences."""
        if 'anomaly' in self.detectors:
            self.detectors['anomaly'].calibrate(normal_sequences)
    
    def train_meta_learner(self, X_features: np.ndarray, y_labels: np.ndarray):
        """
        Train meta-learner for stacking.
        
        Args:
            X_features: (N, D) array of detector predictions
            y_labels: (N,) array of true labels
        """
        self.meta_learner = LogisticRegression(max_iter=1000)
        self.meta_learner.fit(X_features, y_labels)
        self.voting_strategy = 'stacking'
        print(f"Meta-learner trained on {len(y_labels)} samples")
    
    def predict_stacking(self, predictions: Dict[str, Tuple[int, float]]) -> Tuple[int, float]:
        """Predict using stacking (meta-learner)."""
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained. Call train_meta_learner first.")
        
        # Convert predictions to feature vector
        features = []
        for name in sorted(self.detectors.keys()):
            pred, conf = predictions.get(name, (0, 0.0))
            features.append(pred * conf)
        
        features = np.array(features).reshape(1, -1)
        
        # Predict with meta-learner
        prediction = self.meta_learner.predict(features)[0]
        confidence = np.max(self.meta_learner.predict_proba(features))
        
        return prediction, confidence
