"""Pose normalization utilities and dataset objects."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_SEQUENCE_LENGTH = 64
NUM_JOINTS = 18


def load_npz_with_meta(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    data = np.load(path, allow_pickle=True)
    poses = data["poses"]
    timestamps = data.get("timestamps", np.arange(len(poses)))
    raw_meta = data.get("meta", None)
    if raw_meta is None:
        meta = {"source": path.name}
    else:
        if isinstance(raw_meta, (bytes, str)):
            meta = json.loads(raw_meta)
        else:
            meta = json.loads(str(raw_meta.item()))
    return poses, timestamps, meta


def normalize_pose_sequence(sequence: np.ndarray) -> np.ndarray:
    """Center and scale joints relative to the hip midpoint with better robustness."""

    if sequence.ndim != 3:
        raise ValueError("Pose sequence must be (T, J, 2)")
    seq = sequence.copy()
    
    # Replace zeros (missing joints) with NaN for proper handling
    seq[seq == 0] = np.nan
    
    # Use neck (index 1) as center if available, otherwise use mean of visible joints
    neck_idx = 1
    if not np.isnan(seq[:, neck_idx, :]).all():
        center = np.nanmean(seq[:, neck_idx : neck_idx + 1, :], axis=0, keepdims=True)
    else:
        center = np.nanmean(seq, axis=1, keepdims=True)
    
    seq = seq - center
    
    # Scale by shoulder width to obtain scale invariance
    left_shoulder, right_shoulder = 2, 5
    shoulder_dist = np.linalg.norm(
        seq[:, left_shoulder, :] - seq[:, right_shoulder, :], axis=1, keepdims=True
    )
    # Use robust scaling with median and clip outliers
    valid_dists = shoulder_dist[~np.isnan(shoulder_dist)]
    if len(valid_dists) > 0:
        scale = np.median(valid_dists)
        scale = np.maximum(scale, 1.0)  # Increased minimum scale
    else:
        scale = 100.0  # Default scale for missing data
    
    seq /= scale
    
    # Replace NaN back with zeros after normalization
    seq = np.nan_to_num(seq, nan=0.0)
    
    # Clip extreme outliers to prevent gradient explosion
    seq = np.clip(seq, -5.0, 5.0)
    
    return seq


def pad_or_trim(sequence: np.ndarray, target_length: int) -> np.ndarray:
    t = sequence.shape[0]
    if t == target_length:
        return sequence
    if t > target_length:
        return sequence[:target_length]
    pad_shape = (target_length - t, sequence.shape[1], sequence.shape[2])
    pad = np.zeros(pad_shape, dtype=sequence.dtype)
    pad[:] = sequence[-1]
    return np.concatenate([sequence, pad], axis=0)


class PoseSequenceDataset(Dataset):
    """Windowed pose dataset backed by .npz files produced by OpenPose parser."""

    def __init__(
        self,
        pose_dir: Path,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        stride: int = 32,
        normalize: bool = True,
    ) -> None:
        self.pose_dir = Path(pose_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize

        self.files = sorted(self.pose_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No pose sequences found under {self.pose_dir}. Run run_openpose.py first."
            )
        self.index: List[Tuple[Path, int]] = []
        for path in self.files:
            poses, _, _ = load_npz_with_meta(path)
            total = poses.shape[0]
            if total <= 0:
                continue
            start = 0
            while start + sequence_length <= total:
                self.index.append((path, start))
                start += stride
            if total < sequence_length:
                self.index.append((path, 0))
            elif start < total:
                self.index.append((path, total - sequence_length))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        path, start = self.index[idx]
        poses, timestamps, meta = load_npz_with_meta(path)
        clip = poses[start : start + self.sequence_length]
        clip = pad_or_trim(clip, self.sequence_length)
        ts_clip = timestamps[start : start + self.sequence_length]
        if len(ts_clip) < self.sequence_length:
            if len(ts_clip) == 0:
                ts_clip = np.zeros(self.sequence_length)
            else:
                pad = np.full(self.sequence_length - len(ts_clip), ts_clip[-1])
                ts_clip = np.concatenate([ts_clip, pad])
        if self.normalize:
            clip = normalize_pose_sequence(clip)
        tensor = torch.from_numpy(clip.transpose(2, 0, 1)).float()  # (2, T, J)
        meta = {**meta, "source_file": path.name, "start_index": int(start)}
        sample = {
            "pose": tensor,
            "timestamps": ts_clip,
            "meta": meta,
        }
        return tensor, sample
