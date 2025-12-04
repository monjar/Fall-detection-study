import json
from pathlib import Path

import numpy as np

from utils.pose_processing import PoseSequenceDataset


def make_dummy_npz(path: Path, frames: int = 12, joints: int = 18):
    poses = np.random.rand(frames, joints, 2).astype("float32")
    timestamps = np.linspace(0, frames / 25.0, frames, dtype="float32")
    meta = json.dumps({"video_id": "dummy", "action": "boxing"})
    np.savez_compressed(path, poses=poses, timestamps=timestamps, meta=meta)


def test_pose_dataset_windows(tmp_path):
    file_path = tmp_path / "sample.npz"
    make_dummy_npz(file_path)

    dataset = PoseSequenceDataset(tmp_path, sequence_length=8, stride=4)
    assert len(dataset) >= 2
    tensor, sample = dataset[0]
    assert tensor.shape == (2, 8, 18)
    assert sample["timestamps"].shape[0] == 8
    assert sample["meta"]["source_file"] == "sample.npz"
