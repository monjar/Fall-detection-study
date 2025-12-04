"""Pose keypoint extractor for KTH dataset using OpenCV's DNN module."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
from rich.progress import Progress

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.kth_dataset import KTHVideoSample, discover_kth_videos, iter_video_frames


BODY_PARTS = (
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
)
NUM_JOINTS = len(BODY_PARTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pose keypoints with OpenCV")
    parser.add_argument("--video-root", type=Path, default=Path("data/kth_raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/pose_keypoints"))
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("models/opencv_pose/pose_deploy_linevec.prototxt"),
        help="Path to the Caffe prototxt graph for OpenPose (18-point COCO)",
    )
    parser.add_argument(
        "--model-weights",
        type=Path,
        default=Path("models/opencv_pose/pose_iter_440000.caffemodel"),
        help="Path to the trained Caffe weights",
    )
    parser.add_argument("--actions", nargs="*", default=None, help="Subset of actions to process")
    parser.add_argument("--limit", type=int, default=None, help="Optional clip limit for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if npz exists")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame to save time")
    parser.add_argument("--input-width", type=int, default=256, help="DNN input width (lower = faster, less accurate)")
    parser.add_argument("--input-height", type=int, default=256, help="DNN input height (lower = faster, less accurate)")
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--prefer-gpu", action="store_true", help="Try to run the DNN on CUDA if available")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of frames to process per forward pass (trade memory for speed)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers for video processing (0 = auto-detect CPU count, 1 = single-threaded)",
    )
    return parser.parse_args()


def load_pose_net(proto: Path, weights: Path, prefer_gpu: bool) -> Any:
    if not proto.exists() or not weights.exists():
        raise FileNotFoundError(
            "Missing OpenCV pose model files. Provide --model-config/--model-weights or download them to"
            f" {proto} and {weights}."
        )
    net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
    if prefer_gpu:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("✓ Using CUDA GPU for inference")
        except (cv2.error, Exception) as e:
            print(f"⚠ CUDA not available ({str(e)[:60]}...), falling back to CPU")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU for inference")
    return net


def infer_keypoints_batch(
    frames: List[np.ndarray],
    net: Any,
    input_size: Tuple[int, int],
    threshold: float,
) -> np.ndarray:
    if not frames:
        return np.empty((0, NUM_JOINTS, 2), dtype=np.float32)

    blob = cv2.dnn.blobFromImages(
        frames,
        1.0 / 255,
        input_size,
        (0, 0, 0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    output = net.forward()
    heatmap_h, heatmap_w = output.shape[2], output.shape[3]

    batch_points = np.zeros((len(frames), NUM_JOINTS, 2), dtype=np.float32)
    for batch_idx, frame in enumerate(frames):
        frame_height, frame_width = frame.shape[:2]
        for joint_idx in range(NUM_JOINTS):
            prob_map = output[batch_idx, joint_idx, :, :]
            _, prob, _, point = cv2.minMaxLoc(prob_map)
            if prob > threshold:
                x = frame_width * point[0] / heatmap_w
                y = frame_height * point[1] / heatmap_h
                batch_points[batch_idx, joint_idx] = (x, y)
    return batch_points


def process_sample_worker(
    sample: KTHVideoSample,
    args: argparse.Namespace,
) -> tuple[str, Path | None]:
    """Worker function that loads its own DNN model to avoid pickling issues."""
    net = load_pose_net(args.model_config, args.model_weights, args.prefer_gpu)
    result = _process_sample_impl(sample, net, args)
    return sample.video_id, result


def _process_sample_impl(
    sample: KTHVideoSample,
    net: Any,
    args: argparse.Namespace,
) -> Path | None:
    rel_id = sample.video_id
    npz_dir = args.output_root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / f"{rel_id}.npz"

    if npz_path.exists() and not args.overwrite:
        return None

    poses = []
    timestamps = []
    frame_task = None
    if False:  # No progress UI in worker processes
        total_frames = max(1, sample.frame_count // args.frame_stride)
        frame_task = progress.add_task(
            f"[cyan]{sample.video_id}", total=total_frames, transient=True
        )

    batch_frames: List[np.ndarray] = []
    batch_timestamps: List[float] = []
    for _, timestamp, frame in iter_video_frames(sample.path, stride=args.frame_stride):
        batch_frames.append(frame)
        batch_timestamps.append(timestamp)

        if len(batch_frames) == args.batch_size:
            batch_poses = infer_keypoints_batch(
                batch_frames,
                net,
                (args.input_width, args.input_height),
                args.confidence_threshold,
            )
            poses.extend(batch_poses)
            timestamps.extend(batch_timestamps)
            batch_frames.clear()
            batch_timestamps.clear()

    if batch_frames:
        batch_poses = infer_keypoints_batch(
            batch_frames,
            net,
            (args.input_width, args.input_height),
            args.confidence_threshold,
        )
        poses.extend(batch_poses)
        timestamps.extend(batch_timestamps)

    if not poses:
        return None

    poses_arr = np.stack(poses).astype(np.float32)
    timestamps_arr = np.asarray(timestamps, dtype=np.float32)
    meta = {
        "video_id": sample.video_id,
        "action": sample.action,
        "subject": sample.subject,
        "scenario": sample.scenario,
        "fps": sample.fps,
        "frame_count": sample.frame_count,
        "duration": sample.duration,
        "source": str(sample.path.relative_to(args.video_root)),
        "model": "opencv-openpose-coco",
    }
    np.savez_compressed(npz_path, poses=poses_arr, timestamps=timestamps_arr, meta=json.dumps(meta))
    return npz_path


def main():
    args = parse_args()
    samples = discover_kth_videos(args.video_root, actions=args.actions, limit=args.limit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    
    num_workers = args.workers
    if num_workers == 0:
        num_workers = mp.cpu_count()
    
    created = 0
    
    if num_workers == 1:
        # Single-threaded path with progress UI
        net = load_pose_net(args.model_config, args.model_weights, args.prefer_gpu)
        with Progress() as progress:
            task = progress.add_task("Processing videos", total=len(samples))
            for sample in samples:
                result = _process_sample_impl(sample, net, args)
                if result:
                    created += 1
                progress.advance(task)
    else:
        # Parallel processing
        with Progress() as progress:
            task = progress.add_task("Processing videos", total=len(samples))
            with mp.Pool(processes=num_workers) as pool:
                worker_args = [(sample, args) for sample in samples]
                for video_id, npz_path in pool.starmap(process_sample_worker, worker_args):
                    if npz_path:
                        created += 1
                    progress.advance(task)
    
    print(f"Finished. New pose tensors: {created}/{len(samples)}")


if __name__ == "__main__":
    main()
