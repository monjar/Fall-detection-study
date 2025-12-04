"""Utility helpers for discovering and reading the KTH action dataset videos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import cv2

KTH_ACTIONS: Tuple[str, ...] = (
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking",
)


@dataclass(frozen=True)
class KTHVideoSample:
    """Metadata wrapper around a single KTH video clip."""

    action: str
    subject: str
    scenario: str
    repetition: Optional[str]
    path: Path
    fps: float
    frame_count: int
    duration: float

    @property
    def video_id(self) -> str:
        return f"{self.subject}_{self.action}_{self.scenario}" + (
            f"_{self.repetition}" if self.repetition else ""
        )


def discover_kth_videos(
    root: Path,
    actions: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[KTHVideoSample]:
    """Scan the KTH root directory and build metadata records for each clip."""

    actions = tuple(actions) if actions else KTH_ACTIONS
    samples: List[KTHVideoSample] = []

    for action in actions:
        action_dir = root / action
        if not action_dir.exists():
            continue
        for video_path in sorted(action_dir.glob("*.avi")):
            meta = _parse_kth_filename(video_path)
            fps, frame_count, duration = probe_video(video_path)
            samples.append(
                KTHVideoSample(
                    action=action,
                    subject=meta["subject"],
                    scenario=meta["scenario"],
                    repetition=meta.get("repetition"),
                    path=video_path,
                    fps=fps,
                    frame_count=frame_count,
                    duration=duration,
                )
            )
            if limit and len(samples) >= limit:
                return samples
    return samples


def probe_video(video_path: Path) -> Tuple[float, int, float]:
    """Return (fps, frame_count, duration_seconds) for a video."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    capture.release()
    return fps, frame_count, duration


def iter_video_frames(
    video_path: Path,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[int, float, Any]]:
    """Yield (frame_idx, timestamp_seconds, frame_bgr) with optional stride."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    idx = 0
    yielded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if idx % stride == 0:
            timestamp = idx / fps if fps > 0 else 0.0
            yield idx, timestamp, frame
            yielded += 1
            if max_frames and yielded >= max_frames:
                break
        idx += 1
    capture.release()


def _parse_kth_filename(video_path: Path) -> dict:
    """Extract subject/action/scenario from canonical KTH filenames."""

    stem = video_path.stem  # e.g. person01_boxing_d1_uncomp
    tokens = stem.split("_")
    result = {
        "subject": tokens[0],
        "action": tokens[1] if len(tokens) > 1 else "unknown",
        "scenario": tokens[2] if len(tokens) > 2 else "d1",
    }
    if len(tokens) > 3 and tokens[3] not in {"uncomp", "uncompres"}:
        result["repetition"] = tokens[3]
    return result
