"""Download the OpenPose (COCO) prototxt + caffemodel pair used by OpenCV DNN."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ASSETS = (
    {
        "name": "prototxt",
        "filename": "pose_deploy_linevec.prototxt",
        "url": "https://huggingface.co/gaijingeek/openpose-models/resolve/main/pose/coco/pose_deploy_linevec.prototxt",
        "sha256": "17051b87f709aa094e09c5da7b78e9016a1f37b2b452ed1f190fe74cce70b1ad",
    },
    {
        "name": "caffemodel",
        "filename": "pose_iter_440000.caffemodel",
        "url": "https://huggingface.co/gaijingeek/openpose-models/resolve/main/pose/coco/pose_iter_440000.caffemodel",
        "sha256": "b4cf475576abd7b15d5316f1ee65eb492b5c9f5865e70a2e7882ed31fb682549",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenCV pose assets")
    parser.add_argument("--dest", type=Path, default=Path("models/opencv_pose"), help="Output directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if the expected checksum already exists",
    )
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    request = Request(url, headers={"User-Agent": "fall-detection-study/pose-downloader"})
    with urlopen(request) as response, dest.open("wb") as output:
        shutil.copyfileobj(response, output)


def ensure_asset(asset: dict[str, str], dest_dir: Path, force: bool) -> None:
    target = dest_dir / asset["filename"]
    if not force and target.exists():
        current = sha256sum(target)
        if current == asset["sha256"]:
            print(f"✔ {asset['name']} already present ({target})")
            return
        print(f"! {asset['name']} checksum mismatch; re-downloading")
    tmp_path = target.with_suffix(target.suffix + ".part")
    try:
        print(f"➡  Fetching {asset['name']} from {asset['url']}")
        download(asset["url"], tmp_path)
        digest = sha256sum(tmp_path)
        if digest != asset["sha256"]:
            raise RuntimeError(
                f"Checksum mismatch for {asset['name']}: expected {asset['sha256']}, got {digest}"
            )
        tmp_path.replace(target)
        print(f"✔ Saved {asset['filename']} ({target.stat().st_size / (1024**2):.1f} MB)")
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    try:
        for asset in ASSETS:
            ensure_asset(asset, args.dest, args.force)
    except (HTTPError, URLError) as err:
        print(f"Download failed: {err}", file=sys.stderr)
        return 1
    except RuntimeError as err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())