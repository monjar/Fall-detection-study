# Fall Detection Study (KTH + OpenCV Pose + CNN Autoencoder)

This repository implements an end-to-end experimentation stack for fall-detection research when only *normal* human activities are available for training. It uses the KTH action dataset for pretext learning, extracts 2D human poses via OpenCV's built-in OpenPose (Caffe) model, and trains a temporal convolutional autoencoder that can flag high reconstruction-error segments as potential falls (anomalies).

## 🧭 Pipeline Overview

1. **Ingest KTH videos** (`data/kth_raw/<action>/*.avi`).
2. **Extract 2D poses** with OpenCV's built-in OpenPose (Caffe) model and timestamp every frame.
3. **Window pose sequences** into fixed-length clips and normalize joints.
4. **Train a CNN autoencoder** (only on KTH actions) to reconstruct normal motion.
5. **Detect anomalies** by reconstruction error thresholds → candidates for falls.

```
Raw video ──▶ OpenCV pose tensors ──▶ Normalized pose tensors ──▶ CNN autoencoder ──▶ Reconstruction error (anomaly score)
```

## ✅ Requirements & Environment

```bash
python -m venv .venv
source .venv/bin/activate  # (Linux/macOS)
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** PyTorch wheels depend on your CUDA/CuDNN stack. If you have a GPU, install the matching wheels from [pytorch.org](https://pytorch.org/get-started/locally/).

### External dependencies

- OpenCV's pretrained OpenPose (COCO) model files (`pose_deploy_linevec.prototxt` + `pose_iter_440000.caffemodel`). See the download snippet below.
- `ffmpeg` (optional) for quick video sanity checks.

## 📂 Data Preparation

1. Download and unzip the KTH dataset so that raw videos live under `data/kth_raw/<class>/personXX_<class>_dY_uncomp.avi` (already prepared in this repo).
2. Run the OpenCV pose extractor across the dataset. This script produces compressed NumPy pose tensors with timestamps and metadata (no external OpenPose binary required).

```bash
python scripts/run_openpose.py \
	--video-root data/kth_raw \
	--output-root data/pose_keypoints \
	--model-config models/opencv_pose/pose_deploy_linevec.prototxt \
	--model-weights models/opencv_pose/pose_iter_440000.caffemodel \
	--frame-stride 2 \
	--batch-size 16 \
	--workers 0 \
	--input-width 192 \
	--input-height 192
```

- Final `.npz` pose clips live under `data/pose_keypoints/npz/` and contain:
	- `poses`: `(T, num_joints, 2)` array (x, y).
	- `timestamps`: seconds from video start.
	- `meta`: JSON blob (action label, fps, relative path, etc.).

> **Downloading the OpenCV model files**
>
> ```bash
> python scripts/download_pose_models.py --dest models/opencv_pose
> ```
>
> The helper pulls both files from Hugging Face mirrors (`gaijingeek/openpose-models`) and verifies their SHA-256 checksums. Re-run with `--force` to refresh the cache, or pass custom paths via `--model-config` / `--model-weights` if you keep the assets elsewhere.

**Performance tips:**
- `--input-width 192 --input-height 192` (default: 256×256) – Smaller input = 2-3× faster inference with minimal accuracy loss for KTH's low-res videos.
- `--batch-size 16` (default: 1) – Process multiple frames per forward pass; batching gives ~2× speedup.
- `--workers 0` (default: 0 = auto CPU count) – Parallel video processing; scales linearly with cores.
- `--frame-stride 2-4` – Skip frames for even coarser (but faster) temporal sampling.
- `--actions boxing handclapping` – Process only specific action subsets.

## 🧠 Training & Inference

Train the temporal pose autoencoder (default: 64-frame windows, stride 32):

```bash
python scripts/train_anomaly_cnn.py \
	--pose-dir data/pose_keypoints/npz \
	--batch-size 32 \
	--epochs 30 \
	--device cuda
```

Outputs:

- `models/checkpoints/pose_autoencoder.pt` – trained weights.
- `results/anomaly_threshold.json` – reconstruction stats & default threshold (95th percentile of training reconstruction error).

### Evaluate / score sequences

```bash
python scripts/train_anomaly_cnn.py \
	--mode eval \
	--pose-dir data/pose_keypoints/npz \
	--checkpoint models/checkpoints/pose_autoencoder.pt \
	--eval-threshold 0.12
```

Evaluation prints the per-clip reconstruction error and flags `anomaly=True` when the score exceeds the provided (or stored) threshold. You can craft custom fall clips and feed them through the same `.npz` spec for zero-shot anomaly detection.

## 🧩 Project Structure

```
data/
	kth_raw/           # raw KTH .avi files (6 actions)
	pose_keypoints/
		npz/             # Cleaned pose+timestamp tensors for training
models/
	cnn_model.py       # Pose autoencoder definition
results/             # Training logs, anomaly threshold, plots
scripts/
	run_openpose.py    # Video ingestion + OpenCV pose extractor + parser
	train_anomaly_cnn.py
utils/
	kth_dataset.py     # Video discovery helpers
	pose_processing.py # Pose normalization + dataset utilities
tests/
	test_pose_dataset.py
```

## 🔭 Next ideas

- Add a lightweight fall-simulation augmentation (e.g., vertical velocity spike) to better calibrate anomaly thresholds.
- Train a one-class SVM (latent space) or use contrastive pretraining to tighten the manifold of normal motions.
- Plug the trained scorer into the `web_app/` folder to build a demo dashboard.

## 🤝 Contributing

Pull requests are welcome! Please run `pytest` plus any formatting tools you add, and document new scripts in this README.

