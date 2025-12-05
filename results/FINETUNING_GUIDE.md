# Fine-tuning on Kaggle Fall Dataset

This script fine-tunes the pretrained KTH autoencoder on the Kaggle Fall dataset using supervised learning, while keeping the original KTH model separate.

## Overview

### Architecture

The fine-tuning adds a **classification head** on top of the pretrained autoencoder:

```
Input Pose Sequence (2, 64, 18)
    ↓
Pretrained Autoencoder (frozen or trainable)
    ↓
Latent Vector (128)
    ↓
Classification Head (128 → 64 → 32 → 2)
    ↓
Binary Classification (Fall vs No Fall)
```

### Training Strategy

1. **Transfer Learning**: Uses pretrained KTH autoencoder weights
2. **Supervised Learning**: Trains with labeled Fall/No-Fall data
3. **Dual Loss**: Combines classification loss + reconstruction loss
4. **Fine-tuning Options**:
   - Full fine-tuning (train all weights)
   - Freeze encoder (only train classifier)
   - From scratch (no pretrained weights)

## Usage

### Basic Fine-tuning

Uses pretrained KTH model and trains on full Kaggle dataset:

```bash
cd /home/amirali/Projects/fall-detection-study
.venv/bin/python scripts/finetune_kaggle.py \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-4 \
    --device cpu
```

### Quick Test (Small Subset)

Test with 100 samples per class, 10 epochs:

```bash
.venv/bin/python scripts/finetune_kaggle.py \
    --max-samples-per-class 100 \
    --epochs 10 \
    --batch-size 8 \
    --device cpu \
    --num-workers 0
```

### Freeze Encoder (Train Only Classifier)

Faster training, prevents overfitting on small datasets:

```bash
.venv/bin/python scripts/finetune_kaggle.py \
    --freeze-encoder \
    --epochs 15 \
    --batch-size 32 \
    --lr 1e-3
```

### Train from Scratch (No Pretrained Weights)

```bash
.venv/bin/python scripts/finetune_kaggle.py \
    --from-scratch \
    --epochs 50 \
    --batch-size 16 \
    --lr 5e-3
```

### Full Dataset Training (Recommended)

Train on all 6,988 Kaggle videos:

```bash
# With GPU (if available)
.venv/bin/python scripts/finetune_kaggle.py \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --device cuda \
    --num-workers 4

# With CPU (slower, ~2-3 hours)
.venv/bin/python scripts/finetune_kaggle.py \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-4 \
    --device cpu \
    --num-workers 2
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--kaggle-dir` | `data/KaggleRealDataset` | Path to Kaggle dataset |
| `--pretrained-checkpoint` | `models/checkpoints/pose_autoencoder.pt` | Pretrained KTH model |
| `--output-checkpoint` | `models/checkpoints/pose_autoencoder_kaggle_finetuned.pt` | Output path for fine-tuned model |
| `--sequence-length` | `64` | Number of frames per sequence |
| `--batch-size` | `16` | Batch size for training |
| `--epochs` | `20` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate (lower for fine-tuning) |
| `--val-split` | `0.2` | Validation split ratio |
| `--test-split` | `0.1` | Test split ratio |
| `--max-samples-per-class` | `None` | Limit samples (for testing) |
| `--device` | `cuda/cpu` | Device for training |
| `--num-workers` | `4` | Data loading workers |
| `--freeze-encoder` | `False` | Freeze encoder weights |
| `--from-scratch` | `False` | Train without pretrained weights |

## Output Files

### Model Checkpoint
- **Location**: `models/checkpoints/pose_autoencoder_kaggle_finetuned.pt`
- **Contents**: Best model weights based on validation accuracy
- **Note**: Original KTH model remains at `pose_autoencoder.pt`

### Training History
- **Location**: `models/checkpoints/kaggle_finetuning_history.json`
- **Contents**: Per-epoch metrics (loss, accuracy, learning rate)

## Expected Performance

### Small Dataset (100-200 samples per class)
- **Training Time**: 2-5 minutes (CPU), <1 minute (GPU)
- **Expected Accuracy**: 60-70%
- **Note**: May overfit, use --freeze-encoder

### Medium Dataset (500-1000 samples per class)
- **Training Time**: 10-20 minutes (CPU), 2-5 minutes (GPU)
- **Expected Accuracy**: 70-80%
- **Recommended**: Full fine-tuning with dropout

### Full Dataset (3,140 Fall + 3,848 No Fall)
- **Training Time**: 1-2 hours (CPU), 15-30 minutes (GPU)
- **Expected Accuracy**: 80-90%
- **Recommended**: Full fine-tuning, careful hyperparameter tuning

## Data Splits

The script automatically splits data:
- **Training**: 70% (used for training)
- **Validation**: 20% (used for model selection)
- **Test**: 10% (final evaluation, unseen during training)

Splits are stratified to maintain class balance.

## Tips for Best Results

### 1. Start Small
Test with `--max-samples-per-class 100` first to verify everything works

### 2. Monitor Overfitting
Watch for gap between train and validation accuracy:
- **Small gap (<5%)**: Good, increase epochs
- **Large gap (>15%)**: Overfitting, use --freeze-encoder or add regularization

### 3. Learning Rate
- **Fine-tuning**: 1e-4 (default, safe)
- **Freeze encoder**: 1e-3 (faster, more stable)
- **From scratch**: 5e-3 (needs higher LR)

### 4. Batch Size
- **CPU**: 8-16 (memory limited)
- **GPU**: 32-64 (faster training)

### 5. Early Stopping
Model automatically uses learning rate scheduling and saves best checkpoint

## Troubleshooting

### Issue: Model predicts all Falls or all No-Falls
**Solution**: 
- Increase epochs (try 30-50)
- Increase batch size
- Use more data (remove --max-samples-per-class)
- Try --from-scratch

### Issue: Out of memory
**Solution**:
- Reduce --batch-size (try 8 or 4)
- Reduce --num-workers to 0
- Use --device cpu

### Issue: Very slow training
**Solution**:
- Increase --batch-size
- Set --num-workers 4 (or 0 if causing issues)
- Use --freeze-encoder (faster)
- Use GPU with --device cuda

### Issue: Validation accuracy not improving
**Solution**:
- Increase learning rate
- Remove --freeze-encoder
- Increase epochs
- Check data quality (inspect failed samples)

## Comparison with KTH Model

| Model | Training Data | Accuracy | Use Case |
|-------|--------------|----------|----------|
| **KTH Only** | 599 normal activity videos | 51-55% on Kaggle | Unsupervised anomaly detection |
| **Fine-tuned (Small)** | 100 Fall + 100 No-Fall | 65-75% | Quick prototyping |
| **Fine-tuned (Full)** | 3,140 Fall + 3,848 No-Fall | 80-90% | Production deployment |

## Next Steps After Fine-tuning

1. **Evaluate on test set**: Results shown automatically
2. **Test on new data**: Use the fine-tuned model with test_kaggle_dataset.py
3. **Deploy**: Use fine-tuned model in production
4. **Iterate**: Adjust hyperparameters, try different architectures

## Example Complete Workflow

```bash
# 1. Quick test (5 minutes)
.venv/bin/python scripts/finetune_kaggle.py \
    --max-samples-per-class 50 \
    --epochs 10 \
    --batch-size 8

# 2. Medium test (20 minutes)
.venv/bin/python scripts/finetune_kaggle.py \
    --max-samples-per-class 500 \
    --epochs 20 \
    --batch-size 16

# 3. Full training (1-2 hours)
.venv/bin/python scripts/finetune_kaggle.py \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4

# 4. Test fine-tuned model
.venv/bin/python scripts/test_kaggle_dataset.py \
    --checkpoint models/checkpoints/pose_autoencoder_kaggle_finetuned.pt \
    --max-samples 200
```

## Model Files

After training, you will have:

1. **Original KTH Model**: `models/checkpoints/pose_autoencoder.pt`
   - Trained on normal activities only
   - Used for unsupervised anomaly detection

2. **Fine-tuned Kaggle Model**: `models/checkpoints/pose_autoencoder_kaggle_finetuned.pt`
   - Trained on Fall/No-Fall labels
   - Used for supervised classification
   - Better accuracy on real-world data

Both models are kept separate and can be used for different purposes!
