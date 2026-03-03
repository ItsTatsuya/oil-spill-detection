# Oil Spill Detection in SAR Imagery Using SegFormer-B2

**Semantic segmentation of Synthetic Aperture Radar (SAR) satellite images for automated oil spill detection in maritime environments.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Abstract

Oil spills in marine environments cause severe ecological damage, making timely and accurate detection critical for rapid response. This project presents a fully engineered training and evaluation pipeline for oil spill detection in SAR imagery using **SegFormer-B2**, a hierarchical vision transformer (~27.99 M parameters). The system classifies each pixel into five semantic classes — _Sea Surface_, _Oil Spill_, _Look-alike_, _Ship_, and _Land_ — without modifying the base model architecture. Key contributions include: (1) a **hybrid segmentation loss** combining weighted cross-entropy, focal, Dice, and boundary terms; (2) **smart patch sampling** that up-samples rare classes during training; (3) **multi-GPU distributed training** with mixed-precision and EMA weight averaging; and (4) **multi-scale test-time augmentation** with ship-specific post-processing. The pipeline is designed for 8× NVIDIA A100 GPUs and trains with a global batch size of 256.

---

## Method Overview

### Architecture

The backbone is **SegFormer-B2** (Mix Vision Transformer) with a lightweight all-MLP decoder, enhanced with a **CBAM** (Convolutional Block Attention Module) for SAR-specific feature refinement.

| Component            | Specification                              |
| :------------------- | :----------------------------------------- |
| Backbone             | Mix Vision Transformer (MiT-B2)            |
| Input resolution     | 384 × 384 × 1 (grayscale SAR)              |
| Embedding dimensions | [64, 128, 320, 512]                        |
| Transformer layers   | [3, 4, 6, 3]                               |
| Attention heads      | [1, 2, 5, 8]                               |
| MLP ratios           | [8, 8, 4, 4]                               |
| Decoder embedding    | 768                                        |
| Output               | 384 × 384 × 5 (full-resolution, 5 classes) |
| Parameters           | ~27.99 M                                   |
| Attention module     | CBAM (channel + 7×7 spatial)               |

### Hybrid Loss Function

A multi-component loss is designed to address class imbalance, hard-example mining, and boundary fidelity:

$$\mathcal{L} = 0.35 \cdot \mathcal{L}_{\text{CE}} + 0.25 \cdot \mathcal{L}_{\text{Focal}} + 0.30 \cdot \mathcal{L}_{\text{Dice}} + 0.10 \cdot \mathcal{L}_{\text{Boundary}}$$

| Term                   | Purpose                                   | Key Parameters                     |
| :--------------------- | :---------------------------------------- | :--------------------------------- |
| Weighted Cross-Entropy | Per-class balancing via inverse-frequency | Label smoothing ε = 0.1            |
| Focal Loss             | Down-weights easy examples                | γ = 2.0                            |
| Dice Loss              | Region-level overlap optimisation         | Per-class soft Dice                |
| Boundary Loss          | Penalises errors at class boundaries      | Sobel edge detection, boost = 2.5× |

Class weights are computed from inverse pixel frequency across the training set and cached to avoid recomputation.

### Training Strategy

| Hyperparameter    | Value                                                    |
| :---------------- | :------------------------------------------------------- |
| Optimizer         | AdamW (β₁ = 0.9, β₂ = 0.999, ε = 1e-7, AMSGrad)          |
| Learning rate     | 3.4 × 10⁻⁴ (√-scaled from 3 × 10⁻⁵ for batch 256)        |
| LR schedule       | Cosine decay with 25-epoch linear warmup (α_min = 0.001) |
| LR reduction      | ReduceLROnPlateau (factor = 0.5, patience = 15)          |
| Batch size        | 256 global (32 per GPU × 8 GPUs)                         |
| Epochs            | 600 max                                                  |
| Early stopping    | Patience = 30 (monitored on val mIoU)                    |
| Weight decay      | 1 × 10⁻⁴                                                 |
| Gradient clipping | `clip_grad_norm_` max-norm = 1.0                         |
| EMA               | Exponential moving average, decay = 0.999 (`torch-ema`)  |
| Mixed precision   | PyTorch AMP (`autocast` + `GradScaler`)                  |
| Distribution      | PyTorch DDP via `torchrun --nproc_per_node=N`            |
| Reproducibility   | Global seed = 42                                         |

### Data Pipeline

**Train/Val/Test split**: The official training set is deterministically split 85/15 at the file level into training and validation subsets. The test set is held out entirely for final evaluation.

**Smart patch sampling** (training only): Images are loaded at original resolution and a 384×384 crop is extracted. With 50% probability, the crop is centred on an Oil Spill or Ship pixel, naturally up-sampling rare classes without duplicating images.

**Augmentation pipeline** (applied per sample):

| Transform        | Details                                      | Probability |
| :--------------- | :------------------------------------------- | :---------: |
| Horizontal flip  | Mirror left–right                            |     50%     |
| Vertical flip    | Mirror top–bottom                            |     30%     |
| Rotation         | ±15° affine (reflect fill) or 90° discrete   |  80% / 20%  |
| Speckle noise    | Multiplicative N(0, σ = 0.15) — SAR-specific |     70%     |
| Brightness       | ±0.2 delta                                   |     40%     |
| Contrast         | Factor ∈ [0.8, 1.2]                          |     40%     |
| Gamma correction | γ ∈ [0.7, 1.5]                               |     30%     |
| Gaussian blur    | 5×5, σ ∈ [0.5, 1.5]                          |     20%     |
| Cutout           | 1–3 patches, 5–15% of image each             |     30%     |

Validation and test sets are resized to 384×384 deterministically (no augmentation). The pipeline uses PyTorch `DataLoader` with configurable worker processes (`num_workers = 4`).

### Inference

**Multi-scale prediction**: Input is evaluated at scales {0.5, 0.75, 1.0, 1.25, 1.5} and predictions are averaged.

**Test-time augmentation (TTA)**: 8 geometric augmentations (flips + rotations) are applied at each scale; softmax probabilities are averaged before argmax.

**Ship post-processing**: Morphological filtering with probability boosting (1.35×) for the Ship class to recover small detections.

---

## Dataset

The **ROBORDER Oil Spill Detection Dataset** contains SAR images annotated with five semantic classes:

| Class ID | Label       | Description                                             |
| :------: | :---------- | :------------------------------------------------------ |
|    0     | Sea Surface | Background ocean water                                  |
|    1     | Oil Spill   | Petroleum / oil spill regions                           |
|    2     | Look-alike  | Natural phenomena mimicking oil (wind slicks, biogenic) |
|    3     | Ship        | Maritime vessels                                        |
|    4     | Land        | Coastal and land regions                                |

```
dataset/
├── train/
│   ├── images/         # SAR .jpg images
│   ├── labels/         # RGB label visualisations
│   └── labels_1D/      # Single-channel masks (pixel values 0–4)
└── test/
    ├── images/
    ├── labels/
    └── labels_1D/
```

The dataset exhibits extreme class imbalance — Sea Surface and Land dominate, while Ship pixels constitute < 0.1% of the total. The pipeline addresses this through inverse-frequency class weighting and smart patch sampling.

---

## Results

### Baseline Comparison

| Metric          | DeepLabV3+ (baseline) | SegFormer-B2 (ours) |   Δ    |
| :-------------- | :-------------------: | :-----------------: | :----: |
| **mIoU**        |        65.06%         |       66.38%        | +1.32% |
| Sea Surface IoU |           —           |       95.81%        |   —    |
| Oil Spill IoU   |           —           |       54.65%        |   —    |
| Look-alike IoU  |           —           |       55.84%        |   —    |
| Ship IoU        |           —           |       32.12%        | +4.49% |
| Land IoU        |           —           |       93.48%        |   —    |

> **Note**: Results above were obtained with an earlier configuration (batch size 2, single GPU, LR = 3 × 10⁻⁵). The current pipeline with distributed training, EMA, and the improved loss function is expected to yield further gains upon retraining.

### Evaluation Metrics

The evaluation script computes:

- **Mean IoU** and per-class IoU
- **Pixel Accuracy** (overall and per-class)
- **Frequency-Weighted IoU** (FWIoU)
- **Cohen's Kappa** coefficient
- **Precision, Recall, F1-score** per class
- **Bootstrap 95% Confidence Intervals** (1 000 iterations)

---

## Repository Structure

```
oil-spill-detection/
├── config.py                       # Centralised hyperparameter dataclasses
├── utils.py                        # Reproducibility, colour maps (pure PyTorch)
├── train.py                        # Training loop with AMP, DDP, EMA & TensorBoard
├── evaluate.py                     # Full evaluation + metric reporting
├── working.md                      # Detailed pipeline documentation & flowchart
├── data/
│   ├── __init__.py
│   ├── data_loader.py              # Train/val/test split, class weights, DataLoader
│   ├── augmentation.py             # SAR-specific augmentation transforms
│   └── test_time_augmentation.py   # TTA engine for inference
├── model/
│   ├── __init__.py
│   ├── model.py                    # SegFormer-B2 architecture (pure PyTorch)
│   ├── loss.py                     # Hybrid loss (CE + Focal + Dice + Boundary)
│   ├── metrics.py                  # IoU, accuracy, FWIoU, Kappa, bootstrap CI
│   └── prediction.py              # Multi-scale predictor with optional TTA
└── dataset/                        # ROBORDER dataset (see Dataset section)
```

---

## Getting Started

### Prerequisites

- Python ≥ 3.10
- NVIDIA GPU(s) with CUDA 12.x and cuDNN 9.x
- PyTorch ≥ 2.2 with matching CUDA toolkit

### Installation

```bash
# Clone the repository
git clone https://github.com/ItsTatsuya/oil-spill-detection.git
cd oil-spill-detection

# Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` installs: `torch`, `torchvision`, `timm`, `torchmetrics`, `huggingface-hub`, `torch-ema`, `tensorboard`, `numpy`, `opencv-python`, `matplotlib`, and `tqdm`.

Place the ROBORDER dataset under `dataset/` following the directory structure shown above.

### Training

```bash
# Single GPU or CPU
python train.py

# Multi-GPU (e.g. 2 GPUs on one machine)
torchrun --nproc_per_node=2 train.py
```

All hyperparameters are controlled via `config.py`. The script auto-detects available GPUs and launches single-GPU, multi-GPU DDP, or CPU runs accordingly. Checkpoints (`.pt`), TensorBoard logs, and the best model weights are saved to `checkpoints/` and `logs/`.

### Evaluation

```bash
python evaluate.py
```

By default, evaluation loads the best checkpoint from `checkpoints/segformer_b2_best.pt`, runs multi-scale TTA, and writes results to `evaluation_results.md`.

---

## Configuration

All settings are defined as Python dataclasses in `config.py`:

| Dataclass            | Scope                                                             |
| :------------------- | :---------------------------------------------------------------- |
| `DataConfig`         | Image size, val split, class names, class weight cache path       |
| `AugmentationConfig` | Every augmentation probability and parameter range                |
| `LossConfig`         | Component weights, focal γ, label smoothing, boundary boost       |
| `TrainConfig`        | Batch size, LR, warmup, EMA, early stopping, optimiser, paths     |
| `EvalConfig`         | TTA scales, ship post-processing thresholds, bootstrap iterations |

Modify `config.py` to run new experiments without changing any other file.
