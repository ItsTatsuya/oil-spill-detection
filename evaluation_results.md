# Oil Spill Detection Model Evaluation Results

Evaluation Date: 2025-04-21 12:58:06

## Mean IoU Comparison

| Model | Mean IoU (%) |
|-------|-------------|
| improved_deeplabv3plus_best | 64.15 |
| Baseline | 65.06 |

## Class-wise IoU Comparison

| Model | Sea Surface | Oil Spill | Look-alike | Ship | Land |
|-------|------|------|------|------|------|
| improved_deeplabv3plus_best | 96.62 | 54.88 | 60.80 | 18.31 | 90.17 |
| Baseline | 96.43 | 53.38 | 55.40 | 27.63 | 92.44 |

## Inference Time Comparison

| Model | Single-Scale (ms) | Multi-Scale (ms) |
|-------|-------------------|------------------|
| improved_deeplabv3plus_best | 463.48 | 1393.67 |

![Class-wise IoU Comparison](class_ious_comparison.png)
