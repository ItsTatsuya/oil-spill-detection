# Oil Spill Detection Model Evaluation Results

Evaluation Date: 2025-04-21 08:08:21

## Mean IoU Comparison

| Model | Mean IoU (%) |
|-------|-------------|
| improved_deeplabv3plus_best.weights | 56.57 |
| latest_model.weights | 55.74 |

## Class-wise IoU Comparison

| Model | Sea Surface | Oil Spill | Look-alike | Ship | Land |
|-------|------|------|------|------|------|
| improved_deeplabv3plus_best.weights | 96.00 | 47.31 | 53.46 | 2.54 | 83.53 |
| latest_model.weights | 95.61 | 47.55 | 51.33 | 2.14 | 82.06 |

## Inference Time Comparison

| Model | Single-Scale (ms) | Multi-Scale (ms) |
|-------|-------------------|------------------|
| improved_deeplabv3plus_best.weights | 313.19 | 962.33 |
| latest_model.weights | 325.48 | 959.03 |

![Class-wise IoU Comparison](class_ious_comparison.png)
