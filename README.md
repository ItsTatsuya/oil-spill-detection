# Oil Spill Detection using DeepLabV3+ with EfficientNet-B4

## Project Overview

This project implements a deep learning model for detecting oil spills in satellite imagery using semantic segmentation. The model uses a DeepLabV3+ architecture with EfficientNet-B4 backbone to classify each pixel of satellite images into five categories: Sea Surface, Oil Spill, Look-alike, Ship, and Land.

## Model Architecture

- **Architecture**: DeepLabV3+ with enhanced decoder (3 conv layers instead of 2)
- **Backbone**: EfficientNet-B4 (pretrained on ImageNet)
- **ASPP**: Atrous rates [6, 12, 18] with global pooling branch
- **Decoder**: Three 3x3 convolutions (256 filters each)
- **Input Size**: 320×320×3
- **Output**: 5 classes (Sea Surface, Oil Spill, Look-alike, Ship, Land)

## Features

1. **Multi-scale Training and Prediction**: Fuses predictions from different scales (50%, 75%, 100%)
2. **Advanced Data Augmentation**:
   - Per-image augmentations: rotation, flipping, color adjustments
   - Batch-level augmentations: CutMix and MixUp (using Keras-CV)
3. **Hybrid Loss Function**: Combines weighted cross-entropy and Dice loss
4. **Learning Rate Scheduler**: Reduces learning rate when performance plateaus
5. **Early Stopping**: Prevents overfitting

## File Structure

- **model.py**: DeepLabV3+ implementation with EfficientNet-B4 backbone
- **train.py**: Training script with batch augmentation and learning rate scheduling
- **data_loader.py**: Dataset loading and preprocessing
- **augmentation.py**: Custom augmentation pipeline with Keras-CV integration
- **loss.py**: Hybrid loss function implementation
- **evaluate.py**: Model evaluation and visualization tools

## Data Augmentation

The project uses a robust data augmentation pipeline including:

- **Spatial augmentations**: rotation (±10°), horizontal flipping
- **Color augmentations**: brightness, contrast, saturation adjustments
- **Noise injection**: random noise with 30% probability
- **Advanced batch augmentations**: CutMix and MixUp through Keras-CV

## Training Process

The model is trained using:

- **Optimizer**: Adam with gradient clipping (learning rate 5e-5)
- **Batch Size**: 8
- **Epochs**: Up to 600 with early stopping
- **Early Stopping Patience**: 60 epochs (monitoring validation IoU)
- **Learning Rate Schedule**: Reduce on plateau with factor 0.5

## Performance

The model is evaluated using Mean IoU (Intersection over Union) metric and class-specific IoU scores for all five classes.

## Requirements

- TensorFlow 2.x
- Keras-CV (for advanced augmentations)
- NumPy
- Matplotlib
- tqdm

## Usage

1. **Setup Environment**:

   ```
   python -m venv oilspill
   source oilspill/bin/activate
   pip install -r requirements.txt
   ```

2. **Training**:

   ```
   python train.py
   ```

3. **Evaluation**:
   ```
   python evaluate.py
   ```

## Dataset

The model is designed to work with satellite imagery containing oil spills. The dataset should be organized as follows:

```
dataset/
  ├── train/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
```

Each label image is a single-channel image with pixel values representing the class (0-4).

## Future Work

- Integration with object detection for ship identification
- Testing additional backbones (EfficientNetV2, ConvNeXt)
- Deployment optimizations for edge devices
