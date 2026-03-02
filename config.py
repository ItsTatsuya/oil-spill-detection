"""
Central configuration for oil spill detection training and evaluation.

All hyperparameters consolidated in one place for easy experiment management.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    data_dir: str = 'dataset'
    img_size: Tuple[int, int] = (384, 384)
    num_classes: int = 5
    channels: int = 1
    val_split: float = 0.15  # fraction of training data held out for validation
    class_names: Tuple[str, ...] = (
        'Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land'
    )
    class_weights_cache: str = 'dataset/class_weights.npy'
    shuffle_buffer: int = 2000


@dataclass
class AugmentationConfig:
    """Training-time augmentation configuration."""
    # Flips
    p_horizontal_flip: float = 0.5
    p_vertical_flip: float = 0.3

    # Rotation
    max_rotation_deg: float = 15.0
    p_90deg_rotation: float = 0.2

    # Speckle noise (SAR-specific)
    speckle_stddev: float = 0.15
    p_speckle: float = 0.7

    # Brightness / contrast
    p_brightness: float = 0.4
    brightness_delta: float = 0.2
    p_contrast: float = 0.4
    contrast_range: Tuple[float, float] = (0.8, 1.2)

    # Gamma correction
    p_gamma: float = 0.3
    gamma_range: Tuple[float, float] = (0.7, 1.5)

    # Gaussian blur
    p_blur: float = 0.2
    blur_sigma_range: Tuple[float, float] = (0.5, 1.5)

    # Cutout / random erasing
    p_cutout: float = 0.3
    cutout_size_range: Tuple[float, float] = (0.05, 0.15)  # fraction of image
    cutout_max_patches: int = 3


@dataclass
class LossConfig:
    """Loss function configuration."""
    ce_weight: float = 0.35
    focal_weight: float = 0.25
    dice_weight: float = 0.3
    boundary_weight: float = 0.1
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    boundary_boost: float = 2.5
    epsilon: float = 1e-7
    from_logits: bool = True


@dataclass
class TrainConfig:
    """Training loop configuration."""
    # Basic
    batch_size: int = 256           # Global batch size across all GPUs
    per_gpu_batch_size: int = 32    # Per-GPU batch size
    epochs: int = 600
    num_classes: int = 5

    # Learning rate (sqrt scaling: 3e-5 * sqrt(256/2) ≈ 3.4e-4)
    learning_rate: float = 3.4e-4
    warmup_epochs: int = 25
    cosine_alpha: float = 0.001  # minimum LR fraction

    # Optimizer
    weight_decay: float = 1e-4
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7
    clipnorm: float = 1.0
    amsgrad: bool = True

    # Callbacks
    early_stopping_patience: int = 30
    reduce_lr_patience: int = 15
    reduce_lr_factor: float = 0.5
    reduce_lr_min: float = 1e-7

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # Distributed
    use_distributed: bool = True

    # Parallelism (0 = auto-detect)
    inter_op_threads: int = 0
    intra_op_threads: int = 0

    # Paths
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    pretrained_weights: str = 'pretrained_weights/segformer_b2_pretrain.weights.h5'

    # Reproducibility
    seed: int = 42


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    batch_size: int = 8
    scales_with_tta: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5]
    )
    scales_no_tta: List[float] = field(
        default_factory=lambda: [0.75, 1.0, 1.25]
    )
    tta_num_augmentations: int = 8
    use_tta: bool = True
    use_ship_postprocessing: bool = True

    # Ship post-processing thresholds
    ship_probability_threshold: float = 0.25
    ship_min_size: int = 8
    ship_boost_factor: float = 1.35
    ship_suppression_factor: float = 0.7

    # Bootstrap CI
    bootstrap_iterations: int = 1000
    bootstrap_confidence: float = 0.95

    # Model path
    model_path: str = 'checkpoints/segformer_b2_best.weights.h5'
