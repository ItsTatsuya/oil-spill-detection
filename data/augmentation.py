"""
Training-time augmentation for SAR oil-spill imagery — PyTorch implementation.

Convention
----------
- image : torch.Tensor [C, H, W]  float32, values in [0, 1]
- label : torch.Tensor [H, W]     long (int64)

All functions are plain Python (no @tf.function); composable in __getitem__.
"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# True continuous rotation via affine transform
# ---------------------------------------------------------------------------
def rotate_image(image: torch.Tensor, angle_rad: float, method: str = 'bilinear') -> torch.Tensor:
    """Rotate a [C, H, W] tensor by *angle_rad* radians using an affine transform.

    REFLECT padding is approximated by REFLECT in torchvision (fill=0 fallback).
    """
    angle_deg = math.degrees(angle_rad)
    interp = TF.InterpolationMode.BILINEAR if method == 'bilinear' else TF.InterpolationMode.NEAREST
    return TF.affine(
        image,
        angle=angle_deg,
        translate=[0, 0],
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=interp,
        fill=[0.0],
    )


# ---------------------------------------------------------------------------
# Core augmentation primitives
# ---------------------------------------------------------------------------
def random_flip(image: torch.Tensor, label: torch.Tensor,
                p_horizontal: float = 0.5, p_vertical: float = 0.3):
    """Horizontal / vertical flip."""
    if random.random() < p_horizontal:
        image = torch.flip(image, [2])   # flip W  (NCHW → dim 2 is W for [C,H,W])
        label = torch.flip(label, [1])   # flip W  ([H,W] → dim 1)
    if random.random() < p_vertical:
        image = torch.flip(image, [1])   # flip H
        label = torch.flip(label, [0])   # flip H
    return image, label


def random_rotation(image: torch.Tensor, label: torch.Tensor,
                    max_angle: float = 15.0, p_90deg: float = 0.2):
    """90°-step or continuous rotation."""
    if random.random() < p_90deg:
        k = random.randint(1, 3)
        image = torch.rot90(image, k, dims=[1, 2])
        label = torch.rot90(label, k, dims=[0, 1])
    else:
        max_rad = max_angle * (math.pi / 180.0)
        angle = random.uniform(-max_rad, max_rad)
        image = rotate_image(image, angle, method='bilinear')
        label = rotate_image(label.unsqueeze(0).float(), angle, method='nearest')
        label = label.squeeze(0).long()
    return image, label


def add_speckle_noise(image: torch.Tensor, label: torch.Tensor,
                      mean: float = 0.0, stddev: float = 0.15, p: float = 0.7):
    """Multiplicative speckle noise — models SAR sensor noise."""
    if random.random() < p:
        noise = torch.randn_like(image) * stddev + mean
        image = (image * (1.0 + noise)).clamp(0.0, 1.0)
    return image, label


# ---------------------------------------------------------------------------
# SAR-specific augmentations
# ---------------------------------------------------------------------------
def random_brightness_contrast(image: torch.Tensor, label: torch.Tensor,
                                p_bright: float = 0.4, delta: float = 0.2,
                                p_contrast: float = 0.4,
                                low: float = 0.8, high: float = 1.2):
    """Simulate SAR back-scatter variation with incidence angle."""
    img = image.float()
    if random.random() < p_bright:
        d = random.uniform(-delta, delta)
        img = (img + d).clamp(0.0, 1.0)
    if random.random() < p_contrast:
        factor = random.uniform(low, high)
        img = ((img - 0.5) * factor + 0.5).clamp(0.0, 1.0)
    return img, label


def random_gamma(image: torch.Tensor, label: torch.Tensor,
                 p: float = 0.3, low: float = 0.7, high: float = 1.5):
    """Gamma correction: SAR intensity follows non-linear distributions."""
    if random.random() < p:
        gamma = random.uniform(low, high)
        img = torch.pow(image.float().clamp(1e-7, 1.0), gamma)
        return img.clamp(0.0, 1.0), label
    return image, label


def random_gaussian_blur(image: torch.Tensor, label: torch.Tensor,
                         p: float = 0.2, sigma_low: float = 0.5, sigma_high: float = 1.5):
    """Approximate Gaussian blur to simulate varying sensor resolutions."""
    if random.random() < p:
        sigma = random.uniform(sigma_low, sigma_high)
        # kernel_size must be odd; pick 5 for the same coverage as original
        blurred = TF.gaussian_blur(image, kernel_size=[5, 5], sigma=[sigma, sigma])
        return blurred.clamp(0.0, 1.0), label
    return image, label


def random_cutout(image: torch.Tensor, label: torch.Tensor,
                  p: float = 0.3, size_low: float = 0.05, size_high: float = 0.15,
                  max_patches: int = 3):
    """Random erasing (cutout) for regularisation."""
    if random.random() < p:
        img = image.clone().float()
        C, H, W = img.shape
        n_patches = random.randint(1, max_patches)
        for _ in range(n_patches):
            frac = random.uniform(size_low, size_high)
            ch = max(1, int(H * frac))
            cw = max(1, int(W * frac))
            max_y = max(0, H - ch)
            max_x = max(0, W - cw)
            cy = random.randint(0, max_y)
            cx = random.randint(0, max_x)
            img[:, cy:cy + ch, cx:cx + cw] = 0.0
        return img.clamp(0.0, 1.0), label
    return image, label


# ---------------------------------------------------------------------------
# Composite augmentation
# ---------------------------------------------------------------------------
def augment_single_sample(image: torch.Tensor, label: torch.Tensor):
    """Apply the full augmentation pipeline to a single (image, label) pair.

    Parameters
    ----------
    image : [C, H, W]  float32
    label : [H, W]     long

    Returns
    -------
    image : [C, H, W]  float32
    label : [H, W]     long
    """
    image = image.float()
    label = label.long()

    # Geometric
    image, label = random_flip(image, label)
    image, label = random_rotation(image, label, max_angle=15.0, p_90deg=0.2)

    # Intensity / SAR-specific
    image, label = add_speckle_noise(image, label, stddev=0.15)
    image, label = random_brightness_contrast(image, label)
    image, label = random_gamma(image, label)
    image, label = random_gaussian_blur(image, label)

    # Regularisation
    image, label = random_cutout(image, label)

    return image.clamp(0.0, 1.0).float(), label.long()


def apply_augmentation(samples, batch_size: int = 2):
    """Apply augmentation to a list of (image, label) tensors (used at batch level).

    This mirrors the old tf.data `apply_augmentation` signature for compatibility;
    in the PyTorch pipeline augmentation typically happens in Dataset.__getitem__.

    Parameters
    ----------
    samples : list of (image [C,H,W], label [H,W]) tuples  —or—  (batch_imgs, batch_labels)

    Returns
    -------
    aug_images : Tensor [N, C, H, W]
    aug_labels : Tensor [N, H, W]
    """
    if isinstance(samples, (list, tuple)) and isinstance(samples[0], (list, tuple)):
        imgs, lbls = zip(*[augment_single_sample(img, lbl) for img, lbl in samples])
        return torch.stack(list(imgs)), torch.stack(list(lbls))
    # If called with (batch_imgs, batch_labels) tensors directly:
    batch_imgs, batch_labels = samples
    B = batch_imgs.shape[0]
    aug_imgs, aug_lbls = [], []
    for i in range(B):
        ai, al = augment_single_sample(batch_imgs[i], batch_labels[i])
        aug_imgs.append(ai)
        aug_lbls.append(al)
    return torch.stack(aug_imgs), torch.stack(aug_lbls)


