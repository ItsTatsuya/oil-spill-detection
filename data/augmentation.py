"""
Training-time augmentation for SAR oil-spill imagery.

Key improvements:
- True continuous rotation via affine transform (replaces rot90 quantisation)
- SAR-specific augmentations: brightness/contrast, gamma correction, Gaussian blur
- Random erasing (cutout) for regularisation
- Configurable via AugmentationConfig
"""

import math
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# True continuous rotation via affine transform
# ---------------------------------------------------------------------------
@tf.function(reduce_retracing=True)
def rotate_image(image, angle_rad, method='bilinear'):
    """
    Rotate *image* by *angle_rad* using a proper affine transform.

    Works for rank-3 (H, W, C) tensors.  Uses ``tf.raw_ops.ImageProjectiveTransformV3``
    when available, or falls back to building the transform manually.
    """
    squeeze = False
    if image.shape.rank == 3 and image.shape[-1] == 1:
        squeeze = True
        image = tf.squeeze(image, axis=-1)
        image = tf.expand_dims(image, axis=-1)  # keep channel dim

    image = tf.expand_dims(image, 0)  # add batch dim

    cos_a = tf.cos(angle_rad)
    sin_a = tf.sin(angle_rad)
    h = tf.cast(tf.shape(image)[1], tf.float32)
    w = tf.cast(tf.shape(image)[2], tf.float32)
    cx, cy = w / 2.0, h / 2.0

    # 3x3 affine: translate to origin → rotate → translate back
    # Flatten to the 8-element vector expected by TF (row-major, last element = 0)
    transform = tf.stack([
        cos_a, -sin_a, cx - cos_a * cx + sin_a * cy,
        sin_a,  cos_a, cy - sin_a * cx - cos_a * cy,
        0.0, 0.0,
    ])
    transform = tf.reshape(transform, [1, 8])

    fill_value = 0.0
    interp = 'BILINEAR' if method == 'bilinear' else 'NEAREST'
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.cast(image, tf.float32),
        transforms=transform,
        output_shape=tf.shape(image)[1:3],
        interpolation=interp,
        fill_mode='REFLECT',
        fill_value=fill_value,
    )
    rotated = rotated[0]

    if squeeze:
        rotated = rotated  # already has channel dim
    return rotated


# ---------------------------------------------------------------------------
# Core augmentation primitives
# ---------------------------------------------------------------------------
@tf.function(reduce_retracing=True)
def random_flip(image, label, p_horizontal=0.5, p_vertical=0.3):
    if tf.random.uniform(()) < p_horizontal:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    if tf.random.uniform(()) < p_vertical:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)
    return image, label


@tf.function(reduce_retracing=True)
def random_rotation(image, label, max_angle=15.0, p_90deg=0.2):
    max_angle_rad = max_angle * (math.pi / 180.0)
    if tf.random.uniform(()) < p_90deg:
        k = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        label = tf.image.rot90(label, k=k)
    else:
        angle = tf.random.uniform([], minval=-max_angle_rad, maxval=max_angle_rad)
        image = rotate_image(image, angle, method='bilinear')
        label = rotate_image(label, angle, method='nearest')
        label = tf.cast(label, tf.uint8)
    return image, label


@tf.function(reduce_retracing=True)
def add_speckle_noise(image, label, mean=0.0, stddev=0.15, p=0.7):
    """Multiplicative speckle noise — models SAR sensor noise."""
    if tf.random.uniform(()) < p:
        noise = tf.random.normal(tf.shape(image), mean=mean, stddev=stddev, dtype=tf.float32)
        noisy = tf.cast(image, tf.float32) * (1.0 + noise)
        return tf.clip_by_value(noisy, 0.0, 1.0), label
    return image, label


# ---------------------------------------------------------------------------
# NEW SAR-specific augmentations
# ---------------------------------------------------------------------------
@tf.function(reduce_retracing=True)
def random_brightness_contrast(image, label, p_bright=0.4, delta=0.2,
                                p_contrast=0.4, low=0.8, high=1.2):
    """Simulate SAR back-scatter variation with incidence angle."""
    img = tf.cast(image, tf.float32)
    if tf.random.uniform(()) < p_bright:
        img = tf.image.random_brightness(img, max_delta=delta)
    if tf.random.uniform(()) < p_contrast:
        img = tf.image.random_contrast(img, lower=low, upper=high)
    return tf.clip_by_value(img, 0.0, 1.0), label


@tf.function(reduce_retracing=True)
def random_gamma(image, label, p=0.3, low=0.7, high=1.5):
    """Gamma correction: SAR intensity follows non-linear distributions."""
    if tf.random.uniform(()) < p:
        gamma = tf.random.uniform((), low, high)
        img = tf.pow(tf.clip_by_value(tf.cast(image, tf.float32), 1e-7, 1.0), gamma)
        return tf.clip_by_value(img, 0.0, 1.0), label
    return image, label


@tf.function(reduce_retracing=True)
def random_gaussian_blur(image, label, p=0.2, sigma_low=0.5, sigma_high=1.5):
    """Approximate Gaussian blur to simulate varying sensor resolutions."""
    if tf.random.uniform(()) < p:
        sigma = tf.random.uniform((), sigma_low, sigma_high)
        # Build a 5×5 Gaussian kernel
        ksize = 5
        ax = tf.range(-ksize // 2 + 1, ksize // 2 + 1, dtype=tf.float32)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.reshape(kernel, [ksize, ksize, 1, 1])

        img = tf.cast(image, tf.float32)
        img = tf.expand_dims(img, 0)  # add batch dim
        # Handle multi-channel by depthwise conv
        n_ch = tf.shape(image)[-1]
        kernel_tiled = tf.tile(kernel, [1, 1, n_ch, 1])
        blurred = tf.nn.depthwise_conv2d(img, kernel_tiled, strides=[1, 1, 1, 1], padding='SAME')
        return tf.clip_by_value(blurred[0], 0.0, 1.0), label
    return image, label


@tf.function(reduce_retracing=True)
def random_cutout(image, label, p=0.3, size_low=0.05, size_high=0.15, max_patches=3):
    """Random erasing (cutout) for regularisation — forces model to use global context."""
    if tf.random.uniform(()) < p:
        img = tf.cast(image, tf.float32)
        h = tf.cast(tf.shape(img)[0], tf.float32)
        w = tf.cast(tf.shape(img)[1], tf.float32)
        n_patches = tf.random.uniform((), 1, max_patches + 1, dtype=tf.int32)

        for _ in tf.range(n_patches):
            frac = tf.random.uniform((), size_low, size_high)
            ch = tf.cast(h * frac, tf.int32)
            cw = tf.cast(w * frac, tf.int32)
            cy = tf.random.uniform((), 0, tf.shape(img)[0] - ch, dtype=tf.int32)
            cx = tf.random.uniform((), 0, tf.shape(img)[1] - cw, dtype=tf.int32)

            # Create mask: 1 everywhere except the cutout region
            mask = tf.ones_like(img)
            padding = tf.pad(
                tf.zeros([ch, cw, tf.shape(img)[2]]),
                [[cy, tf.shape(img)[0] - cy - ch], [cx, tf.shape(img)[1] - cx - cw], [0, 0]],
            )
            mask = mask - padding
            img = img * mask

        return tf.clip_by_value(img, 0.0, 1.0), label
    return image, label


# ---------------------------------------------------------------------------
# Composite augmentation
# ---------------------------------------------------------------------------
@tf.function(reduce_retracing=True)
def augment_single_sample(image, label):
    """Apply the full augmentation pipeline to a single (image, label) pair."""
    image = tf.cast(image, tf.float32)

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

    image = tf.clip_by_value(image, 0.0, 1.0)
    return tf.cast(image, tf.float16), tf.cast(label, tf.uint8)


def apply_augmentation(dataset, batch_size=2):
    """Unbatch → augment per-sample → re-batch."""
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(buffer_size=2000)  # shuffle after unbatch for diversity
    dataset = dataset.map(augment_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
