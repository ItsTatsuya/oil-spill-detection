"""
Data loading pipeline for oil spill segmentation.

Key improvements over the original:
- Proper train/val/test split (val carved from training data, test reserved for final eval)
- Class weights computed from actual pixel distribution (inverse-frequency), cached to disk
- Patch sampling from original-resolution images (no redundant resize-then-crop)
- No .cache() on training set so stochastic augmentation produces fresh samples every epoch
- Removed fixed shuffle seed so reshuffling is different each epoch
"""

import os
import logging
import glob

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision  # type: ignore

logger = logging.getLogger('oil_spill')

policy = mixed_precision.global_policy()

logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

IMG_SIZE = 384
NUM_CLASSES = 5


# ---------------------------------------------------------------------------
# Class weight computation (inverse-frequency from actual labels)
# ---------------------------------------------------------------------------
def analyze_class_distribution(label_paths, num_classes=NUM_CLASSES, cache_path=None):
    """
    Compute inverse-frequency class weights from actual label statistics.

    If *cache_path* is given and already exists, weights are loaded from disk
    instead of recomputing. Otherwise they are computed and saved.
    """
    if cache_path and os.path.exists(cache_path):
        weights = np.load(cache_path)
        logger.info("Loaded cached class weights from %s", cache_path)
        return {i: float(weights[i]) for i in range(num_classes)}

    logger.info("Computing class weights from %d label files …", len(label_paths))
    counts = np.zeros(num_classes, dtype=np.int64)
    for path in label_paths:
        label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            continue
        label = np.clip(label, 0, num_classes - 1)
        for c in range(num_classes):
            counts[c] += np.sum(label == c)

    freq = counts / (counts.sum() + 1e-12)
    raw_weights = 1.0 / (freq + 1e-6)
    raw_weights /= raw_weights.sum()  # normalise so they sum to 1

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.save(cache_path, raw_weights)
        logger.info("Saved class weights to %s", cache_path)

    print(f"Computed class weights: {dict(enumerate(np.round(raw_weights, 4)))}")
    return {i: float(raw_weights[i]) for i in range(num_classes)}


# ---------------------------------------------------------------------------
# Patch sampling from original resolution
# ---------------------------------------------------------------------------
@tf.function
def sample_patches(image, label, patch_size=IMG_SIZE):
    """
    Random crop of *patch_size* × *patch_size* from the (possibly larger) image.

    With 50 % probability the crop is centred on an Oil-Spill (class 1) or
    Ship (class 3) pixel, which up-samples rare classes without duplication.
    If the image is smaller than the patch, it is padded.
    """
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    # Pad if needed
    pad_h = tf.maximum(patch_size - h, 0)
    pad_w = tf.maximum(patch_size - w, 0)
    if pad_h > 0 or pad_w > 0:
        image = tf.pad(image, [[0, pad_h], [0, pad_w], [0, 0]])
        label = tf.pad(label, [[0, pad_h], [0, pad_w], [0, 0]])
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]

    def _class_centred_crop():
        label_sq = tf.cast(tf.squeeze(label, axis=-1), tf.int32)
        has_ship = tf.reduce_any(tf.equal(label_sq, 3))
        has_oil = tf.reduce_any(tf.equal(label_sq, 1))

        target_class = tf.cond(has_ship, lambda: 3, lambda: 1)
        has_target = tf.logical_or(has_ship, has_oil)

        indices = tf.where(tf.equal(label_sq, target_class))
        n_idx = tf.shape(indices)[0]
        valid = tf.logical_and(has_target, n_idx > 0)

        def _crop_around():
            rid = tf.random.uniform((), 0, n_idx, dtype=tf.int32)
            cy = tf.cast(indices[rid, 0], tf.int32)
            cx = tf.cast(indices[rid, 1], tf.int32)
            half = patch_size // 2
            top = tf.clip_by_value(cy - half, 0, h - patch_size)
            left = tf.clip_by_value(cx - half, 0, w - patch_size)
            return (
                tf.image.crop_to_bounding_box(image, top, left, patch_size, patch_size),
                tf.image.crop_to_bounding_box(label, top, left, patch_size, patch_size),
            )

        def _random_crop():
            top = tf.random.uniform((), 0, h - patch_size + 1, dtype=tf.int32)
            left = tf.random.uniform((), 0, w - patch_size + 1, dtype=tf.int32)
            return (
                tf.image.crop_to_bounding_box(image, top, left, patch_size, patch_size),
                tf.image.crop_to_bounding_box(label, top, left, patch_size, patch_size),
            )

        return tf.cond(valid, _crop_around, _random_crop)

    def _random_crop():
        top = tf.random.uniform((), 0, h - patch_size + 1, dtype=tf.int32)
        left = tf.random.uniform((), 0, w - patch_size + 1, dtype=tf.int32)
        return (
            tf.image.crop_to_bounding_box(image, top, left, patch_size, patch_size),
            tf.image.crop_to_bounding_box(label, top, left, patch_size, patch_size),
        )

    return tf.cond(tf.random.uniform(()) < 0.5, _class_centred_crop, _random_crop)


# ---------------------------------------------------------------------------
# Image + label parsing  (no resize — keep original resolution for patch crop)
# ---------------------------------------------------------------------------
def parse_image_label_original(image_path, label_path):
    """Load image and label at ORIGINAL resolution (for random-crop training)."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.cast(label, tf.uint8)
    label = tf.clip_by_value(label, 0, NUM_CLASSES - 1)
    return image, label


def parse_image_label_resized(image_path, label_path):
    """Load image and label resized to IMG_SIZE (for val / test)."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.image.resize(label, [IMG_SIZE, IMG_SIZE], method='nearest')
    label = tf.cast(label, tf.uint8)
    label = tf.clip_by_value(label, 0, NUM_CLASSES - 1)
    return image, label


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
def load_dataset(
    data_dir='dataset',
    split='train',
    batch_size=16,
    val_split=0.15,
    class_weights_cache='dataset/class_weights.npy',
):
    """
    Build a tf.data pipeline.

    Splits
    ------
    * ``train``  – training portion (1 - val_split) of dataset/train, with
      random crop & shuffling.  No .cache() so augmentation is fresh each epoch.
    * ``val``    – held-out portion (val_split) of dataset/train.  Deterministic.
    * ``test``   – dataset/test.  Deterministic.  Reserved for final evaluation.
    """
    if split not in ('train', 'val', 'test'):
        raise ValueError(f"Invalid split: {split}. Choose from 'train', 'val', 'test'.")

    # ---- resolve file lists ---------------------------------------------------
    if split in ('train', 'val'):
        base_dir = os.path.join(data_dir, 'train')
    else:
        base_dir = os.path.join(data_dir, 'test')

    image_dir = os.path.join(base_dir, 'images')
    label_dir = os.path.join(base_dir, 'labels_1D')

    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.png')))

    if len(image_files) != len(label_files):
        raise ValueError(
            f"Image-label mismatch in {base_dir}: "
            f"{len(image_files)} images vs {len(label_files)} labels"
        )

    # ---- train / val split (file-level, deterministic) -----------------------
    if split in ('train', 'val'):
        n_total = len(image_files)
        n_val = max(1, int(val_split * n_total))
        if split == 'val':
            image_files = image_files[-n_val:]
            label_files = label_files[-n_val:]
        else:
            image_files = image_files[:-n_val]
            label_files = label_files[:-n_val]

    num_samples = len(image_files)
    num_batches = num_samples // batch_size + (1 if num_samples % batch_size > 0 else 0)

    # ---- class weights (only needed for training) ----------------------------
    class_weights = None
    if split == 'train':
        class_weights = analyze_class_distribution(
            label_files, num_classes=NUM_CLASSES, cache_path=class_weights_cache
        )

    # ---- build pipeline ------------------------------------------------------
    dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))

    opts = tf.data.Options()
    opts.experimental_optimization.apply_default_optimizations = True

    if split == 'train':
        # Parse at original resolution → random-crop → shuffle → batch
        dataset = dataset.map(parse_image_label_original, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda img, lbl: sample_patches(img, lbl, IMG_SIZE),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float32), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.shuffle(buffer_size=min(2000, num_samples))
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # NO .cache() — augmentation must be stochastic each epoch
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.map_fusion = True
        opts.experimental_optimization.parallel_batch = True
        opts.deterministic = False
        dataset = dataset.with_options(opts)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        # Val / test: resize to fixed size, cache for speed
        dataset = dataset.map(parse_image_label_resized, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, policy.compute_dtype), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.with_options(opts)
        dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

    print(
        f"Loaded {split} dataset: {num_samples} samples, "
        f"{num_batches} batches (batch_size={batch_size})"
    )
    return dataset, class_weights, num_batches
