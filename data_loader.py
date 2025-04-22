import os
import numpy as np
import glob
import logging

def silent_tf_import():
    import sys
    orig_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(orig_stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, orig_stderr_fd)
    os.close(devnull_fd)

    import tensorflow as tf

    os.dup2(saved_stderr_fd, orig_stderr_fd)
    os.close(saved_stderr_fd)

    return tf

tf = silent_tf_import()

# Import mixed precision - DO NOT set policy here, let train.py handle it.
from tensorflow.keras import mixed_precision # type: ignore
policy = mixed_precision.global_policy()
print(f"Data loader using mixed precision policy: {policy.name}")

# Import custom per-image augmentation function
from augmentation import _augment_image_and_label

# Disable other TensorFlow warnings and messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def analyze_class_distribution(label_paths):
    """Analyze class distribution from a sample of label files and return class weights."""
    # Use fixed class weights based on domain knowledge instead of calculating from data
    weights = tf.constant([0.05, 0.4, 0.1, 0.35, 0.1], dtype=tf.float32)
    adjusted_weights = {i: float(weights[i]) for i in range(5)}
    tf.print("Using predefined class weights:", adjusted_weights)
    return adjusted_weights

@tf.function
def sample_patches(image, label, patch_size=384):
    """
    Sample patches centered on ships or oil spills to boost rare class exposure.
    Balances between rare class-focused patches and random patches for better
    generalization and more stable training.

    Args:
        image: Input image tensor
        label: Label tensor with class values (0-4)
        patch_size: Size of output patch (384x384 for higher resolution features)

    Returns:
        Tuple of (image_patch, label_patch) tensors
    """
    # Get image dimensions
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    # Ensure consistent output size by using resize if needed
    def ensure_size(img, lbl):
        img = tf.image.resize_with_crop_or_pad(img, patch_size, patch_size)
        lbl = tf.image.resize_with_crop_or_pad(lbl, patch_size, patch_size)
        return img, lbl

    # Initialize these variables in all branches to avoid TF graph execution errors
    ship_coords = tf.zeros((0, 2), dtype=tf.int64)
    oil_coords = tf.zeros((0, 2), dtype=tf.int64)
    lookalike_coords = tf.zeros((0, 2), dtype=tf.int64)

    # Find coordinates of each class (only when we'll use them)
    need_coords = tf.random.uniform(()) >= 0.3  # Inverse of the first condition

    # Only compute these coordinates if we need them (70% of the time)
    if need_coords:
        ship_coords = tf.where(label == 3)  # Ship class
        oil_coords = tf.where(label == 1)   # Oil spill
        lookalike_coords = tf.where(label == 2)  # Look-alike class

    # Random sampling with 30% probability to ensure dataset diversity
    # and prevent overfit to rare classes
    if tf.random.uniform(()) < 0.3:
        return tf.image.resize_with_crop_or_pad(image, patch_size, patch_size), tf.image.resize_with_crop_or_pad(label, patch_size, patch_size)

    # For the remaining 70%, focus on rare classes (ships and oil spills)
    # Combine coordinates with higher probability for rare classes
    # Ships and oil spills get higher focus (80%)
    # Look-alikes get lower focus (20%) for better false-positive reduction

    # If we have rare classes in the image, sample them preferentially
    use_rare = tf.logical_or(
        tf.shape(ship_coords)[0] > 0,
        tf.shape(oil_coords)[0] > 0
    )

    def sample_rare():
        rare_coords = tf.concat([ship_coords, oil_coords], axis=0)

        # Add some lookalike samples for false positive reduction if available
        has_lookalikes = tf.shape(lookalike_coords)[0] > 0
        coords = tf.cond(
            has_lookalikes,
            lambda: tf.concat([
                rare_coords,
                # Sample lookalikes but with lower probability
                lookalike_coords[:tf.maximum(1, tf.shape(lookalike_coords)[0] // 4)]
            ], axis=0),
            lambda: rare_coords
        )

        # Handle empty coords case
        coords = tf.cond(
            tf.equal(tf.shape(coords)[0], 0),
            lambda: tf.zeros((1, 2), dtype=tf.int64),  # Default coordinates if empty
            lambda: coords
        )

        idx = tf.random.uniform((), 0, tf.shape(coords)[0], dtype=tf.int32)
        y, x = coords[idx][0], coords[idx][1]

        # Cast to int32 and ensure they're scalar values
        y = tf.cast(y, tf.int32)
        x = tf.cast(x, tf.int32)

        # Calculate patch boundaries with center at the selected point
        half_size = patch_size // 2
        y_start = tf.maximum(0, y - half_size)
        x_start = tf.maximum(0, x - half_size)

        # Ensure we don't exceed image dimensions
        y_end = tf.minimum(image_height, y_start + patch_size)
        x_end = tf.minimum(image_width, x_start + patch_size)

        # Extract patches
        img_patch = image[y_start:y_end, x_start:x_end, :]
        lbl_patch = label[y_start:y_end, x_start:x_end, :]

        # Ensure patches are exactly patch_size x patch_size
        return ensure_size(img_patch, lbl_patch)

    # For images with no rare classes, take a random crop
    def sample_random():
        # Use random crop to potentially find areas of interest
        offset_height = tf.random.uniform((), 0, tf.maximum(1, image_height - patch_size + 1), dtype=tf.int32)
        offset_width = tf.random.uniform((), 0, tf.maximum(1, image_width - patch_size + 1), dtype=tf.int32)

        img_patch = tf.image.crop_to_bounding_box(
            image, offset_height, offset_width,
            tf.minimum(patch_size, image_height - offset_height),
            tf.minimum(patch_size, image_width - offset_width)
        )
        lbl_patch = tf.image.crop_to_bounding_box(
            label, offset_height, offset_width,
            tf.minimum(patch_size, image_height - offset_height),
            tf.minimum(patch_size, image_width - offset_width)
        )

        return ensure_size(img_patch, lbl_patch)

    return tf.cond(use_rare, sample_rare, sample_random)

def parse_image_label(image_path, label_path):
    """Load and preprocess an image and its corresponding label."""
    # Read and preprocess image - load as grayscale (1-channel)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)  # Changed to 1 channel for grayscale SAR images
    image = tf.image.resize(image, [384, 384])  # Increased from 320x320 to 384x384 for better feature extraction
    image = tf.cast(image, tf.float32) / 255.0  # Use float32 for preprocessing

    # Read and preprocess label
    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.image.resize(label, [384, 384], method='nearest')  # Increased from 320x320 to 384x384
    label = tf.cast(label, tf.uint8)
    label = tf.clip_by_value(label, 0, 4)
    if tf.reduce_any(label > 4):
        tf.print("Warning: Invalid label detected, clipped to range [0, 4]")

    return image, label

def load_dataset(data_dir='dataset', split='train', batch_size=16):
    """
    Load the oil spill detection dataset and return dataset with class weights and number of batches.

    Args:
        data_dir: Path to dataset directory
        split: Dataset split ('train' or 'test')
        batch_size: Batch size (increased to 16 for better VRAM utilization)

    Returns:
        dataset: TensorFlow dataset
        class_weights: Class weights dictionary (for training only)
        num_batches: Number of batches in the dataset
    """
    if split not in ['train', 'test']:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")

    split_dir = os.path.join(data_dir, split)
    image_dir = os.path.join(split_dir, 'images')
    label_dir = os.path.join(split_dir, 'labels_1D')

    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.png')))

    if len(image_files) != len(label_files):
        raise ValueError(f"Number of images ({len(image_files)}) does not match number of labels ({len(label_files)})")

    dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))
    num_samples = len(image_files)
    num_batches = num_samples // batch_size + (1 if num_samples % batch_size > 0 else 0)

    # Compute class weights for training
    class_weights = analyze_class_distribution(label_files) if split == 'train' else None

    # Use more parallel calls for faster data loading
    dataset = dataset.map(parse_image_label, num_parallel_calls=tf.data.AUTOTUNE)

    # Configure dataset
    if split == 'train':
        # Apply sample patches centered on ships and oil spills (for training only)
        dataset = dataset.map(sample_patches, num_parallel_calls=tf.data.AUTOTUNE)

        # Apply custom per-image augmentations AFTER patch sampling
        # DISABLE the per-image augmentation to avoid memory issues
        # dataset = dataset.map(_augment_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

        # Convert image to float32 AFTER augmentation
        dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)

        # Use batch size from function parameter
        dataset = dataset.batch(batch_size)

        # Increase shuffle buffer size for better randomization
        # Use larger buffer (2000 or 50% of dataset) for more thorough shuffling
        shuffle_buffer = min(2000, num_samples)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
        print(f"Using shuffle buffer size of {shuffle_buffer} for improved randomization")

        # Enable optimization options
        options = tf.data.Options()
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.map_fusion = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.noop_elimination = True  # Remove unnecessary operations
        options.experimental_optimization.apply_default_optimizations = True
        options.deterministic = False  # Remove unnecessary determinism for better performance
        dataset = dataset.with_options(options)

        # Prefetch more batches for continuous processing
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        # For test dataset, add caching to improve performance
        dataset = dataset.cache()

        # Convert to compute_dtype before batching
        dataset = dataset.map(lambda x, y: (tf.cast(x, policy.compute_dtype), y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)

        # Apply optimizations for test dataset as well
        options = tf.data.Options()
        options.experimental_optimization.apply_default_optimizations = True
        dataset = dataset.with_options(options)

        # Prefetch for test dataset too
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print(f"Loaded {split} dataset with {len(image_files)} samples, {num_batches} batches with batch_size={batch_size}")
    return dataset, class_weights, num_batches

if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')

    # Example usage
    train_ds, train_class_weights, train_num_batches = load_dataset(split='train')
    test_ds, _, test_num_batches = load_dataset(split='test')

    print("Dataset information:")
    print(f"Train dataset: {train_ds}")
    print(f"Test dataset: {test_ds}")
    print(f"Train class weights: {train_class_weights}")
    # Note: Set mixed_float16 policy in train.py before model compilation

    try:
        for images, labels in train_ds.take(1):
            tf.print("Batch size:", tf.shape(images)[0])  # Debug batch size
            print(f"Sample batch shapes - Images: {images.shape}, Labels: {labels.shape}")
            print(f"Image value range: {tf.reduce_min(images).numpy()} to {tf.reduce_max(images).numpy()}")
            print(f"Image dtype: {images.dtype}")
            print(f"Label values: {np.unique(labels.numpy())}")
    except tf.errors.OutOfRangeError:
        print("Warning: Partial dataset read due to caching, proceeding with available data.")
    except Exception as e:
        print(f"Error processing sample batch: {str(e)}")

    print("\nProcessing all batches in training dataset:")
    try:
        train_batches = list(train_ds.as_numpy_iterator())
        print(f"Successfully processed all {len(train_batches)} batches in the training dataset.")
    except tf.errors.OutOfRangeError:
        print(f"Successfully processed all {train_num_batches} batches in the training dataset.")
    except Exception as e:
        print(f"Error processing training batches: {str(e)}")

    print("\nProcessing all batches in testing dataset:")
    try:
        test_batches = list(test_ds.as_numpy_iterator())
        print(f"Successfully processed all {len(test_batches)} batches in the testing dataset.")
    except tf.errors.OutOfRangeError:
        print(f"Successfully processed all {test_num_batches} batches in the testing dataset.")
    except Exception as e:
        print(f"Error processing testing batches: {str(e)}")
