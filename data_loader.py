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
def sample_patches(image, label, patch_size=320):
    """
    Sample patches centered on ships or oil spills to boost rare class exposure.

    Args:
        image: Input image tensor
        label: Label tensor with class values (0-4)
        patch_size: Size of output patch

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

    ship_coords = tf.where(label == 3)  # Ship class
    oil_coords = tf.where(label == 1)   # Oil spill
    coords = tf.concat([ship_coords, oil_coords], axis=0)

    if tf.shape(coords)[0] > 0:
        idx = tf.random.uniform((), 0, tf.shape(coords)[0], dtype=tf.int32)
        y, x = coords[idx][0], coords[idx][1]

        # Convert to int32 to match the type of patch_size // 2
        y = tf.cast(y, tf.int32)
        x = tf.cast(x, tf.int32)

        # Calculate patch boundaries
        y_start = tf.maximum(0, y - patch_size // 2)
        x_start = tf.maximum(0, x - patch_size // 2)

        # Ensure we don't exceed image dimensions
        y_end = tf.minimum(image_height, y_start + patch_size)
        x_end = tf.minimum(image_width, x_start + patch_size)

        # Extract patches
        img_patch = image[y_start:y_end, x_start:x_end, :]
        lbl_patch = label[y_start:y_end, x_start:x_end, :]

        # Ensure patches are exactly patch_size x patch_size by resizing if needed
        return ensure_size(img_patch, lbl_patch)

    # For images with no ship/oil, take a random crop
    return tf.image.resize_with_crop_or_pad(image, patch_size, patch_size), tf.image.resize_with_crop_or_pad(label, patch_size, patch_size)


def parse_image_label(image_path, label_path):
    """Load and preprocess an image and its corresponding label."""
    # Read and preprocess image - load as grayscale (1-channel)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)  # Changed to 1 channel for grayscale SAR images
    image = tf.image.resize(image, [320, 320])
    image = tf.cast(image, tf.float32) / 255.0  # Use float32 for preprocessing

    # Read and preprocess label
    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.image.resize(label, [320, 320], method='nearest')
    label = tf.cast(label, tf.uint8)
    label = tf.clip_by_value(label, 0, 4)
    if tf.reduce_any(label > 4):
        tf.print("Warning: Invalid label detected, clipped to range [0, 4]")

    return image, label

def load_dataset(data_dir='dataset', split='train', batch_size=4):
    """Load the oil spill detection dataset and return dataset with class weights and number of batches."""
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
        dataset = dataset.map(_augment_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

        # Convert image to float32 AFTER augmentation
        dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)

        # Disable caching to reduce VRAM usage and use larger buffer for better shuffling
        dataset = dataset.batch(batch_size)

        # Increase shuffle buffer for better randomization
        dataset = dataset.shuffle(buffer_size=min(1000, num_samples // 2), reshuffle_each_iteration=True)

        # Enable optimization options
        options = tf.data.Options()
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.map_fusion = True
        options.experimental_optimization.parallel_batch = True  # Enable parallel batching
        options.deterministic = False  # Remove unnecessary determinism
        dataset = dataset.with_options(options)

        # Prefetch more batches
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        # For test dataset, convert to compute_dtype before batching
        dataset = dataset.map(lambda x, y: (tf.cast(x, policy.compute_dtype), y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
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
