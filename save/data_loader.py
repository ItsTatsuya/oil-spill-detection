"""
Oil Spill Detection Dataset Loader

This module provides functionality to load and preprocess the oil spill detection dataset.
The dataset consists of:
- Images (JPG, 1250x650)
- Labels (RGB PNG)
- Labels_1D (PNG with values 0-4 representing different classes)

Classes:
0 - Sea Surface
1 - Oil Spill
2 - Look-alike
3 - Ship
4 - Land
"""

import os
import tensorflow as tf
import numpy as np
import glob
import logging

# Configure TensorFlow logging more aggressively
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
tf.get_logger().setLevel('ERROR')

# Disable other TensorFlow warnings and messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Disable the deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Suppress specific TensorFlow messages related to rendezvous
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def parse_image_label(image_path, label_path):
    """
    Load and preprocess an image and its corresponding label.

    Args:
        image_path: Path to the image file
        label_path: Path to the label file

    Returns:
        Tuple of (preprocessed image, label)
    """
    # Read the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Resize image to 320x320
    image = tf.image.resize(image, [320, 320])

    # Normalize image to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # Read the label mask
    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)

    # Resize label to 320x320 using nearest neighbor to preserve class values
    label = tf.image.resize(label, [320, 320], method='nearest')

    # Ensure label is uint8 with values 0-4
    label = tf.cast(label, tf.uint8)

    return image, label


def load_dataset(data_dir='dataset', split='train', batch_size=8):
    """
    Load the oil spill detection dataset.

    Args:
        data_dir: Root directory of the dataset
        split: Either 'train' or 'test'
        batch_size: Batch size for the returned tf.data.Dataset

    Returns:
        tf.data.Dataset containing (image, label) pairs
    """
    # Check if split is valid
    if split not in ['train', 'test']:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")

    # Define directories
    split_dir = os.path.join(data_dir, split)
    image_dir = os.path.join(split_dir, 'images')
    label_dir = os.path.join(split_dir, 'labels_1D')

    # Get sorted lists of image and label files
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.png')))

    # Ensure we have matching files
    if len(image_files) != len(label_files):
        raise ValueError(f"Number of images ({len(image_files)}) does not match number of labels ({len(label_files)})")

    # Create a tf.data.Dataset from the file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))

    # Map the parse_image_label function to preprocess each pair
    dataset = dataset.map(
        lambda img_path, lbl_path: parse_image_label(img_path, lbl_path),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Configure the dataset based on split
    if split == 'train':
        # For training: shuffle, batch, and prefetch
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        # For testing: batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Print dataset information
    print(f"Loaded {split} dataset with {len(image_files)} samples")

    return dataset


if __name__ == "__main__":
    # Set TensorFlow to only use CPU
    # This avoids GPU-related warnings if GPU support is not properly configured
    tf.config.set_visible_devices([], 'GPU')

    # Example usage
    train_ds = load_dataset(split='train')
    test_ds = load_dataset(split='test')

    # Print dataset information
    print("Dataset information:")
    print(f"Train dataset: {train_ds}")
    print(f"Test dataset: {test_ds}")

    # Show a sample batch
    for images, labels in train_ds.take(1):
        print(f"Sample batch shapes - Images: {images.shape}, Labels: {labels.shape}")
        print(f"Image value range: {tf.reduce_min(images).numpy()} to {tf.reduce_max(images).numpy()}")
        print(f"Label values: {np.unique(labels.numpy())}")

    # Process datasets safely without OUT_OF_RANGE errors
    print("\nProcessing all batches in training dataset:")

    # Convert to a list to avoid OUT_OF_RANGE errors (materializes the dataset)
    # This approach consumes more memory but avoids the errors
    train_batches = list(train_ds.as_numpy_iterator())
    print(f"Successfully processed all {len(train_batches)} batches in the training dataset.")

    print("\nProcessing all batches in testing dataset:")
    test_batches = list(test_ds.as_numpy_iterator())
    print(f"Successfully processed all {len(test_batches)} batches in the testing dataset.")
