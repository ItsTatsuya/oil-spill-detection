"""
Oil Spill Detection Dataset Augmentation

This module provides functions for data augmentation of oil spill detection images and labels.
It implements the following augmentations:
1. Random rotation (90°, 180°, 270°)
2. Adding Gaussian noise
3. Random brightness and contrast
4. Random flipping

All augmentations are applied with 50% probability.
"""

import tensorflow as tf
import numpy as np
import os

# Configure TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
tf.get_logger().setLevel('ERROR')


def simple_augment(image, label):
    """
    Apply simple data augmentation to images and labels.

    Args:
        image: Input image tensor
        label: Input label tensor

    Returns:
        Tuple of (augmented_image, augmented_label)
    """
    # Get the original shape
    original_shape = tf.shape(image)

    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)

    # Random vertical flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)

    # Random rotation (90°, 180°, 270°) with 50% probability
    if tf.random.uniform(()) > 0.5:
        k = tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        label = tf.image.rot90(label, k=k)

    # Random brightness
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, 0.1)

    # Random contrast
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, 0.9, 1.1)

    # Ensure the image values are clipped between 0 and 1
    image = tf.clip_by_value(image, 0.0, 1.0)

    # Ensure label type remains the same
    label = tf.cast(label, tf.uint8)

    return image, label


def apply_augmentation(dataset, batch_size=8):
    """
    Apply augmentation to a tf.data.Dataset.

    Args:
        dataset: A tf.data.Dataset containing (image, label) pairs
        batch_size: Batch size for the returned dataset

    Returns:
        An augmented tf.data.Dataset
    """
    # First, ensure we have an unbatched dataset
    if hasattr(dataset, 'unbatch'):
        dataset = dataset.unbatch()

    # Apply the simple augmentation to each element
    augmented_dataset = dataset.map(
        simple_augment,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch the dataset and enable prefetching for better performance
    augmented_dataset = augmented_dataset.batch(batch_size)
    augmented_dataset = augmented_dataset.repeat()  # Repeat the dataset for multiple epochs
    augmented_dataset = augmented_dataset.prefetch(tf.data.AUTOTUNE)

    return augmented_dataset


if __name__ == "__main__":
    # Import the data_loader module to test the augmentation
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

    from data_loader import load_dataset
    import matplotlib.pyplot as plt

    # Configure TensorFlow to use CPU only
    tf.config.set_visible_devices([], 'GPU')

    # Load a small dataset for testing
    test_ds = load_dataset(split='test', batch_size=1)

    # Get a single sample for demonstration
    for image, label in test_ds.take(1):
        original_image = image[0].numpy()
        original_label = label[0].numpy()

        # Apply augmentation multiple times to demonstrate
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.imshow(original_image)

        plt.subplot(2, 3, 4)
        plt.title("Original Label")
        plt.imshow(original_label[:,:,0], cmap='jet', vmin=0, vmax=4)

        # Apply various augmentations
        for i in range(2):
            aug_image, aug_label = simple_augment(original_image, original_label)

            plt.subplot(2, 3, i+2)
            plt.title(f"Augmented Image {i+1}")
            plt.imshow(aug_image.numpy())

            plt.subplot(2, 3, i+5)
            plt.title(f"Augmented Label {i+1}")
            plt.imshow(aug_label.numpy()[:,:,0], cmap='jet', vmin=0, vmax=4)

        plt.tight_layout()
        plt.savefig('augmentation_examples.png')
        plt.close()

        print("Augmentation examples saved as 'augmentation_examples.png'")
        break

    # Test the dataset augmentation pipeline
    print("\nTesting the augmentation pipeline...")
    augmented_ds = apply_augmentation(test_ds, batch_size=4)

    print("Original dataset:", test_ds)
    print("Augmented dataset:", augmented_ds)

    try:
        # Check a batch from the augmented dataset
        for aug_images, aug_labels in augmented_ds.take(1):
            print(f"Augmented batch - Images shape: {aug_images.shape}, Labels shape: {aug_labels.shape}")
            print(f"Image value range: {tf.reduce_min(aug_images).numpy()} to {tf.reduce_max(aug_images).numpy()}")
            print(f"Label values: {np.unique(aug_labels.numpy())}")

            # Visualize the first image in the augmented batch
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Augmented Batch Image")
            plt.imshow(aug_images[0].numpy())

            plt.subplot(1, 2, 2)
            plt.title("Augmented Batch Label")
            plt.imshow(aug_labels[0].numpy()[:,:,0], cmap='jet', vmin=0, vmax=4)

            plt.savefig('augmented_batch_example.png')
            plt.close()
            print("Augmented batch example saved as 'augmented_batch_example.png'")

        print("Augmentation pipeline test completed successfully!")
    except Exception as e:
        print(f"Error in augmentation pipeline: {e}")
