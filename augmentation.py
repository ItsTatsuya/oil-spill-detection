"""
Data augmentation for the oil spill detection dataset.
Augmentations are performed on the GPU when possible and resource optimized.
Incorporates Keras-CV for advanced batched augmentations like CutMix and MixUp.
"""

import tensorflow as tf
import numpy as np
import os

# Import keras_cv for advanced augmentations
try:
    import keras_cv
    KERAS_CV_AVAILABLE = True
    print("Keras-CV available for advanced augmentations")
except ImportError:
    KERAS_CV_AVAILABLE = False
    print("Keras-CV not available, falling back to basic augmentations")

# Configure TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
tf.get_logger().setLevel('ERROR')


@tf.function
def rotate_image(image, angle, interpolation='nearest'):
    """
    Rotate image using native TensorFlow operations (no TF Addons required).

    Args:
        image: A tensor of shape [height, width, channels]
        angle: Rotation angle in radians
        interpolation: Interpolation method ('nearest' or 'bilinear')

    Returns:
        Rotated image tensor
    """
    # Get image shape
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    # Calculate rotation center
    # Convert to float32 to avoid type mismatches
    height_float = tf.cast(height, tf.float32)
    width_float = tf.cast(width, tf.float32)
    center_x = width_float / 2.0
    center_y = height_float / 2.0

    # Calculate the rotation using TF's affine transformation
    # Create rotation matrix components
    costheta = tf.cos(angle)
    sintheta = tf.sin(angle)

    # Build the 8-element transformation matrix required by TF
    # Format: [a0, a1, a2, b0, b1, b2, c0, c1]
    # where the transform is [a0 a1 a2; b0 b1 b2; c0 c1 1]
    # See: https://www.tensorflow.org/api_docs/python/tf/raw_ops/ImageProjectiveTransformV3

    # Rotation around center:
    # [cosθ, -sinθ, center_x - center_x*cosθ + center_y*sinθ]
    # [sinθ,  cosθ, center_y - center_x*sinθ - center_y*cosθ]
    # [0,     0,    1]

    a0 = costheta
    a1 = -sintheta
    a2 = center_x - center_x * costheta + center_y * sintheta
    b0 = sintheta
    b1 = costheta
    b2 = center_y - center_x * sintheta - center_y * costheta

    # Create the transformation matrix with shape [1, 8]
    transforms = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0], axis=0)
    transforms = tf.reshape(transforms, [1, 8])

    # Apply the transformation
    rotated_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=transforms,
        output_shape=[height, width],
        interpolation=interpolation.upper(),
        fill_mode="CONSTANT",
        fill_value=0.0
    )

    return tf.squeeze(rotated_image, 0)


@tf.function
def _augment_image_and_label(image, label):
    """
    Apply augmentation to an image and its corresponding label.
    Using tf.function for GPU acceleration.

    Args:
        image: A tensor of shape [height, width, 3]
        label: A tensor of shape [height, width, 1]

    Returns:
        Tuple of (augmented_image, augmented_label)
    """
    # Get original dimensions
    original_shape = tf.shape(image)
    h, w = original_shape[0], original_shape[1]

    # Combine image and label for consistent spatial transformations
    # This ensures augmentations are applied identically to both
    combined = tf.concat([image, tf.cast(label, tf.float32) / 4.0], axis=2)

    # Use GPU if available
    # Random flip left-right (50% chance)
    if tf.random.uniform(()) > 0.5:
        combined = tf.image.flip_left_right(combined)

    # Random rotation (max 10 degrees)
    if tf.random.uniform(()) > 0.5:  # 50% chance to apply rotation
        angle = tf.random.uniform([], minval=-0.174, maxval=0.174)  # +/- 10 degrees in radians
        combined = rotate_image(combined, angle, interpolation='nearest')

    # Extract image and label back after spatial transformations
    image = combined[..., :3]
    label = combined[..., 3:]

    # Random brightness (image only)
    image = tf.image.random_brightness(image, max_delta=0.1)

    # Random contrast (image only)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Random saturation (image only)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)

    # Add random noise with 30% probability
    if tf.random.uniform(()) > 0.7:
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.01)
        image = image + noise

    # Ensure image values remain in [0, 1]
    image = tf.clip_by_value(image, 0, 1)

    # Convert label back to original range and type
    label = label * 4.0
    label = tf.cast(tf.round(label), tf.uint8)

    return image, label


@tf.function
def apply_cutmix(images, labels, alpha=0.2, prob=0.25):
    """
    Apply CutMix augmentation to a batch of images and labels.

    Args:
        images: Tensor of shape [batch_size, height, width, channels]
        labels: Tensor of shape [batch_size, height, width, 1]
        alpha: Alpha parameter for beta distribution
        prob: Probability of applying CutMix

    Returns:
        Tuple of augmented (images, labels)
    """
    # Only apply CutMix with specified probability
    if tf.random.uniform(()) > prob:
        return images, labels

    if not KERAS_CV_AVAILABLE:
        return images, labels

    # For segmentation, we need to process each sample independently
    batch_size = tf.shape(images)[0]
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]

    # Convert to float32 to avoid type mismatches during multiplication
    height_float = tf.cast(height, tf.float32)
    width_float = tf.cast(width, tf.float32)

    # Generate random indices for mixing (within the batch)
    rand_indices = tf.random.shuffle(tf.range(batch_size))

    # Generate random coordinates and size for the cut
    lambda_param = tf.random.uniform([], 0, 1)
    cut_ratio = tf.math.sqrt(1.0 - lambda_param)

    # Calculate cut size using float versions of height and width, then cast to int
    cut_h = tf.cast(height_float * cut_ratio, tf.int32)
    cut_w = tf.cast(width_float * cut_ratio, tf.int32)

    # Random center position
    center_x = tf.random.uniform([], 0, width, dtype=tf.int32)
    center_y = tf.random.uniform([], 0, height, dtype=tf.int32)

    # Calculate box coordinates
    x1 = tf.maximum(0, center_x - cut_w // 2)
    y1 = tf.maximum(0, center_y - cut_h // 2)
    x2 = tf.minimum(width, center_x + cut_w // 2)
    y2 = tf.minimum(height, center_y + cut_h // 2)

    # Create the cutmix mask using a simpler approach
    # Start with a mask of ones (all original image)
    mask = tf.ones([height, width, 1], dtype=tf.float32)

    # Create a mask of zeros for the cut region
    cut_mask = tf.zeros([y2-y1, x2-x1, 1], dtype=tf.float32)

    # Create paddings for the cut region to position it correctly
    paddings = [
        [y1, height - y2],  # Padding for height (before, after)
        [x1, width - x2],   # Padding for width (before, after)
        [0, 0]              # No padding for channels
    ]

    # Pad the cut mask to create a full-sized mask with zeros in the cut region
    cut_mask_padded = tf.pad(cut_mask, paddings)

    # Invert the padded mask (0 becomes 1, 1 becomes 0)
    mask = mask * (1.0 - cut_mask_padded)

    # Reshape to add batch dimension
    mask = tf.reshape(mask, [1, height, width, 1])

    # Apply the cutmix to images and labels
    mixed_images = images * mask + tf.gather(images, rand_indices) * (1.0 - mask)
    mixed_labels = labels * tf.cast(mask, labels.dtype) + tf.gather(labels, rand_indices) * tf.cast(1.0 - mask, labels.dtype)

    return mixed_images, mixed_labels


@tf.function
def apply_mixup(images, labels, alpha=0.2, prob=0.25):
    """
    Apply MixUp augmentation to a batch of images and labels.

    Args:
        images: Tensor of shape [batch_size, height, width, channels]
        labels: Tensor of shape [batch_size, height, width, 1]
        alpha: Alpha parameter for beta distribution
        prob: Probability of applying MixUp

    Returns:
        Tuple of augmented (images, labels)
    """
    # Only apply MixUp with specified probability
    if tf.random.uniform(()) > prob:
        return images, labels

    if not KERAS_CV_AVAILABLE:
        return images, labels

    # Generate mixing coefficient from beta distribution
    gamma = tf.random.gamma([1], alpha, 1)[0]
    lam = tf.minimum(gamma / (gamma + tf.random.gamma([1], alpha, 1)[0]), tf.ones([1])[0])

    # Generate random indices for mixing
    batch_size = tf.shape(images)[0]
    rand_indices = tf.random.shuffle(tf.range(batch_size))

    # Apply mixup
    mixed_images = lam * images + (1 - lam) * tf.gather(images, rand_indices)

    # For semantic segmentation, we need to handle labels differently than classification
    # We'll use a one-hot encoding approach to properly mix the labels
    mixed_labels = lam * tf.cast(labels, tf.float32) + (1 - lam) * tf.cast(tf.gather(labels, rand_indices), tf.float32)
    mixed_labels = tf.cast(tf.math.round(mixed_labels), tf.uint8)  # Round to nearest label

    return mixed_images, mixed_labels


def apply_augmentation(dataset, batch_size=8):
    """
    Apply augmentation to a dataset of images and labels.

    Args:
        dataset: tf.data.Dataset containing (image, label) pairs
        batch_size: Batch size for the returned dataset

    Returns:
        tf.data.Dataset with augmented (image, label) pairs
    """
    # Extract any batching that might already be applied
    was_batched = False
    try:
        # Check if the dataset is already batched
        shapes = tf.compat.v1.data.get_output_shapes(dataset)
        if len(shapes[0].as_list()) == 4:  # Batched images have shape [batch, H, W, C]
            was_batched = True
            # Unbatch if needed
            dataset = dataset.unbatch()
            print("Unbatched existing dataset before applying augmentation")
    except Exception:
        pass  # Assume dataset is not batched

    # Apply per-image augmentation using map with GPU acceleration
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True

    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Using GPU for data augmentation")
        parallel_calls = min(4, len(gpus))  # Use fewer parallel calls to avoid memory issues
    else:
        print("GPU not available for data augmentation")
        parallel_calls = tf.data.AUTOTUNE

    # Apply per-image augmentations first
    augmented_ds = dataset.with_options(options).map(
        _augment_image_and_label,
        num_parallel_calls=parallel_calls
    )

    # Shuffle before batching to increase randomness
    augmented_ds = augmented_ds.shuffle(
        buffer_size=min(100, 1000),  # Limit buffer size to avoid memory issues
        reshuffle_each_iteration=True
    )

    # Batch the dataset
    augmented_ds = augmented_ds.batch(batch_size)

    # Apply batch-level augmentations (CutMix and MixUp) if Keras-CV is available
    if KERAS_CV_AVAILABLE:
        augmented_ds = augmented_ds.map(
            lambda x, y: apply_cutmix(x, y, alpha=0.2, prob=0.25),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        augmented_ds = augmented_ds.map(
            lambda x, y: apply_mixup(x, y, alpha=0.2, prob=0.25),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Use a small prefetch buffer to prevent memory warnings
    augmented_ds = augmented_ds.prefetch(buffer_size=2)

    return augmented_ds


if __name__ == "__main__":
    # Import the data_loader module to test the augmentation
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

    from data_loader import load_dataset
    import matplotlib.pyplot as plt

    # Configure TensorFlow to use CPU only for this test
    # Comment out if you want to use GPU
    # tf.config.set_visible_devices([], 'GPU')

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
            aug_image, aug_label = _augment_image_and_label(original_image, original_label)

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

            # Visualize the first two images in the augmented batch
            plt.figure(figsize=(15, 10))

            for i in range(min(4, aug_images.shape[0])):
                plt.subplot(2, 4, i+1)
                plt.title(f"Aug Image {i+1}")
                plt.imshow(aug_images[i].numpy())

                plt.subplot(2, 4, i+5)
                plt.title(f"Aug Label {i+1}")
                plt.imshow(aug_labels[i].numpy()[:,:,0], cmap='jet', vmin=0, vmax=4)

            plt.savefig('augmented_batch_example.png')
            plt.close()
            print("Augmented batch example saved as 'augmented_batch_example.png'")

        print("Augmentation pipeline test completed successfully!")
    except Exception as e:
        print(f"Error in augmentation pipeline: {e}")
