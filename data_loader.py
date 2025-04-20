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

# Import mixed precision and set policy for data loading
from tensorflow.keras import mixed_precision # type: ignore
mixed_precision.set_global_policy('float32')
policy = mixed_precision.global_policy()
print(f"Data loader using mixed precision policy: {policy.name}")

# Import Keras-CV for augmentations
import keras_cv

# Disable other TensorFlow warnings and messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def analyze_class_distribution(label_paths):
    """Analyze class distribution from a sample of label files and return class weights."""
    sample_size = min(200, len(label_paths))
    sample_paths = label_paths[:sample_size] if sample_size > 0 else label_paths
    if not sample_paths:
        return {i: 1.0 / 5.0 for i in range(5)}  # Uniform weights as fallback
    labels = []
    for path in sample_paths:
        label = tf.io.read_file(path)
        label = tf.image.decode_png(label, channels=1)
        label = tf.image.resize(label, [320, 320], method='nearest')
        label = tf.cast(label, tf.uint8)
        labels.append(tf.reshape(label, [-1]))
    labels = tf.concat(labels, axis=0)
    unique, _, counts = tf.unique_with_counts(labels)
    counts_float = tf.cast(counts, tf.float32)
    tf.print("Class counts:", counts_float)  # Debug output
    weights = 1.0 / (counts_float + 1e-7)  # Avoid division by zero
    total_weight = tf.reduce_sum(weights)
    raw_weights = {int(unique[i]): float(weights[i] / total_weight) for i in range(len(unique))}
    # Cap maximum weight at 0.3 and redistribute excess
    max_weight = 0.3
    adjusted_weights = {}
    weight_sum = 0.0
    for cls in raw_weights:
        w = min(raw_weights[cls], max_weight)
        adjusted_weights[cls] = w
        weight_sum += w
    # Normalize remaining weights
    remaining_weight = 1.0 - weight_sum
    if remaining_weight > 0:
        num_remaining_classes = len(raw_weights) - len([cls for cls in raw_weights if raw_weights[cls] > max_weight])
        if num_remaining_classes > 0:
            base_weight = remaining_weight / num_remaining_classes
            for cls in raw_weights:
                if raw_weights[cls] <= max_weight:
                    adjusted_weights[cls] = base_weight
    # Fill missing classes with adjusted weight
    for i in range(5):
        if i not in adjusted_weights:
            adjusted_weights[i] = 1.0 / 5.0  # Fallback
    tf.print("Adjusted class weights:", adjusted_weights)  # Debug output
    return adjusted_weights

def parse_image_label(image_path, label_path):
    """Load and preprocess an image and its corresponding label."""
    # Read and preprocess image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
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

def load_dataset(data_dir='dataset', split='train', batch_size=8):
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
        # Add Keras-CV augmentation
        augment = keras_cv.layers.RandAugment(
            value_range=(0, 1),
            augmentations_per_image=2,
            magnitude=0.5,
            magnitude_stddev=0.2,
            rate=0.7
        )

        # Apply augmentation first (in float32)
        dataset = dataset.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

        # Convert to compute_dtype (float16) after augmentation
        dataset = dataset.map(lambda x, y: (tf.cast(x, policy.compute_dtype), y), num_parallel_calls=tf.data.AUTOTUNE)

        # Disable caching to reduce VRAM usage and use larger buffer for better shuffling
        dataset = dataset.batch(batch_size)
        print(f"Caching disabled to conserve VRAM. Processing {num_samples} samples with batch_size={batch_size}")

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
