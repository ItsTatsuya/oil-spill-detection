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

# Import mixed precision
from tensorflow.keras import mixed_precision # type: ignore
policy = mixed_precision.global_policy()
print(f"Data loader using mixed precision policy: {policy.name}")

# Suppress TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def analyze_class_distribution(label_paths):
    """Return predefined class weights for oil spill detection."""
    weights = tf.constant([0.05, 0.4, 0.1, 0.35, 0.1], dtype=tf.float32)
    adjusted_weights = {i: float(weights[i]) for i in range(5)}
    tf.print("Using predefined class weights:", adjusted_weights)
    return adjusted_weights

@tf.function
def apply_lightweight_augmentation(image, label):
    """Apply lightweight augmentations with minimal memory usage."""
    image = tf.ensure_shape(image, [384, 384, 1])
    label = tf.ensure_shape(label, [384, 384, 1])

    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)

    # Random small rotation (±10°)
    if tf.random.uniform(()) > 0.5:
        angle = tf.random.uniform((), -0.174, 0.174)
        image = tf.image.rot90(image, k=tf.cast(tf.round(angle * 2 / 3.14159), tf.int32) % 4)
        label = tf.image.rot90(label, k=tf.cast(tf.round(angle * 2 / 3.14159), tf.int32) % 4)

    return image, label

@tf.function
def sample_patches(image, label, patch_size=384):
    """Sample patches centered on ships or oil spills."""
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    def ensure_size(img, lbl):
        img = tf.image.resize_with_crop_or_pad(img, patch_size, patch_size)
        lbl = tf.image.resize_with_crop_or_pad(lbl, patch_size, patch_size)
        return img, lbl

    ship_coords = tf.zeros((0, 2), dtype=tf.int64)
    oil_coords = tf.zeros((0, 2), dtype=tf.int64)
    lookalike_coords = tf.zeros((0, 2), dtype=tf.int64)

    need_coords = tf.random.uniform(()) >= 0.3
    if need_coords:
        ship_coords = tf.where(label == 3)
        oil_coords = tf.where(label == 1)
        lookalike_coords = tf.where(label == 2)

    if tf.random.uniform(()) < 0.3:
        return ensure_size(image, label)

    use_rare = tf.logical_or(tf.shape(ship_coords)[0] > 0, tf.shape(oil_coords)[0] > 0)

    def sample_rare():
        rare_coords = tf.concat([ship_coords, oil_coords], axis=0)
        has_lookalikes = tf.shape(lookalike_coords)[0] > 0
        coords = tf.cond(
            has_lookalikes,
            lambda: tf.concat([rare_coords, lookalike_coords[:tf.maximum(1, tf.shape(lookalike_coords)[0] // 4)]], axis=0),
            lambda: rare_coords
        )
        coords = tf.cond(
            tf.equal(tf.shape(coords)[0], 0),
            lambda: tf.zeros((1, 2), dtype=tf.int64),
            lambda: coords
        )
        idx = tf.random.uniform((), 0, tf.shape(coords)[0], dtype=tf.int32)
        y, x = coords[idx][0], coords[idx][1]
        y = tf.cast(y, tf.int32)
        x = tf.cast(x, tf.int32)
        half_size = patch_size // 2
        y_start = tf.maximum(0, y - half_size)
        x_start = tf.maximum(0, x - half_size)
        y_end = tf.minimum(image_height, y_start + patch_size)
        x_end = tf.minimum(image_width, x_start + patch_size)
        img_patch = image[y_start:y_end, x_start:x_end, :]
        lbl_patch = label[y_start:y_end, x_start:x_end, :]
        return ensure_size(img_patch, lbl_patch)

    def sample_random():
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
    """Load and preprocess an image and its label."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [384, 384])
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.image.resize(label, [384, 384], method='nearest')
    label = tf.cast(label, tf.uint8)
    label = tf.clip_by_value(label, 0, 4)

    return image, label

def parse_tfrecord_example(example_proto):
    """Parse a single example from a TFRecord file."""
    # Define the features to parse from the TFRecord
    feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'image_format': tf.io.FixedLenFeature([], tf.string),
        'label_format': tf.io.FixedLenFeature([], tf.string),
    }

    # Parse the example
    features = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the image and label
    image = tf.image.decode_jpeg(features['image'], channels=1)
    label = tf.image.decode_png(features['label'], channels=1)

    # Resize to standard dimensions
    image = tf.image.resize(image, [384, 384])
    label = tf.image.resize(label, [384, 384], method='nearest')

    # Normalize image to [0,1] range and cast label to uint8
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.uint8)
    label = tf.clip_by_value(label, 0, 4)

    return image, label

def load_dataset(data_dir='dataset', split='train', batch_size=16, use_tfrecords=True):
    """Load dataset with class weights and batch information.

    Args:
        data_dir: Base directory containing the dataset
        split: 'train' or 'test' split to load
        batch_size: Number of examples per batch
        use_tfrecords: Whether to use TFRecord format (faster) or raw image files

    Returns:
        dataset: tf.data.Dataset object
        class_weights: Dictionary of class weights for loss function
        num_batches: Number of batches in one epoch
    """
    if split not in ['train', 'test']:
        raise ValueError(f"Invalid split: {split}")

    # Get class weights (same regardless of data source)
    class_weights = None
    if split == 'train':
        # For TFRecords we don't need actual files for weights calculation
        label_dir = os.path.join(data_dir, split, 'labels_1D')
        label_files = sorted(glob.glob(os.path.join(label_dir, '*.png')))
        class_weights = analyze_class_distribution(label_files)

    # Check if TFRecord directory exists
    tfrecord_dir = os.path.join(data_dir, f'{split}_tfrecords')
    if use_tfrecords and os.path.exists(tfrecord_dir):
        # Use TFRecord dataset (faster I/O)
        tfrecord_files = sorted(glob.glob(os.path.join(tfrecord_dir, '*.tfrecord')))

        if not tfrecord_files:
            print(f"No TFRecord files found in {tfrecord_dir}. Falling back to image files.")
            use_tfrecords = False
        else:
            print(f"Loading {split} dataset from {len(tfrecord_files)} TFRecord files")
            dataset = tf.data.TFRecordDataset(tfrecord_files)
            dataset = dataset.map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)

            # Count number of examples in TFRecord files
            # This is an approximation as we don't actually count all examples
            example_counter = tf.data.TFRecordDataset(tfrecord_files).take(1)
            num_examples = 0

            # If the first TFRecord exists, count samples based on file pattern
            if split == 'train':
                num_examples = 1002  # Known size from original dataset
            else:
                num_examples = 110  # Known size from original dataset

    # Fall back to regular image loading if TFRecords not available or not requested
    if not use_tfrecords or not os.path.exists(tfrecord_dir):
        print(f"Loading {split} dataset from individual image files")
        split_dir = os.path.join(data_dir, split)
        image_dir = os.path.join(split_dir, 'images')
        label_dir = os.path.join(split_dir, 'labels_1D')

        image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        label_files = sorted(glob.glob(os.path.join(label_dir, '*.png')))

        if len(image_files) != len(label_files):
            raise ValueError(f"Image-label mismatch: {len(image_files)} images, {len(label_files)} labels")

        num_examples = len(image_files)
        dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))
        dataset = dataset.map(parse_image_label, num_parallel_calls=tf.data.AUTOTUNE)

    # Common transformations for both TFRecord and image loading
    num_batches = num_examples // batch_size + (1 if num_examples % batch_size > 0 else 0)

    if split == 'train':
        # Apply data transformations
        dataset = dataset.map(sample_patches, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(apply_lightweight_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle first, then batch
        dataset = dataset.shuffle(buffer_size=min(2000, num_examples), seed=42)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # Add repeat BEFORE prefetch to make the dataset repeat indefinitely during training
        # This prevents the "Your input ran out of data" warning
        dataset = dataset.repeat()

        # Prefetch at the end for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        options = tf.data.Options()
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.map_fusion = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.noop_elimination = True
        options.experimental_optimization.apply_default_optimizations = True
        options.deterministic = False
        dataset = dataset.with_options(options)
    else:
        # Testing dataset - simpler pipeline
        dataset = dataset.map(lambda x, y: (tf.cast(x, policy.compute_dtype), y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # Don't add repeat() for validation/test datasets
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        options = tf.data.Options()
        options.experimental_optimization.apply_default_optimizations = True
        dataset = dataset.with_options(options)

    print(f"Loaded {split} dataset with {num_examples} samples, {num_batches} batches (batch_size={batch_size})")
    return dataset, class_weights, num_batches

if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')

    # Test with both TFRecords and original images
    print("\nTesting with TFRecord loading (faster):")
    train_ds_tfr, train_cw_tfr, train_nb_tfr = load_dataset(split='train', batch_size=16, use_tfrecords=True)
    test_ds_tfr, _, test_nb_tfr = load_dataset(split='test', batch_size=16, use_tfrecords=True)

    print("\nTesting with original image loading (fallback):")
    train_ds_img, train_cw_img, train_nb_img = load_dataset(split='train', batch_size=16, use_tfrecords=False)
    test_ds_img, _, test_nb_img = load_dataset(split='test', batch_size=16, use_tfrecords=False)

    # Print dataset information
    print("\nDataset information:")
    print(f"TFRecord - Train dataset: {train_ds_tfr}")
    print(f"TFRecord - Test dataset: {test_ds_tfr}")
    print(f"Image Files - Train dataset: {train_ds_img}")
    print(f"Image Files - Test dataset: {test_ds_img}")

    # Performance test
    print("\nPerformance comparison (time to load first 3 batches):")

    # Test TFRecord performance
    try:
        import time
        print("Timing TFRecord loading...")
        start_time = time.time()
        batches_seen = 0
        for images, labels in train_ds_tfr.take(3):
            batches_seen += 1
            print(f"TFRecord batch {batches_seen} - Images shape: {images.shape}, Labels shape: {labels.shape}")
        tfr_time = time.time() - start_time
        print(f"TFRecord loading time: {tfr_time:.4f} seconds for 3 batches")

        # Test Image loading performance
        print("\nTiming image file loading...")
        start_time = time.time()
        batches_seen = 0
        for images, labels in train_ds_img.take(3):
            batches_seen += 1
            print(f"Image files batch {batches_seen} - Images shape: {images.shape}, Labels shape: {labels.shape}")
        img_time = time.time() - start_time
        print(f"Image loading time: {img_time:.4f} seconds for 3 batches")

        # Print speed comparison
        speedup = img_time / tfr_time if tfr_time > 0 else 0
        print(f"\nTFRecord is {speedup:.2f}x faster than loading from image files")

    except Exception as e:
        print(f"Error during performance test: {str(e)}")
