import os
import tensorflow as tf
import logging
import glob
from tensorflow.keras import mixed_precision # type: ignore

policy = mixed_precision.global_policy()
print(f"Data loader using mixed precision policy: {policy.name}")

logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def analyze_class_distribution(label_paths):
    weights = tf.constant([0.05, 0.4, 0.1, 0.35, 0.1], dtype=tf.float32)
    adjusted_weights = {i: float(weights[i]) for i in range(5)}
    tf.print("Using predefined class weights:", adjusted_weights)
    return adjusted_weights

@tf.function
def sample_patches(image, label, patch_size=384):
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]

    if height < patch_size or width < patch_size:
        image = tf.image.resize_with_pad(image, patch_size, patch_size)
        label = tf.image.resize_with_pad(label, patch_size, patch_size, method='nearest')
        return image, label

    if tf.random.uniform(()) < 0.5:
        label_classes = tf.cast(tf.squeeze(label), tf.int32)

        has_ship = tf.reduce_any(tf.equal(label_classes, 3))
        has_oil = tf.reduce_any(tf.equal(label_classes, 1))

        if has_ship or has_oil:
            if has_ship:
                interesting_class = 3  # Ship
            else:
                interesting_class = 1  # Oil spill

            mask = tf.cast(tf.equal(label_classes, interesting_class), tf.int32)
            indices = tf.where(tf.equal(mask, 1))

            if tf.shape(indices)[0] > 0:
                random_idx = tf.random.uniform((), 0, tf.shape(indices)[0], dtype=tf.int32)
                center_y, center_x = indices[random_idx][0], indices[random_idx][1]

                half_size = patch_size // 2

                top = tf.maximum(0, center_y - half_size)
                left = tf.maximum(0, center_x - half_size)

                top = tf.minimum(height - patch_size, top)
                left = tf.minimum(width - patch_size, left)

                image_patch = tf.image.crop_to_bounding_box(image, top, left, patch_size, patch_size)
                label_patch = tf.image.crop_to_bounding_box(label, top, left, patch_size, patch_size)

                return image_patch, label_patch

    top = tf.random.uniform((), 0, height - patch_size + 1, dtype=tf.int32)
    left = tf.random.uniform((), 0, width - patch_size + 1, dtype=tf.int32)

    image_patch = tf.image.crop_to_bounding_box(image, top, left, patch_size, patch_size)
    label_patch = tf.image.crop_to_bounding_box(label, top, left, patch_size, patch_size)

    return image_patch, label_patch

def parse_image_label(image_path, label_path):
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

def load_dataset(data_dir='dataset', split='train', batch_size=16):
    if split not in ['train', 'test']:
        raise ValueError(f"Invalid split: {split}")

    split_dir = os.path.join(data_dir, split)
    image_dir = os.path.join(split_dir, 'images')
    label_dir = os.path.join(split_dir, 'labels_1D')

    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.png')))

    if len(image_files) != len(label_files):
        raise ValueError(f"Image-label mismatch: {len(image_files)} images, {len(label_files)} labels")

    dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))
    num_samples = len(image_files)
    num_batches = num_samples // batch_size + (1 if num_samples % batch_size > 0 else 0)

    class_weights = analyze_class_distribution(label_files) if split == 'train' else None

    dataset = dataset.map(parse_image_label, num_parallel_calls=tf.data.AUTOTUNE)

    if split == 'train':
        dataset = dataset.map(sample_patches, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=min(2000, num_samples), seed=42)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        options = tf.data.Options()
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.map_fusion = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.noop_elimination = True
        options.experimental_optimization.apply_default_optimizations = True
        options.deterministic = False
        dataset = dataset.with_options(options)

        dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda x, y: (tf.cast(x, policy.compute_dtype), y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

        options = tf.data.Options()
        options.experimental_optimization.apply_default_optimizations = True
        dataset = dataset.with_options(options)

    print(f"Loaded {split} dataset with {num_samples} samples, {num_batches} batches (batch_size={batch_size})")
    return dataset, class_weights, num_batches
