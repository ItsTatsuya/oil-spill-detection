import numpy as np
import os
import tensorflow as tf

# ANSI color codes (unchanged)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_info(message):
    print(f"{Colors.CYAN}[INFO] {message}{Colors.ENDC}")

def log_success(message):
    print(f"{Colors.GREEN}[SUCCESS] {message}{Colors.ENDC}")

def log_warning(message):
    print(f"{Colors.YELLOW}[WARNING] {message}{Colors.ENDC}")

def log_error(message):
    print(f"{Colors.RED}[ERROR] {message}{Colors.ENDC}")

def log_augmentation(message):
    print(f"{Colors.HEADER}{Colors.BOLD}[AUGMENTATION] {message}{Colors.ENDC}")

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

@tf.function
def rotate_image(image, angle, interpolation='nearest'):
    height = 384
    width = 384
    height_float = tf.cast(height, tf.float32)
    width_float = tf.cast(width, tf.float32)
    center_x = width_float / 2.0
    center_y = height_float / 2.0
    costheta = tf.cos(angle)
    sintheta = tf.sin(angle)
    a0, a1 = costheta, -sintheta
    a2 = center_x - center_x * costheta + center_y * sintheta
    b0, b1 = sintheta, costheta
    b2 = center_y - center_x * sintheta - center_y * costheta
    transforms = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0], axis=0)
    transforms = tf.reshape(transforms, [1, 8])
    interpolation = interpolation.upper()
    rotated_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=transforms,
        output_shape=[height, width],
        interpolation=interpolation,
        fill_mode="CONSTANT",
        fill_value=0.0
    )
    return tf.squeeze(rotated_image, 0)

@tf.function(reduce_retracing=True)
def _augment_image_and_label(image, label):
    """Augment a single image-label pair (unbatched)."""
    image = tf.ensure_shape(image, [384, 384, 1])
    label = tf.ensure_shape(label, [384, 384, 1])
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.uint8)

    # Flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)

    # Rotate
    if tf.random.uniform(()) > 0.5:
        angle = tf.random.uniform([], minval=-0.1745, maxval=0.1745)  # ±10°
        image = rotate_image(image, angle, interpolation='bilinear')
        label = rotate_image(label, angle, interpolation='nearest')

    # Photometric augmentations
    image = tf.image.random_brightness(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.95, upper=1.05)

    # Clip values
    image = tf.clip_by_value(image, 0.0, 1.0)

    return tf.cast(image, tf.float16), tf.cast(label, tf.uint8)

@tf.function(reduce_retracing=True)
def apply_augmentation(dataset, batch_size=2):
    """Augmentation pipeline for batched dataset."""
    # Optimize dataset
    options = tf.data.Options()
    options.experimental_deterministic = False
    dataset = dataset.with_options(options)

    # Unbatch to apply augmentations to individual samples
    dataset = dataset.unbatch()

    # Apply augmentations to individual samples
    dataset = dataset.map(
        _augment_image_and_label,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Re-batch
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Simplified CutMix
    @tf.function
    def simplified_cutmix(images, labels):
        batch_size = tf.shape(images)[0]
        height, width = 384, 384
        should_apply = tf.random.uniform(()) < 0.2

        def apply_mix():
            patch_size = 96
            y_pos = tf.random.uniform([], 0, height - patch_size, dtype=tf.int32)
            x_pos = tf.random.uniform([], 0, width - patch_size, dtype=tf.int32)
            mask = tf.zeros([height, width, 1], dtype=images.dtype)
            patch = tf.ones([patch_size, patch_size, 1], dtype=images.dtype)
            paddings = [[y_pos, height - y_pos - patch_size], [x_pos, width - x_pos - patch_size], [0, 0]]
            mask = tf.pad(patch, paddings)
            mask = tf.expand_dims(mask, 0)
            mask = tf.tile(mask, [batch_size, 1, 1, 1])
            indices = tf.random.shuffle(tf.range(batch_size))
            shuffled_images = tf.gather(images, indices)
            shuffled_labels = tf.gather(labels, indices)
            mixed_images = images * (1 - mask) + shuffled_images * mask
            mixed_labels = labels * tf.cast(1 - mask, tf.uint8) + shuffled_labels * tf.cast(mask, tf.uint8)
            return mixed_images, mixed_labels

        return tf.cond(should_apply, apply_mix, lambda: (images, labels))

    dataset = dataset.map(simplified_cutmix, num_parallel_calls=tf.data.AUTOTUNE)

    # Ensure final shapes
    dataset = dataset.map(
        lambda images, labels: (
            tf.ensure_shape(images, [None, 384, 384, 1]),
            tf.ensure_shape(labels, [None, 384, 384, 1])
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Cache and prefetch
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

    return dataset

# Main block unchanged
if __name__ == "__main__":
    from data_loader import load_dataset
    import matplotlib.pyplot as plt
    policy = tf.keras.mixed_precision.global_policy()
    log_info(f"Mixed precision policy: {policy.name}")
    compute_dtype = policy.compute_dtype
    variable_dtype = policy.variable_dtype
    log_info(f"Compute dtype: {compute_dtype}, Variable dtype: {variable_dtype}")
    test_dataset, _, _ = load_dataset(split='test')
    for image, label in test_dataset.take(1):
        original_image = image[0].numpy()
        original_label = label[0].numpy()
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.imshow(original_image)
        plt.subplot(2, 3, 4)
        plt.title("Original Label")
        plt.imshow(original_label[:,:,0], cmap='jet', vmin=0, vmax=4)
        for i in range(2):
            img_tensor = tf.cast(original_image, compute_dtype)
            aug_image, aug_label = _augment_image_and_label(img_tensor, original_label)
            plt.subplot(2, 3, i+2)
            plt.title(f"Augmented Image {i+1}")
            plt.imshow(aug_image.numpy())
            plt.subplot(2, 3, i+5)
            plt.title(f"Augmented Label {i+1}")
            plt.imshow(aug_label.numpy()[:,:,0], cmap='jet', vmin=0, vmax=4)
        plt.tight_layout()
        plt.savefig('augmentation_examples.png')
        plt.close()
        log_success("Augmentation examples saved as 'augmentation_examples.png'")
        break
    log_augmentation("Testing augmentation pipeline...")
    augmented_ds = apply_augmentation(test_dataset, batch_size=2)
    try:
        for aug_images, aug_labels in augmented_ds.take(1):
            log_info(f"Augmented batch - Images shape: {aug_images.shape}, Labels shape: {aug_labels.shape}")
            log_info(f"Image dtype: {aug_images.dtype}, Label dtype: {aug_labels.dtype}")
            log_info(f"Image value range: {tf.reduce_min(aug_images).numpy()} to {tf.reduce_max(aug_images).numpy()}")
            log_info(f"Label values: {np.unique(aug_labels.numpy())}")
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
            log_success("Augmented batch example saved as 'augmented_batch_example.png'")
        log_success("Augmentation pipeline test completed successfully!")
    except Exception as e:
        import traceback
        log_error(f"Error in augmentation pipeline: {e}")
        log_error(f"Error details: {traceback.format_exc()}")
