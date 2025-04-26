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
def random_flip(image, label, p_horizontal=0.5, p_vertical=0.3):
    """Apply random horizontal and vertical flips to image and label."""
    if tf.random.uniform(()) < p_horizontal:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    if tf.random.uniform(()) < p_vertical:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)
    return image, label

@tf.function(reduce_retracing=True)
def random_rotation(image, label, max_angle=15.0, p_90deg=0.2):
    """Apply random rotation to image and label."""
    max_angle_rad = max_angle * (np.pi / 180.0)
    if tf.random.uniform(()) < p_90deg:
        k = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        label = tf.image.rot90(label, k=k)
    else:
        angle = tf.random.uniform([], minval=-max_angle_rad, maxval=max_angle_rad)
        image = rotate_image(image, angle, interpolation='bilinear')
        label = rotate_image(label, angle, interpolation='nearest')
    return image, label

@tf.function(reduce_retracing=True)
def add_speckle_noise(image, label, mean=0.0, stddev=0.1, p=0.7):
    """Add speckle noise to SAR imagery."""
    if tf.random.uniform(()) < p:
        noise = tf.random.normal(tf.shape(image), mean=mean, stddev=stddev, dtype=image.dtype)
        noisy_image = image * (1 + noise)
        noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
        return noisy_image, label
    else:
        return image, label

@tf.function(reduce_retracing=True)
def augment_single_sample(image, label):
    """Apply augmentations to a single image-label pair."""
    image, label = random_flip(image, label)
    image, label = random_rotation(image, label, max_angle=15.0, p_90deg=0.2)
    image, label = add_speckle_noise(image, label, stddev=0.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return tf.cast(image, tf.float16), tf.cast(label, tf.uint8)

@tf.function(reduce_retracing=True)
def apply_augmentation(dataset, batch_size=2, cutmix_prob=0.3):
    """Apply the complete augmentation pipeline to a dataset."""
    dataset = dataset.unbatch()
    dataset = dataset.map(augment_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
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
            aug_image, aug_label = augment_single_sample(img_tensor, original_label)
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
