import numpy as np
import os

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

def tf_isin(x, values):
    # x: tensor of any shape
    # values: list or tensor of values to match
    comparisons = [tf.equal(x, v) for v in values]
    return tf.reduce_any(tf.stack(comparisons, axis=0), axis=0)

@tf.function
def rotate_image(image, angle, interpolation='nearest'):
    """Rotate image using native TensorFlow operations."""
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
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
def elastic_deform(image, label, alpha=20.0, sigma=4.0, grid_size=8):
    """Apply elastic deformation to simulate water surface variations."""
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    grid_x, grid_y = tf.meshgrid(tf.range(0, width, grid_size), tf.range(0, height, grid_size))
    grid_x = tf.cast(grid_x, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)

    dx = tf.random.normal(tf.shape(grid_x), mean=0.0, stddev=sigma) * alpha
    dy = tf.random.normal(tf.shape(grid_y), mean=0.0, stddev=sigma) * alpha

    dx = tf.image.resize(tf.expand_dims(dx, -1), [height, width], method='bilinear')[:,:,0]
    dy = tf.image.resize(tf.expand_dims(dy, -1), [height, width], method='bilinear')[:,:,0]

    x, y = tf.meshgrid(tf.range(width), tf.range(height))
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x_new = tf.clip_by_value(x + dx, 0, tf.cast(width - 1, tf.float32))
    y_new = tf.clip_by_value(y + dy, 0, tf.cast(height - 1, tf.float32))

    image_deformed = tf.image.resize(
        tf.expand_dims(image, 0),
        [height, width],
        method='bilinear',
        antialias=True
    )[0]
    label_deformed = tf.image.resize(
        tf.expand_dims(tf.cast(label, tf.float32), 0),
        [height, width],
        method='nearest',
        antialias=False
    )[0]
    return tf.cast(image_deformed, tf.float32), tf.cast(tf.round(label_deformed), tf.uint8)

@tf.function
def _augment_image_and_label(image, label):
    """Apply per-image augmentations with class-aware enhancements."""
    original_shape = tf.shape(image)
    h, w = original_shape[0], original_shape[1]
    combined = tf.concat([image, tf.cast(label, tf.float32) / 4.0], axis=2)

    if tf.random.uniform(()) > 0.5:
        combined = tf.image.flip_left_right(combined)

    if tf.random.uniform(()) > 0.5:
        angle = tf.random.uniform([], minval=-0.5236, maxval=0.5236)  # ±30°
        combined = rotate_image(combined, angle, interpolation='nearest')

    if tf.random.uniform(()) > 0.5:
        scale = tf.random.uniform([], minval=0.8, maxval=1.0)
        new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
        combined = tf.image.resize(combined, [new_h, new_w], method='nearest')
        combined = tf.image.resize_with_pad(combined, h, w, method='nearest')

    image = combined[..., :3]
    label = combined[..., 3:]

    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)

    if tf.random.uniform(()) > 0.5:
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.02)
        image = image + noise

    image, label = elastic_deform(image, label)

    # Check for rare classes using tf.isin
    flat_label = tf.reshape(label, [-1])
    has_rare = tf.reduce_any(tf_isin(flat_label, [1, 3]))
    if has_rare and tf.random.uniform(()) > 0.5:
        if tf.random.uniform(()) > 0.5:
            combined = tf.image.flip_left_right(combined)
        if tf.random.uniform(()) > 0.5:
            scale = tf.random.uniform([], minval=0.8, maxval=1.0)
            new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
            new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
            combined = tf.image.resize(combined, [new_h, new_w], method='nearest')
            combined = tf.image.resize_with_pad(combined, h, w, method='nearest')

    image = combined[..., :3]
    label = combined[..., 3:]

    image = tf.clip_by_value(image, 0, 1)
    label = label * 4.0
    label = tf.cast(tf.round(label), tf.uint8)

    return image, label

@tf.function
def apply_cutmix(images, labels, alpha=0.3, prob=0.5):
    """Apply CutMix targeting rare class regions."""
    if tf.random.uniform(()) > prob:
        return images, labels

    batch_size = tf.shape(images)[0]
    height, width = tf.shape(images)[1], tf.shape(images)[2]

    rare_mask = tf.reduce_any(tf.equal(labels, [1, 3]), axis=-1, keepdims=True)
    rare_indices = tf.where(rare_mask)

    if tf.size(rare_indices) == 0:
        return images, labels

    idx = tf.random.uniform([], maxval=tf.size(rare_indices), dtype=tf.int32)
    y_coord = tf.cast(rare_indices[idx][0], tf.int32)
    x_coord = tf.cast(rare_indices[idx][1], tf.int32)

    lambda_param = tf.random.uniform([], 0, 1)
    cut_ratio = tf.math.sqrt(1.0 - lambda_param)
    cut_h = tf.cast(tf.cast(height, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(width, tf.float32) * cut_ratio, tf.int32)

    x1 = tf.maximum(0, x_coord - tf.cast(cut_w // 2, tf.int32))
    y1 = tf.maximum(0, y_coord - tf.cast(cut_h // 2, tf.int32))
    x2 = tf.minimum(width, x_coord + tf.cast(cut_w // 2, tf.int32))
    y2 = tf.minimum(height, y_coord + tf.cast(cut_h // 2, tf.int32))

    mask = tf.ones([height, width, 1], dtype=tf.float32)
    cut_mask = tf.zeros([y2-y1, x2-x1, 1], dtype=tf.float32)
    paddings = [[y1, height - y2], [x1, width - x2], [0, 0]]
    cut_mask_padded = tf.pad(cut_mask, paddings)
    mask = mask * (1.0 - cut_mask_padded)

    rand_indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = images * mask + tf.gather(images, rand_indices, axis=0) * (1.0 - mask)
    mixed_labels = labels * tf.cast(mask, labels.dtype) + tf.gather(labels, rand_indices, axis=0) * tf.cast(1.0 - mask, labels.dtype)

    return mixed_images, mixed_labels

@tf.function
def apply_mixup(images, labels, alpha=0.3, prob=0.5):
    """Apply MixUp with soft label blending for segmentation."""
    if tf.random.uniform(()) > prob:
        return images, labels

    batch_size = tf.shape(images)[0]
    gamma = tf.random.gamma([1], alpha, 1)[0]
    lam = tf.minimum(gamma / (gamma + tf.random.gamma([1], alpha, 1)[0]), tf.ones([1])[0])

    rand_indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam * images + (1 - lam) * tf.gather(images, rand_indices, axis=0)

    mixed_labels = lam * tf.cast(labels, tf.float32) + (1 - lam) * tf.cast(tf.gather(labels, rand_indices, axis=0), tf.float32)
    mixed_labels = tf.cast(tf.clip_by_value(tf.round(mixed_labels), 0, 4), tf.uint8)

    return mixed_images, mixed_labels

def apply_augmentation(dataset, batch_size=8):
    """Apply augmentation pipeline compatible with data_loader.py."""
    was_batched = False
    try:
        shapes = tf.compat.v1.data.get_output_shapes(dataset)
        if len(shapes[0].as_list()) == 4:  # Batched images
            was_batched = True
            dataset = dataset.unbatch()
            print("Unbatched existing dataset before applying augmentation")
    except Exception:
        pass

    # Enable all tf.data optimizations
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.apply_default_optimizations = True
    options.deterministic = False  # Allow non-deterministic ops for speed

    # Use maximum parallelism based on available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    # Use more parallel calls for faster processing
    parallel_calls = tf.data.AUTOTUNE
    print(f"Using {'GPU' if gpus else 'CPU'} with AUTOTUNE parallel calls")

    # Apply per-image augmentations
    augmented_ds = dataset.with_options(options).map(
        _augment_image_and_label,
        num_parallel_calls=parallel_calls
    )

    # Shuffle before batching with larger buffer
    augmented_ds = augmented_ds.shuffle(
        buffer_size=2000,  # Increased buffer size
        reshuffle_each_iteration=True
    )

    # Batch the dataset
    augmented_ds = augmented_ds.batch(batch_size)

    # Apply batch-level augmentations
    augmented_ds = augmented_ds.map(
        lambda x, y: apply_cutmix(x, y, alpha=0.3, prob=0.5),
        num_parallel_calls=parallel_calls
    )
    augmented_ds = augmented_ds.map(
        lambda x, y: apply_mixup(x, y, alpha=0.3, prob=0.5),
        num_parallel_calls=parallel_calls
    )

    # Prefetch more batches for better performance
    augmented_ds = augmented_ds.prefetch(tf.data.AUTOTUNE)

    return augmented_ds

if __name__ == "__main__":
    from data_loader import load_dataset
    import matplotlib.pyplot as plt

    # Get the current mixed precision policy
    policy = tf.keras.mixed_precision.global_policy()
    print(f"Current mixed precision policy: {policy.name}")

    # Ensure input tensors use the correct precision during testing
    compute_dtype = policy.compute_dtype
    variable_dtype = policy.variable_dtype
    print(f"Compute dtype: {compute_dtype}, Variable dtype: {variable_dtype}")

    # Unpack the tuple returned by load_dataset
    test_dataset, _, _ = load_dataset(split='test', batch_size=1)
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
            # Cast to proper dtype before augmentation to maintain precision compatibility
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
        print("Augmentation examples saved as 'augmentation_examples.png'")
        break

    print("\nTesting the augmentation pipeline...")
    augmented_ds = apply_augmentation(test_dataset, batch_size=4)

    print("Original dataset:", test_dataset)
    print("Augmented dataset:", augmented_ds)

    try:
        for aug_images, aug_labels in augmented_ds.take(1):
            print(f"Augmented batch - Images shape: {aug_images.shape}, Labels shape: {aug_labels.shape}")
            print(f"Image dtype: {aug_images.dtype}, Label dtype: {aug_labels.dtype}")
            print(f"Image value range: {tf.reduce_min(aug_images).numpy()} to {tf.reduce_max(aug_images).numpy()}")
            print(f"Label values: {np.unique(aug_labels.numpy())}")

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
        # Print more detailed error information
        import traceback
        print(f"Error in augmentation pipeline: {e}")
        print(f"Error details: {traceback.format_exc()}")
