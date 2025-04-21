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

    # Always keep label as float32 - important fix!
    label_deformed = tf.image.resize(
        tf.expand_dims(tf.cast(label, tf.float32), 0),
        [height, width],
        method='nearest',
        antialias=False
    )[0]

    # Return both as float32 (don't convert label back to uint8)
    return tf.cast(image_deformed, tf.float32), tf.cast(label_deformed, tf.float32)

@tf.function
def ship_specific_augmentation(image, label):
    """Special augmentation pipeline specifically for images containing ships (class 3)"""
    # Convert label to float32 at the very beginning
    label_float = tf.cast(label, tf.float32) / 4.0

    original_shape = tf.shape(image)
    h, w = original_shape[0], original_shape[1]
    combined = tf.concat([image, label_float], axis=2)

    # More aggressive rotation for ships (up to ±45°)
    angle = tf.random.uniform([], minval=-0.7854, maxval=0.7854)  # ±45°
    combined = rotate_image(combined, angle, interpolation='nearest')

    # More aggressive scaling to help learn ships at various sizes
    scale = tf.random.uniform([], minval=0.6, maxval=1.2)  # More extreme scaling
    new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    combined = tf.image.resize(combined, [new_h, new_w], method='nearest')
    combined = tf.image.resize_with_pad(combined, h, w, method='nearest')

    # Always apply left-right flip for ships
    combined = tf.image.flip_left_right(combined)

    # Random up-down flip (ships can be oriented in any direction)
    if tf.random.uniform(()) > 0.5:
        combined = tf.image.flip_up_down(combined)

    # Split back into image and label
    image = combined[..., :3]
    label_float = combined[..., 3:]

    # Color augmentation for ships (more aggressive)
    image = tf.image.random_brightness(image, max_delta=0.3)  # Increased from 0.2
    image = tf.image.random_contrast(image, lower=0.7, upper=1.4)  # Increased range
    image = tf.image.random_saturation(image, lower=0.7, upper=1.4)  # Increased range
    image = tf.image.random_hue(image, max_delta=0.2)  # Increased from 0.1

    # Add more noise to simulate various sea conditions
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.05)  # Increased noise
    image = image + noise

    # Apply stronger elastic deformation
    image, label_float = elastic_deform(image, label_float, alpha=30.0, sigma=5.0)  # Increased strength

    # Clip values and finalize
    image = tf.clip_by_value(image, 0, 1)
    # Convert back to original class labels (0-4)
    label = tf.cast(tf.round(label_float * 4.0), tf.uint8)

    return image, label

@tf.function
def _augment_image_and_label(image, label):
    """Apply per-image augmentations with class-aware enhancements."""
    # Check if image contains ships (class 3)
    flat_label = tf.reshape(label, [-1])
    has_ship = tf.reduce_any(tf.equal(flat_label, 3))

    # Apply special ship augmentation with high probability if ships are present
    if has_ship and tf.random.uniform(()) > 0.2:  # 80% chance to use special augmentation
        return ship_specific_augmentation(image, label)

    # Standard augmentation for non-ship images
    # Convert label to float32 at the beginning
    label_float = tf.cast(label, tf.float32) / 4.0

    original_shape = tf.shape(image)
    h, w = original_shape[0], original_shape[1]
    combined = tf.concat([image, label_float], axis=2)

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
    label_float = combined[..., 3:]

    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)

    if tf.random.uniform(()) > 0.5:
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.02)
        image = image + noise

    image, label_float = elastic_deform(image, label_float)

    # Check for rare classes using tf.isin
    # Convert label_float back to uint8 for class checking
    temp_label = tf.cast(tf.round(label_float * 4.0), tf.uint8)
    flat_label = tf.reshape(temp_label, [-1])
    has_rare = tf.reduce_any(tf_isin(flat_label, [1, 3]))

    if has_rare and tf.random.uniform(()) > 0.5:
        # Create combined tensor again for additional transformations
        combined = tf.concat([image, label_float], axis=2)

        if tf.random.uniform(()) > 0.5:
            combined = tf.image.flip_left_right(combined)
        if tf.random.uniform(()) > 0.5:
            scale = tf.random.uniform([], minval=0.8, maxval=1.0)
            new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
            new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
            combined = tf.image.resize(combined, [new_h, new_w], method='nearest')
            combined = tf.image.resize_with_pad(combined, h, w, method='nearest')

        image = combined[..., :3]
        label_float = combined[..., 3:]

    image = tf.clip_by_value(image, 0, 1)
    # Convert back to original label format
    label = tf.cast(tf.round(label_float * 4.0), tf.uint8)

    return image, label

@tf.function
def apply_cutmix(images, labels, alpha=0.3, prob=0.5):
    """Apply CutMix targeting rare class regions."""
    if tf.random.uniform(()) > prob:
        return images, labels

    batch_size = tf.shape(images)[0]
    height, width = tf.shape(images)[1], tf.shape(images)[2]

    # Identify pixels belonging to rare classes (1: Oil Spill, 3: Ship)
    # Ensure labels are integer type for comparison
    int_labels = tf.cast(labels, tf.int32)

    # Modified: Prioritize Ship class (3) more highly than Oil Spill class (1)
    # Create two separate masks for ships and oil spills
    ship_mask = tf.equal(int_labels, 3)
    oil_mask = tf.equal(int_labels, 1)

    # Check if the batch contains ships
    has_ships = tf.reduce_any(ship_mask)

    # If ships are present, use them with higher probability (80%)
    use_ships = tf.logical_and(has_ships, tf.random.uniform(()) < 0.8)

    # Create the effective mask based on the decision
    rare_mask = tf.cond(
        use_ships,
        lambda: ship_mask,  # If using ships, only consider ship pixels
        lambda: tf.logical_or(ship_mask, oil_mask)  # Otherwise use both
    )

    # Find indices [batch_idx, y, x, channel] of rare pixels
    rare_indices = tf.where(rare_mask)
    num_rare_pixels = tf.shape(rare_indices)[0]

    # If no rare pixels are found in the batch, return original images/labels
    if num_rare_pixels == 0:
        return images, labels

    # Select a random rare pixel location from the found indices
    idx = tf.random.uniform([], maxval=num_rare_pixels, dtype=tf.int32)

    # Extract coordinates [batch_idx, y, x, channel] from the selected rare pixel index
    # Convert all indices to int32 to ensure consistent data types
    batch_idx = tf.cast(rare_indices[idx][0], tf.int32)
    y_coord = tf.cast(rare_indices[idx][1], tf.int32)
    x_coord = tf.cast(rare_indices[idx][2], tf.int32)
    channel_idx = tf.cast(0, tf.int32)  # Always use channel 0

    # Get the cut size - make it larger for ships to better capture their structure
    is_ship_pixel = tf.equal(int_labels[batch_idx, y_coord, x_coord, channel_idx], 3)

    # Use different cut ratios based on class
    lambda_param = tf.random.uniform([], 0, 1)
    cut_ratio = tf.cond(
        is_ship_pixel,
        lambda: tf.math.sqrt(1.0 - lambda_param) * 1.5,  # Larger for ships
        lambda: tf.math.sqrt(1.0 - lambda_param)  # Normal for other classes
    )

    cut_h = tf.cast(tf.cast(height, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(width, tf.float32) * cut_ratio, tf.int32)

    # Center the cut box around the selected rare pixel coordinates
    x1 = tf.maximum(0, x_coord - cut_w // 2)
    y1 = tf.maximum(0, y_coord - cut_h // 2)
    x2 = tf.minimum(width, x_coord + (cut_w + 1) // 2)
    y2 = tf.minimum(height, y_coord + (cut_h + 1) // 2)

    # Ensure width and height are at least 1
    actual_cut_w = x2 - x1
    actual_cut_h = y2 - y1

    # Create a mask for the patch area
    actual_cut_h = tf.maximum(0, actual_cut_h)
    actual_cut_w = tf.maximum(0, actual_cut_w)
    patch_mask = tf.zeros([actual_cut_h, actual_cut_w, 1], dtype=tf.float32)

    # Pad the patch mask to the full image size
    paddings = [[y1, height - y2], [x1, width - x2], [0, 0]]
    paddings = tf.maximum(0, paddings)

    # Inverse mask: 1s outside the patch, 0s inside
    inverse_mask = 1.0 - tf.pad(patch_mask, paddings, constant_values=1.0)
    # Mask: 0s outside the patch, 1s inside
    mask = 1.0 - inverse_mask

    # Mixing images and labels
    rand_indices = tf.random.shuffle(tf.range(batch_size))
    mask_img = tf.cast(mask, images.dtype)
    mask_lbl = tf.cast(mask, tf.float32)

    # Cast labels to float32 for mixing calculation
    labels_float = tf.cast(labels, tf.float32)

    # Perform mixing using float32
    mixed_labels_float = labels_float * (1.0 - mask_lbl) + tf.gather(labels_float, rand_indices, axis=0) * mask_lbl

    # Cast mixed labels back to uint8
    mixed_labels = tf.cast(tf.clip_by_value(mixed_labels_float, 0, 4), tf.uint8)

    mixed_images = images * (1.0 - mask_img) + tf.gather(images, rand_indices, axis=0) * mask_img

    return mixed_images, mixed_labels

@tf.function
def apply_mixup(images, labels, alpha=0.3, prob=0.5):
    """Apply MixUp with soft label blending for segmentation."""
    if tf.random.uniform(()) > prob:
        return images, labels

    batch_size = tf.shape(images)[0]
    # Use tf.random.gamma to generate Beta-distributed lambda
    # lam ~ Beta(alpha, alpha) is equivalent to G1 / (G1 + G2) where G1, G2 ~ Gamma(alpha, 1)
    gamma1 = tf.random.gamma([batch_size, 1, 1, 1], alpha=alpha, beta=1.0)
    gamma2 = tf.random.gamma([batch_size, 1, 1, 1], alpha=alpha, beta=1.0)
    lam = gamma1 / (gamma1 + gamma2 + 1e-7) # Add epsilon for stability

    # Cast lam to the same dtype as images (e.g., float16 for mixed precision)
    lam_img_dtype = tf.cast(lam, images.dtype)

    rand_indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam_img_dtype * images + (1.0 - lam_img_dtype) * tf.gather(images, rand_indices, axis=0) # Use 1.0 for float subtraction

    # Ensure labels are float32 for mixing (as before)
    labels_float = tf.cast(labels, tf.float32)
    # Cast lam back to float32 for label mixing
    lam_float32 = tf.cast(lam, tf.float32)
    mixed_labels = lam_float32 * labels_float + (1.0 - lam_float32) * tf.gather(labels_float, rand_indices, axis=0)
    # Convert back to uint8 after mixing
    mixed_labels = tf.cast(tf.clip_by_value(mixed_labels, 0, 4), tf.uint8) # No rounding needed if loss handles floats

    return mixed_images, mixed_labels

def apply_augmentation(dataset):
    """Apply batch-level augmentations (CutMix, MixUp) to an already batched dataset."""

    # Assume dataset is already batched as it comes from data_loader via train.py
    print("Applying batch-level augmentations (CutMix, MixUp)...")

    # Enable tf.data optimizations
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.apply_default_optimizations = True
    options.deterministic = False  # Allow non-deterministic ops for speed

    parallel_calls = tf.data.AUTOTUNE

    # Apply batch-level augmentations
    augmented_ds = dataset.with_options(options).map(
        lambda x, y: apply_cutmix(x, y, alpha=0.3, prob=0.5),
        num_parallel_calls=parallel_calls
    )
    augmented_ds = augmented_ds.map(
        lambda x, y: apply_mixup(x, y, alpha=0.3, prob=0.5),
        num_parallel_calls=parallel_calls
    )

    # Prefetch for performance
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
    augmented_ds = apply_augmentation(test_dataset)

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
