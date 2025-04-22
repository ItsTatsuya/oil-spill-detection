import numpy as np
import os
import tensorflow as tf

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
    comparisons = [tf.equal(x, v) for v in values]
    return tf.reduce_any(tf.stack(comparisons, axis=0), axis=0)

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
    if interpolation.upper() == "BILINEAR":
        rotated_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(image, 0),
            transforms=transforms,
            output_shape=[height, width],
            interpolation="BILINEAR",
            fill_mode="CONSTANT",
            fill_value=0.0
        )
    else:
        rotated_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(image, 0),
            transforms=transforms,
            output_shape=[height, width],
            interpolation="NEAREST",
            fill_mode="CONSTANT",
            fill_value=0.0
        )
    return tf.squeeze(rotated_image, 0)

@tf.function
def elastic_deform(image, label, alpha=15.0, sigma=3.0, grid_size=8):
    tf.debugging.assert_shapes([(image, ('H', 'W', 1)), (label, ('H', 'W', 1))])
    height = 384
    width = 384
    x_grid = tf.cast(tf.range(0, width, grid_size), tf.float32)
    y_grid = tf.cast(tf.range(0, height, grid_size), tf.float32)
    grid_height = tf.shape(y_grid)[0]
    grid_width = tf.shape(x_grid)[0]
    dx_coarse = tf.random.normal([grid_height, grid_width], mean=0.0, stddev=sigma) * alpha
    dy_coarse = tf.random.normal([grid_height, grid_width], mean=0.0, stddev=sigma) * alpha
    dx = tf.image.resize(tf.expand_dims(dx_coarse, -1), [height, width], method='nearest')[:,:,0]
    dy = tf.image.resize(tf.expand_dims(dy_coarse, -1), [height, width], method='nearest')[:,:,0]
    y_indices, x_indices = tf.meshgrid(tf.range(height, dtype=tf.float32), tf.range(width, dtype=tf.float32), indexing='ij')
    x_displaced = tf.clip_by_value(x_indices + dx, 0, width - 1)
    y_displaced = tf.clip_by_value(y_indices + dy, 0, height - 1)
    x0 = tf.cast(tf.floor(x_displaced), tf.int32)
    x1 = tf.minimum(x0 + 1, width - 1)
    y0 = tf.cast(tf.floor(y_displaced), tf.int32)
    y1 = tf.minimum(y0 + 1, height - 1)
    x_weight = tf.expand_dims(x_displaced - tf.cast(x0, tf.float32), -1)
    y_weight = tf.expand_dims(y_displaced - tf.cast(y0, tf.float32), -1)

    def gather_pixel(image, y, x):
        indices = tf.stack([y, x], axis=-1)
        return tf.gather_nd(image, indices)

    top_left = gather_pixel(image, y0, x0)
    top_right = gather_pixel(image, y0, x1)
    bottom_left = gather_pixel(image, y1, x0)
    bottom_right = gather_pixel(image, y1, x1)
    top = top_left * (1.0 - x_weight) + top_right * x_weight
    bottom = bottom_left * (1.0 - x_weight) + bottom_right * x_weight
    image_deformed = top * (1.0 - y_weight) + bottom * y_weight
    y_nearest = tf.cast(tf.round(y_displaced), tf.int32)
    x_nearest = tf.cast(tf.round(x_displaced), tf.int32)
    y_nearest = tf.clip_by_value(y_nearest, 0, height - 1)
    x_nearest = tf.clip_by_value(x_nearest, 0, width - 1)
    indices = tf.stack([y_nearest, x_nearest], axis=-1)
    label_deformed = tf.gather_nd(label, indices)
    return image_deformed, label_deformed

@tf.function
def _augment_image_and_label(image, label):
    # Ensure batched input shapes and cast image to float16
    image = tf.ensure_shape(image, [None, 384, 384, 1])
    label = tf.ensure_shape(label, [None, 384, 384, 1])
    image = tf.cast(image, tf.float16)

    # Process each image-label pair in the batch
    def process_single_image_label(img, lbl):
        tf.debugging.assert_shapes([(img, (384, 384, 1)), (lbl, (384, 384, 1))])
        lbl = tf.cast(lbl, tf.uint8)
        num_channels = 1
        # Cast img to float32 for concatenation, then back to float16
        combined = tf.concat([tf.cast(img, tf.float32), tf.cast(lbl, tf.float32)], axis=-1)
        if tf.random.uniform(()) > 0.5:
            combined = tf.image.flip_left_right(combined)
        if tf.random.uniform(()) > 0.5:
            angle = tf.random.uniform([], minval=-0.5236, maxval=0.5236)  # ±30°
            combined = rotate_image(combined, angle, interpolation='nearest')
        if tf.random.uniform(()) > 0.5:
            shear = tf.random.uniform([], minval=-0.1, maxval=0.1)
            h, w = 384, 384
            transform = [1.0, shear, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            combined = tf.raw_ops.ImageProjectiveTransformV3(
                images=tf.expand_dims(combined, 0),
                transforms=[transform],
                output_shape=[h, w],
                interpolation='NEAREST',
                fill_mode='CONSTANT',
                fill_value=0.0
            )[0]
        if tf.random.uniform(()) > 0.5:
            h, w = 384, 384
            scale = tf.random.uniform([], minval=0.9, maxval=1.0)
            new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
            new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
            combined = tf.image.resize(combined, [new_h, new_w], method='nearest')
            combined = tf.image.resize_with_pad(combined, h, w, method='nearest')
        img = tf.cast(combined[..., :num_channels], tf.float16)
        lbl = tf.cast(combined[..., num_channels:], tf.uint8)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        if tf.random.uniform(()) > 0.5:
            noise = tf.random.normal([384, 384, 1], mean=0.0, stddev=0.01, dtype=tf.float16)
            img = img + noise
        if tf.random.uniform(()) > 0.5:
            speckle = tf.random.normal([384, 384, 1], mean=0.0, stddev=0.02, dtype=tf.float16)
            img = tf.clip_by_value(img + speckle, 0, 1)
        img = tf.clip_by_value(img, 0, 1)
        return img, lbl

    # Apply to each image-label pair in the batch
    image, label = tf.map_fn(
        lambda x: process_single_image_label(x[0], x[1]),
        (image, label),
        fn_output_signature=(tf.float16, tf.uint8)
    )
    return image, label

@tf.function
def apply_cutmix(images, labels, alpha=0.3, prob=0.8):
    if tf.random.uniform(()) > prob:
        return images, labels
    batch_size = tf.shape(images)[0]
    height, width = 384, 384
    int_labels = tf.cast(labels, tf.int32)
    rare_mask = tf.logical_or(tf.equal(int_labels, 1), tf.equal(int_labels, 3))
    rare_indices = tf.where(rare_mask)
    num_rare_pixels = tf.shape(rare_indices)[0]
    if num_rare_pixels == 0:
        return images, labels
    idx = tf.random.uniform([], maxval=num_rare_pixels, dtype=tf.int32)
    y_coord = tf.cast(rare_indices[idx][1], tf.int32)
    x_coord = tf.cast(rare_indices[idx][2], tf.int32)
    lambda_param = tf.random.uniform([], 0, 1)
    cut_ratio = tf.math.sqrt(1.0 - lambda_param)
    cut_h = tf.cast(tf.cast(height, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(width, tf.float32) * cut_ratio, tf.int32)
    x1 = tf.maximum(0, x_coord - cut_w // 2)
    y1 = tf.maximum(0, y_coord - cut_h // 2)
    x2 = tf.minimum(width, x_coord + (cut_w + 1) // 2)
    y2 = tf.minimum(height, y_coord + (cut_h + 1) // 2)
    actual_cut_h = tf.maximum(0, y2 - y1)
    actual_cut_w = tf.maximum(0, x2 - x1)
    patch_mask = tf.zeros([actual_cut_h, actual_cut_w, 1], dtype=tf.float32)
    paddings = tf.maximum(0, tf.convert_to_tensor([[y1, height - y2], [x1, width - x2], [0, 0]]))
    inverse_mask = 1.0 - tf.pad(patch_mask, paddings, constant_values=1.0)
    mask = 1.0 - inverse_mask
    mask_img = tf.cast(mask, images.dtype)
    rand_indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = images * (1.0 - mask_img) + tf.gather(images, rand_indices, axis=0) * mask_img
    mixed_labels = tf.cast(labels, tf.uint8)
    return mixed_images, mixed_labels

@tf.function
def apply_mixup(images, labels, alpha=0.3, prob=0.6):
    if tf.random.uniform(()) > prob:
        return images, labels
    batch_size = tf.shape(images)[0]
    gamma1 = tf.random.gamma([batch_size, 1, 1, 1], alpha=alpha, beta=1.0)
    gamma2 = tf.random.gamma([batch_size, 1, 1, 1], alpha=alpha, beta=1.0)
    lam = gamma1 / (gamma1 + gamma2 + 1e-7)
    lam_img_dtype = tf.cast(lam, images.dtype)
    rand_indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam_img_dtype * images + (1.0 - lam_img_dtype) * tf.gather(images, rand_indices, axis=0)
    mixed_labels = tf.cast(labels, tf.uint8)
    return mixed_images, mixed_labels

@tf.function
def apply_augmentation(dataset):
    print("Applying batch-level augmentations (CutMix, MixUp)...")
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.apply_default_optimizations = True
    options.deterministic = False
    dataset = dataset.with_options(options)
    dataset = dataset.map(_augment_image_and_label, num_parallel_calls=2)
    dataset = dataset.map(lambda x, y: apply_cutmix(x, y, alpha=0.3, prob=0.8), num_parallel_calls=2)
    dataset = dataset.map(lambda x, y: apply_mixup(x, y, alpha=0.3, prob=0.6), num_parallel_calls=2)
    dataset = dataset.prefetch(2)
    return dataset

if __name__ == "__main__":
    from data_loader import load_dataset
    import matplotlib.pyplot as plt
    policy = tf.keras.mixed_precision.global_policy()
    print(f"Mixed precision policy: {policy.name}")
    compute_dtype = policy.compute_dtype
    variable_dtype = policy.variable_dtype
    print(f"Compute dtype: {compute_dtype}, Variable dtype: {variable_dtype}")
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
    print("\nTesting augmentation pipeline...")
    augmented_ds = apply_augmentation(test_dataset)
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
        import traceback
        print(f"Error in augmentation pipeline: {e}")
        print(f"Error details: {traceback.format_exc()}")
