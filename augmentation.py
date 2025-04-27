import numpy as np
import tensorflow as tf

@tf.function(reduce_retracing=True)
def rotate_image(image, angle, method='bilinear'):
    orig_shape = tf.shape(image)

    if len(orig_shape) == 3 and orig_shape[2] == 1:
        squeeze_needed = True
        image = tf.squeeze(image, axis=-1)
    else:
        squeeze_needed = False

    image = tf.expand_dims(image, axis=0)

    angle_rad = angle
    image = tf.image.rot90(image, k=tf.cast(tf.round(angle_rad / (np.pi/2)), tf.int32))

    rotated = image[0]

    if squeeze_needed:
        rotated = tf.expand_dims(rotated, axis=-1)

    return rotated

@tf.function(reduce_retracing=True)
def random_flip(image, label, p_horizontal=0.5, p_vertical=0.3):
    if tf.random.uniform(()) < p_horizontal:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    if tf.random.uniform(()) < p_vertical:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)
    return image, label

@tf.function(reduce_retracing=True)
def random_rotation(image, label, max_angle=15.0, p_90deg=0.2):
    max_angle_rad = max_angle * (np.pi / 180.0)
    if tf.random.uniform(()) < p_90deg:
        k = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        label = tf.image.rot90(label, k=k)
    else:
        angle = tf.random.uniform([], minval=-max_angle_rad, maxval=max_angle_rad)
        image = rotate_image(image, angle, method='bilinear')
        label = rotate_image(label, angle, method='nearest')
    return image, label

@tf.function(reduce_retracing=True)
def add_speckle_noise(image, label, mean=0.0, stddev=0.1, p=0.7):
    if tf.random.uniform(()) < p:
        noise = tf.random.normal(tf.shape(image), mean=mean, stddev=stddev, dtype=image.dtype)
        noisy_image = image * (1 + noise)
        noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
        return noisy_image, label
    else:
        return image, label

@tf.function(reduce_retracing=True)
def augment_single_sample(image, label):
    image, label = random_flip(image, label)
    image, label = random_rotation(image, label, max_angle=15.0, p_90deg=0.2)
    image, label = add_speckle_noise(image, label, stddev=0.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return tf.cast(image, tf.float16), tf.cast(label, tf.uint8)

@tf.function(reduce_retracing=True)
def apply_augmentation(dataset, batch_size=2):
    dataset = dataset.unbatch()
    dataset = dataset.map(augment_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
