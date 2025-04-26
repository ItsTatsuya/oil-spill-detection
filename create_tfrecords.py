#!/usr/bin/env python3
"""
Convert oil spill detection dataset to TFRecord format for improved I/O performance.

This script converts JPEG images and PNG labels to TFRecord files.
Usage: python3 create_tfrecords.py
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord_example(image_path, label_path):
    """Create a TFRecord Example from an image and label file."""
    # Read image and label
    image_data = tf.io.read_file(image_path)
    label_data = tf.io.read_file(label_path)

    # Get shape information (will be needed during parsing)
    image = tf.image.decode_jpeg(image_data, channels=1)
    label = tf.image.decode_png(label_data, channels=1)
    height = image.shape[0]
    width = image.shape[1]

    # Create a feature dictionary
    feature = {
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image': _bytes_feature(image_data),
        'label': _bytes_feature(label_data),
        'image_format': _bytes_feature(b'jpeg'),
        'label_format': _bytes_feature(b'png'),
    }

    # Create an Example
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

def convert_dataset_to_tfrecord(data_dir, split, num_shards=10):
    """Convert dataset to TFRecord format with sharding."""
    split_dir = os.path.join(data_dir, split)
    image_dir = os.path.join(split_dir, 'images')
    label_dir = os.path.join(split_dir, 'labels_1D')

    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.png')))

    assert len(image_files) == len(label_files), f"Mismatch: {len(image_files)} images, {len(label_files)} labels"

    # Create output directory
    tfrecord_dir = os.path.join(data_dir, f'{split}_tfrecords')
    os.makedirs(tfrecord_dir, exist_ok=True)

    num_examples = len(image_files)
    examples_per_shard = int(np.ceil(num_examples / num_shards))

    print(f"Converting {split} dataset with {num_examples} examples into {num_shards} shards")

    # Process each shard
    for shard_id in range(num_shards):
        start_idx = shard_id * examples_per_shard
        end_idx = min((shard_id + 1) * examples_per_shard, num_examples)

        # Skip empty shards
        if start_idx >= num_examples:
            break

        shard_size = end_idx - start_idx
        output_filename = os.path.join(
            tfrecord_dir,
            f'{split}-{shard_id:05d}-of-{num_shards:05d}.tfrecord'
        )

        print(f"Writing shard {shard_id+1}/{num_shards} with {shard_size} examples to {output_filename}")

        with tf.io.TFRecordWriter(output_filename) as writer:
            for i in tqdm(range(start_idx, end_idx)):
                example = create_tfrecord_example(image_files[i], label_files[i])
                writer.write(example.SerializeToString())

    print(f"Conversion of {split} dataset complete. TFRecords saved to {tfrecord_dir}")
    return tfrecord_dir

if __name__ == "__main__":
    # Convert both train and test datasets
    data_dir = 'dataset'
    train_tfrecord_dir = convert_dataset_to_tfrecord(data_dir, 'train', num_shards=10)
    test_tfrecord_dir = convert_dataset_to_tfrecord(data_dir, 'test', num_shards=2)

    print("\nDataset conversion complete!")
    print(f"Train TFRecords: {train_tfrecord_dir}")
    print(f"Test TFRecords: {test_tfrecord_dir}")
    print("\nNext, update data_loader.py to use these TFRecord files for faster loading.")
