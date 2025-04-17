"""
Oil Spill Detection Hybrid Loss Function

This module implements a hybrid loss function for semantic segmentation of the oil spill dataset.
The hybrid loss combines:
1. Weighted Cross-Entropy Loss
2. Focal Loss
3. Dice Loss

The input y_true is expected to be uint8 [batch, 320, 320, 1] with values 0-4 representing:
0 - Sea Surface
1 - Oil Spill
2 - Look-alike
3 - Ship
4 - Land

The input y_pred is expected to be logits [batch, height, width, 5].
"""

import tensorflow as tf
import numpy as np


def weighted_cross_entropy_loss(y_true_one_hot, y_pred, class_weights):
    """
    Weighted cross-entropy loss with class weights.

    Args:
        y_true_one_hot: One-hot encoded ground truth, shape [batch, height, width, num_classes]
        y_pred: Predicted logits, shape [batch, height, width, num_classes]
        class_weights: Tensor of shape [num_classes] containing weights for each class

    Returns:
        Weighted cross-entropy loss
    """
    # Apply softmax to get probabilities
    y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)

    # Compute cross-entropy loss
    cross_entropy = -tf.reduce_sum(y_true_one_hot * tf.math.log(tf.clip_by_value(y_pred_softmax, 1e-7, 1.0)), axis=-1)

    # Apply class weights to each pixel based on its true class
    weights = tf.reduce_sum(class_weights * y_true_one_hot, axis=-1)

    # Weight the cross-entropy loss
    weighted_ce = weights * cross_entropy

    # Return the mean loss
    return tf.reduce_mean(weighted_ce)


def focal_loss(y_true_one_hot, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss for addressing class imbalance.

    Args:
        y_true_one_hot: One-hot encoded ground truth, shape [batch, height, width, num_classes]
        y_pred: Predicted logits, shape [batch, height, width, num_classes]
        alpha: Weighting factor to balance positive and negative examples
        gamma: Focusing parameter to focus on hard examples

    Returns:
        Focal loss
    """
    # Apply softmax to get probabilities
    y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)

    # Clip values for numerical stability
    y_pred_softmax = tf.clip_by_value(y_pred_softmax, 1e-7, 1.0)

    # Calculate focal weight
    focal_weight = tf.pow(1.0 - y_pred_softmax, gamma)

    # Compute the binary cross-entropy
    ce = -y_true_one_hot * tf.math.log(y_pred_softmax) * alpha

    # Apply focal weighting
    focal = focal_weight * ce

    # Sum over classes and compute mean over batch, height, width
    return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))


def dice_loss(y_true_one_hot, y_pred, smooth=1.0):
    """
    Dice loss for measuring overlap between predictions and ground truth.

    Args:
        y_true_one_hot: One-hot encoded ground truth, shape [batch, height, width, num_classes]
        y_pred: Predicted logits, shape [batch, height, width, num_classes]
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice loss
    """
    # Apply softmax to get probabilities
    y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_one_hot * y_pred_softmax, axis=[1, 2])
    union = tf.reduce_sum(y_true_one_hot + y_pred_softmax, axis=[1, 2])

    # Calculate Dice coefficient for each class and each image in batch
    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Average over classes and batch
    dice_avg = tf.reduce_mean(dice)

    # Return Dice loss
    return 1.0 - dice_avg


def hybrid_loss(y_true, y_pred):
    """
    Hybrid loss function combining weighted cross-entropy, focal loss and dice loss.

    Args:
        y_true: Ground truth labels, shape [batch, height, width, 1], values 0-4
        y_pred: Predicted logits, shape [batch, height, width, 5]

    Returns:
        Scalar loss value
    """
    # Get shapes
    y_true_shape = tf.shape(y_true)
    y_pred_shape = tf.shape(y_pred)

    # Use tf.cond instead of Python if statement for shape checking
    # Reshape y_pred if dimensions don't match
    y_pred = tf.cond(
        tf.logical_or(
            tf.not_equal(y_true_shape[1], y_pred_shape[1]),
            tf.not_equal(y_true_shape[2], y_pred_shape[2])
        ),
        lambda: tf.image.resize(y_pred, [y_true_shape[1], y_true_shape[2]], method='bilinear'),
        lambda: y_pred
    )

    # Convert y_true to one-hot encoding
    num_classes = 5
    y_true = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(tf.squeeze(y_true, axis=-1), depth=num_classes)

    # Define class weights based on provided pixel counts
    # (sea surface=797.7M, oil spill=9.1M, look-alike=50.4M, ship=0.3M, land=45.7M)
    pixel_counts = tf.constant([797.7e6, 9.1e6, 50.4e6, 0.3e6, 45.7e6], dtype=tf.float32)

    # Normalize the weights (inverse frequency)
    class_weights = 1.0 / (pixel_counts + 1e-7)
    class_weights = class_weights / tf.reduce_sum(class_weights)

    # Calculate individual losses
    ce_loss = weighted_cross_entropy_loss(y_true_one_hot, y_pred, class_weights)
    f_loss = focal_loss(y_true_one_hot, y_pred, alpha=0.25, gamma=2.0)
    d_loss = dice_loss(y_true_one_hot, y_pred)

    # Combine losses with specified weights (0.4, 0.3, 0.3)
    combined_loss = 0.4 * ce_loss + 0.3 * f_loss + 0.3 * d_loss

    return combined_loss


if __name__ == "__main__":
    # Test the loss function with some dummy data
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

    # Generate dummy data
    batch_size = 2
    height, width = 320, 320
    num_classes = 5

    # Test with different sizes to verify resizing works
    print("Testing with matching dimensions...")
    # Create random predictions (logits)
    y_pred = tf.random.normal([batch_size, height, width, num_classes])
    # Create random ground truth labels (values 0-4)
    y_true = tf.random.uniform([batch_size, height, width, 1], minval=0, maxval=5, dtype=tf.int32)
    # Calculate the hybrid loss
    loss_value = hybrid_loss(y_true, y_pred)
    print("Hybrid Loss Value:", loss_value.numpy())

    print("\nTesting with mismatched dimensions...")
    # Create random predictions with smaller dimensions (160x160)
    y_pred_small = tf.random.normal([batch_size, height//2, width//2, num_classes])
    # Calculate the hybrid loss with dimension mismatch
    loss_value_small = hybrid_loss(y_true, y_pred_small)
    print("Hybrid Loss Value with resizing:", loss_value_small.numpy())

    # Test with different class distributions to verify class weighting
    print("\nTesting with different class distributions:")
    # Create ground truth with mostly class 0 (sea surface)
    y_true_sea = tf.zeros([batch_size, height, width, 1], dtype=tf.int32)
    loss_sea = hybrid_loss(y_true_sea, y_pred)
    print("Loss with mostly sea surface:", loss_sea.numpy())

    # Create ground truth with mostly class 3 (ship)
    y_true_ship = tf.ones([batch_size, height, width, 1], dtype=tf.int32) * 3
    loss_ship = hybrid_loss(y_true_ship, y_pred)
    print("Loss with mostly ship:", loss_ship.numpy())
