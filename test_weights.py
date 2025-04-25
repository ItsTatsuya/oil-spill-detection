#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from model import OilSpillSegformer

def test_weight_loading():
    """
    Test weight loading for the OilSpillSegformer model without training.
    """
    # Set up mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision policy set to mixed_float16")

    # Define the model parameters
    input_shape = (384, 384, 1)
    num_classes = 5

    # Path to pretrained weights
    pretrained_weights_path = 'pretrained_weights/segformer_b2_pretrain.weights.h5'
    if os.path.exists(pretrained_weights_path):
        print(f"Found pretrained weights at {pretrained_weights_path}")
    else:
        print(f"Warning: Pretrained weights not found at {pretrained_weights_path}")
        pretrained_weights_path = None

    # Create the model
    print("\nCreating OilSpillSegformer model...")
    model = OilSpillSegformer(
        input_shape=input_shape,
        num_classes=num_classes,
        use_cbam=True,
        pretrained_weights=pretrained_weights_path
    )

    # Check if weights were loaded
    print("\nVerifying weights were loaded correctly...")
    total_layers = 0
    non_zero_layers = 0
    total_params = 0
    non_zero_params = 0

    for layer in model.layers:
        if layer.weights:
            total_layers += 1
            has_non_zero = False

            for w in layer.weights:
                total_params += np.prod(w.shape)
                # Calculate percentage of non-zero values
                w_np = w.numpy()
                non_zeros = np.count_nonzero(w_np)
                non_zero_params += non_zeros

                if non_zeros > 0:
                    has_non_zero = True

                print(f"Layer {layer.name} weights shape {w.shape}: {non_zeros}/{np.prod(w.shape)} non-zero values ({non_zeros/np.prod(w.shape)*100:.2f}%)")

            if has_non_zero:
                non_zero_layers += 1

    print(f"\nTotal layers with weights: {total_layers}")
    print(f"Layers with non-zero weights: {non_zero_layers}")
    print(f"Total parameters: {total_params:,}")
    print(f"Non-zero parameters: {non_zero_params:,} ({non_zero_params/total_params*100:.2f}%)")

    # Test with a dummy input
    print("\nTesting inference with random input...")
    dummy_input = np.random.random((1, *input_shape)).astype(np.float32)
    output = model.predict(dummy_input, verbose=1)

    print(f"Output shape: {output.shape}")
    print(f"Output range: Min={output.min():.4f}, Max={output.max():.4f}")
    print(f"Output contains NaN values: {np.isnan(output).any()}")

    # Check output classes distribution
    classes = np.argmax(output, axis=-1)
    unique_classes, counts = np.unique(classes, return_counts=True)
    print("\nOutput class distribution:")
    for cls, count in zip(unique_classes, counts):
        print(f"  Class {cls}: {count} pixels ({count/(output.shape[1]*output.shape[2])*100:.2f}%)")

if __name__ == "__main__":
    test_weight_loading()
