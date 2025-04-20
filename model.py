def silent_tf_import():
    import sys
    import os
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

from tensorflow.keras import layers, models, applications # type: ignore


def ConvBlock(filters, kernel_size, dilation_rate=1, use_bias=False, name=None, dtype=None):
    def apply(x):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            use_bias=use_bias,
            dilation_rate=dilation_rate,
            name=f"{name}_conv" if name else None,
            dtype=dtype
        )(x)
        x = layers.BatchNormalization(
            name=f"{name}_bn" if name else None,
            dtype=dtype
        )(x)
        # Ensure ReLU operation maintains the input dtype
        x = layers.ReLU(
            name=f"{name}_relu" if name else None,
            dtype=dtype
        )(x)
        return x

    return apply


def ASPP(filters=256, dtype=None):
    def apply(inputs):
        input_shape = tf.keras.backend.int_shape(inputs)

        # 1x1 convolution branch
        branch1 = ConvBlock(filters, 1, name='aspp_branch1', dtype=dtype)(inputs)

        # 3x3 convolutions with different atrous rates
        branch2 = ConvBlock(filters, 3, dilation_rate=6, name='aspp_branch2', dtype=dtype)(inputs)
        branch3 = ConvBlock(filters, 3, dilation_rate=12, name='aspp_branch3', dtype=dtype)(inputs)
        branch4 = ConvBlock(filters, 3, dilation_rate=18, name='aspp_branch4', dtype=dtype)(inputs)

        # Global average pooling branch with dtype specified
        branch5 = layers.GlobalAveragePooling2D(keepdims=True, dtype=dtype)(inputs)
        branch5 = ConvBlock(filters, 1, name='aspp_branch5', dtype=dtype)(branch5)
        branch5 = ConvBlock(filters, 1, name='aspp_branch5_up', dtype=dtype)(branch5)
        branch5 = layers.UpSampling2D(
            size=(input_shape[1], input_shape[2]),
            interpolation='bilinear',
            dtype=dtype
        )(branch5)

        # Concatenate all branches with explicit dtype
        concat = layers.Concatenate(name='aspp_concat', dtype=dtype)([branch1, branch2, branch3, branch4, branch5])

        # Final 1x1 convolution
        output = ConvBlock(filters, 1, name='aspp_output', dtype=dtype)(concat)

        return output

    return apply


def Decoder(filters=256, num_classes=5, output_size=(320, 320), dtype=None):
    def apply(inputs, encoder_features):
        # Get dimensions for proper upsampling
        input_shape = tf.keras.backend.int_shape(inputs)
        encoder_shape = tf.keras.backend.int_shape(encoder_features)

        # Upsample ASPP output to match encoder features with explicit casting
        x = layers.UpSampling2D(
            size=(encoder_shape[1] // input_shape[1], encoder_shape[2] // input_shape[2]),
            interpolation='bilinear',
            name='decoder_upsample_to_encoder',
            dtype=dtype
        )(inputs)

        encoder_features_processed = layers.Lambda(
            lambda t: tf.cast(t, dtype) if dtype else t,
            name='encoder_features_cast'
        )(encoder_features)

        encoder_features_processed = ConvBlock(48, 1, name='decoder_encoder_features', dtype=dtype)(encoder_features_processed)

        # Concatenate upsampled features with encoder features
        x = layers.Concatenate(name='decoder_concat', dtype=dtype)([x, encoder_features_processed])

        # Apply three 3x3 convolutions with consistent dtype
        x = ConvBlock(filters, 3, name='decoder_conv1', dtype=dtype)(x)
        x = ConvBlock(filters, 3, name='decoder_conv2', dtype=dtype)(x)
        x = ConvBlock(filters, 3, name='decoder_conv3', dtype=dtype)(x)

        # Final convolution to get logits with consistent dtype
        x = layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            padding='same',
            name='decoder_output',
            dtype=dtype
        )(x)

        # Calculate upsampling size to reach the target output size
        current_shape = tf.keras.backend.int_shape(x)
        upsampling_factor_h = output_size[0] // current_shape[1]
        upsampling_factor_w = output_size[1] // current_shape[2]

        if upsampling_factor_h > 1 or upsampling_factor_w > 1:
            # Upsample to original resolution
            x = layers.UpSampling2D(
                size=(upsampling_factor_h, upsampling_factor_w),
                interpolation='bilinear',
                name='decoder_final_upsample',
                dtype=dtype
            )(x)

        # Cast the final output to float32 for training stability using Lambda layer
        x = layers.Lambda(lambda t: tf.cast(t, tf.float32), name='final_output_cast')(x)

        return x

    return apply


def DeepLabv3Plus(input_shape=(320, 320, 3), num_classes=5, output_stride=16):
    # Get the global policy dtype - respect the global policy
    policy = tf.keras.mixed_precision.global_policy()
    compute_dtype = policy.compute_dtype
    print(f"Building DeepLabv3+ model with compute dtype: {compute_dtype}")

    # Input layer
    inputs = layers.Input(shape=input_shape, name='input_image')

    # EfficientNet-B4 backbone with specific output stride
    base_model = applications.EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    try:
        low_level_features = base_model.get_layer('block3b_add').output
    except ValueError as e:
        print("Available layers in EfficientNetB4:")
        for l in base_model.layers:
            print(l.name)
        raise e

    high_level_features = base_model.output

    # Apply ASPP to high-level features with consistent dtype
    aspp_output = ASPP(256, dtype=compute_dtype)(high_level_features)

    # Apply decoder to combine features and get final predictions with consistent dtype
    output = Decoder(256, num_classes, output_size=(input_shape[0], input_shape[1]), dtype=compute_dtype)(aspp_output, low_level_features)

    # Create model
    model = models.Model(inputs=inputs, outputs=output, name='DeepLabv3Plus_EfficientNetB4')

    return model


if __name__ == "__main__":
    # Enable mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision policy set to mixed_float16")

    try:
        # Create a test model with smaller size to ensure it loads quickly
        print("Creating DeepLabv3+ model with mixed precision...")
        model = DeepLabv3Plus(input_shape=(160, 160, 3))  # Smaller input size for faster testing

        # Print summary to verify dtype of layers
        print("\nModel summary with layer dtypes:")
        model.summary()

        # Print the dtype of key layers to verify mixed precision setup
        print("\nChecking compute_dtype of key layers:")
        for layer in model.layers:
            if hasattr(layer, 'compute_dtype'):
                print(f"Layer {layer.name}: compute_dtype = {layer.compute_dtype}")

        # Test the model with a small sample input
        import numpy as np
        print("\nTesting inference with a small batch...")

        # Create a random input image with batch_size=1
        test_input = np.random.random((1, 160, 160, 3)).astype(np.float16)  # Using float16 for mixed precision
        print(f"Test input shape: {test_input.shape}, dtype: {test_input.dtype}")

        # Get model prediction
        test_output = model.predict(test_input, verbose=1)
        print(f"Test output shape: {test_output.shape}, dtype: {test_output.dtype}")

        # Verify no NaN values in output
        if np.isnan(test_output).any():
            print("WARNING: Output contains NaN values!")
        else:
            print("Output validation: No NaN values detected")

        # Check memory usage
        print("\nGPU memory usage during inference:")
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        memory_used = result.stdout.strip()
        print(f"Memory used: {memory_used} MiB")

        print("\nMixed precision implementation test completed successfully!")

    except Exception as e:
        print(f"Error during model test: {str(e)}")
    finally:
        # Reset to default policy for other code that might run
        tf.keras.mixed_precision.set_global_policy('float32')
        print("Reset precision policy to float32")
