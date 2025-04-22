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


def CBAM(channels, reduction_ratio=16, dtype=None):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel and spatial attention mechanisms.
    """
    def apply(inputs):
        # Channel Attention
        avg_pool = layers.GlobalAveragePooling2D(keepdims=True, dtype=dtype)(inputs)
        max_pool = layers.GlobalMaxPooling2D(keepdims=True, dtype=dtype)(inputs)

        avg_pool = layers.Conv2D(filters=channels // reduction_ratio, kernel_size=1,
                                 use_bias=True, dtype=dtype)(avg_pool)
        avg_pool = layers.ReLU(dtype=dtype)(avg_pool)
        avg_pool = layers.Conv2D(filters=channels, kernel_size=1,
                                 use_bias=True, dtype=dtype)(avg_pool)

        max_pool = layers.Conv2D(filters=channels // reduction_ratio, kernel_size=1,
                                 use_bias=True, dtype=dtype)(max_pool)
        max_pool = layers.ReLU(dtype=dtype)(max_pool)
        max_pool = layers.Conv2D(filters=channels, kernel_size=1,
                                 use_bias=True, dtype=dtype)(max_pool)

        channel_attention = layers.Add(dtype=dtype)([avg_pool, max_pool])
        channel_attention = layers.Activation('sigmoid', dtype=dtype)(channel_attention)

        # Apply channel attention
        channel_refined = layers.Multiply(dtype=dtype)([inputs, channel_attention])

        # Spatial Attention
        avg_spatial = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
                                   dtype=dtype)(channel_refined)
        max_spatial = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True),
                                   dtype=dtype)(channel_refined)
        spatial_concat = layers.Concatenate(axis=-1, dtype=dtype)([avg_spatial, max_spatial])

        spatial_attention = layers.Conv2D(filters=1, kernel_size=7, padding='same',
                                         use_bias=False, dtype=dtype)(spatial_concat)
        spatial_attention = layers.Activation('sigmoid', dtype=dtype)(spatial_attention)

        # Apply spatial attention
        output = layers.Multiply(dtype=dtype)([channel_refined, spatial_attention])

        return output

    return apply


def ASPP(filters=256, dtype=None):
    def apply(inputs):
        # Ensure inputs have consistent dtype if specified
        if dtype:
            # Use Lambda layer instead of direct tf.cast for KerasTensor
            inputs = layers.Lambda(lambda x: tf.cast(x, dtype), dtype=dtype, name='aspp_inputs_cast')(inputs)

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
        branch5 = layers.UpSampling2D(
            size=(input_shape[1], input_shape[2]),
            interpolation='bilinear',
            dtype=dtype
        )(branch5)

        # Concatenate all branches with explicit dtype
        concat = layers.Concatenate(name='aspp_concat', dtype=dtype)([branch1, branch2, branch3, branch4, branch5])

        # Final 1x1 convolution
        output = ConvBlock(filters, 1, name='aspp_output', dtype=dtype)(concat)

        # Ensure output has consistent dtype using Lambda layer instead of direct tf.cast
        if dtype:
            output = layers.Lambda(lambda x: tf.cast(x, dtype), dtype=dtype, name='aspp_output_cast')(output)

        return output

    return apply


def Decoder(filters=256, num_classes=5, output_size=(320, 320), dtype=None):
    def apply(inputs, encoder_features):
        # Get dimensions for proper upsampling
        input_shape = tf.keras.backend.int_shape(inputs)
        encoder_shape = tf.keras.backend.int_shape(encoder_features)

        # Ensure dtype is consistent if specified
        if dtype:
            # Use Lambda layers to properly cast tensors in Keras functional API
            inputs = layers.Lambda(lambda x: tf.cast(x, dtype), dtype=dtype, name='decoder_input_cast')(inputs)
            encoder_features = layers.Lambda(lambda x: tf.cast(x, dtype), dtype=dtype, name='decoder_encoder_cast')(encoder_features)

        # Upsample ASPP output to match encoder features with explicit casting
        x = layers.UpSampling2D(
            size=(encoder_shape[1] // input_shape[1], encoder_shape[2] // input_shape[2]),
            interpolation='bilinear',
            name='decoder_upsample_to_encoder',
            dtype=dtype
        )(inputs)

        # Process encoder features
        encoder_features_processed = ConvBlock(48, 1, name='decoder_encoder_features', dtype=dtype)(encoder_features)

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

        # Ensure final output has consistent dtype using Lambda layer
        if dtype:
            x = layers.Lambda(lambda x: tf.cast(x, dtype), dtype=dtype, name='decoder_output_cast')(x)

        return x

    return apply


def DeepLabv3Plus(input_shape=(320, 320, 1), num_classes=5, output_stride=16):
    """
    Implementation of DeepLabV3+ with EfficientNetV2-M backbone.
    Modified to handle grayscale SAR images and incorporate CBAM attention.

    Args:
        input_shape: Input image dimensions (height, width, channels)
        num_classes: Number of output classes
        output_stride: Output stride for ASPP dilation rates

    Returns:
        Keras Model instance of DeepLabV3+
    """
    # Use fixed compute dtype to avoid mixing precision during graph execution
    # Use float32 throughout the model for consistency with XLA compiler
    compute_dtype = 'float32'
    print(f"Using fixed compute dtype: {compute_dtype} for model construction to avoid mixed precision errors")

    # Input layer with explicit dtype
    inputs = layers.Input(shape=input_shape, dtype=compute_dtype, name='input_image')

    # Handle grayscale to RGB conversion with learnable 1x1 convolution
    if input_shape[-1] == 1:
        print("Converting 1-channel grayscale input to 3-channel using learnable 1x1 convolution")
        # Use standard Conv2D with proper initialization for 1->3 channel mapping
        x = layers.Conv2D(
            filters=3,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',  # Use He initialization for better gradient flow
            dtype=compute_dtype,
            name='input_to_rgb'
        )(inputs)
    else:
        x = inputs

    # EfficientNetV2-M backbone WITHOUT pre-loaded weights (we'll load them manually)
    base_model = applications.EfficientNetV2M(
        include_top=False,
        weights=None,  # Don't load weights automatically
        input_tensor=x  # Use the converted 3-channel input
    )

    # Manually load weights to handle the layer count mismatch
    if input_shape[-1] == 1:
        print("Loading pre-trained ImageNet weights correctly")
        # Create a temporary model to get the pre-trained weights
        temp_model = applications.EfficientNetV2M(
            include_top=False,
            weights='imagenet',
            input_shape=(input_shape[0], input_shape[1], 3)
        )

        # Copy weights for EfficientNet layers only, not our custom layer
        for i in range(len(base_model.layers)):
            layer = base_model.layers[i]
            # Skip the input layer and our custom 1x1 conv layer
            if 'input_to_rgb' not in layer.name and 'input_' not in layer.name:
                try:
                    # Find the corresponding layer in temp_model by name
                    temp_layer = temp_model.get_layer(layer.name)
                    layer.set_weights(temp_layer.get_weights())
                except ValueError:
                    print(f"Skipping weight transfer for layer: {layer.name}")

        # Free memory
        del temp_model
    else:
        # If using 3-channel input, we can directly load pre-trained weights
        print("Loading pre-trained ImageNet weights")
        temp_model = applications.EfficientNetV2M(
            include_top=False,
            weights='imagenet',
            input_shape=(input_shape[0], input_shape[1], 3)
        )

        # Copy weights for all layers except input
        for i in range(len(base_model.layers)):
            layer = base_model.layers[i]
            if 'input_' not in layer.name:
                try:
                    temp_layer = temp_model.get_layer(layer.name)
                    layer.set_weights(temp_layer.get_weights())
                except ValueError:
                    print(f"Skipping weight transfer for layer: {layer.name}")

        del temp_model

    # Get features from the backbone - don't use tf.cast directly on KerasTensors
    low_level_features = base_model.get_layer('block2e_add').output
    high_level_features = base_model.output

    # Use a Lambda layer to cast if needed - proper way to handle KerasTensors
    if compute_dtype == 'float32':
        # Use Lambda layers for proper handling of KerasTensors in the model
        low_level_features = layers.Lambda(lambda x: x, dtype=compute_dtype, name='low_level_cast')(low_level_features)
        high_level_features = layers.Lambda(lambda x: x, dtype=compute_dtype, name='high_level_cast')(high_level_features)

    # Apply ASPP to high-level features with consistent dtype
    aspp_output = ASPP(256, dtype=compute_dtype)(high_level_features)

    # Apply CBAM attention after ASPP to focus on ships
    aspp_output = CBAM(channels=tf.keras.backend.int_shape(aspp_output)[-1], dtype=compute_dtype)(aspp_output)

    # Apply decoder to combine features and get final predictions with consistent dtype
    output = Decoder(256, num_classes, output_size=(input_shape[0], input_shape[1]), dtype=compute_dtype)(aspp_output, low_level_features)

    # Use Lambda layer for final output instead of tf.cast directly
    output = layers.Lambda(lambda x: x, dtype=compute_dtype, name='output_cast')(output)

    # Create model
    model = models.Model(inputs=inputs, outputs=output, name='DeepLabv3Plus_EfficientNetV2M')

    # Use model._set_dtype_policy if available, otherwise use attribute directly
    try:
        mixed_precision_policy = tf.keras.mixed_precision.Policy(compute_dtype)
        model._set_dtype_policy(mixed_precision_policy)
    except AttributeError:
        print("Using direct attribute setting for dtype policy")
        model._dtype_policy = tf.keras.mixed_precision.Policy(compute_dtype)

    return model


if __name__ == "__main__":
    # Enable mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision policy set to mixed_float16")

    try:
        # Create a test model with smaller size to ensure it loads quickly
        print("Creating DeepLabv3+ model with fixed precision...")
        model = DeepLabv3Plus(input_shape=(160, 160, 1))  # Smaller input size for faster testing

        # Print summary to verify dtype of layers
        print("\nModel summary with layer dtypes:")
        model.summary()

        # Print the dtype of key layers to verify fixed precision setup
        print("\nChecking compute_dtype of key layers:")
        for layer in model.layers:
            if hasattr(layer, 'compute_dtype'):
                print(f"Layer {layer.name}: compute_dtype = {layer.compute_dtype}")

        # Test the model with a small sample input
        import numpy as np
        print("\nTesting inference with a small batch...")

        # Create a random input image with batch_size=1
        test_input = np.random.random((1, 160, 160, 1)).astype(np.float32)  # Using float32 for fixed precision
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

        print("\nFixed precision implementation test completed successfully!")

    except Exception as e:
        print(f"Error during model test: {str(e)}")
    finally:
        # Reset to default policy for other code that might run
        tf.keras.mixed_precision.set_global_policy('float32')
        print("Reset precision policy to float32")
