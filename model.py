import tensorflow as tf
from tensorflow.keras import layers, models, applications # type: ignore

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


def CBAM(filters, reduction_ratio=16, kernel_size=7, dtype=None):
    """
    Convolutional Block Attention Module
    Combines both channel and spatial attention to enhance feature representation
    """
    def apply(x):
        # Channel Attention
        avg_pool = layers.GlobalAveragePooling2D(dtype=dtype)(x)
        max_pool = layers.GlobalMaxPooling2D(dtype=dtype)(x)
        
        avg_pool = layers.Reshape((1, 1, filters), dtype=dtype)(avg_pool)
        max_pool = layers.Reshape((1, 1, filters), dtype=dtype)(max_pool)
        
        shared_dense_1 = layers.Dense(filters // reduction_ratio, 
                                    activation='relu', 
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros',
                                    dtype=dtype)
        
        shared_dense_2 = layers.Dense(filters,
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros',
                                    dtype=dtype)
        
        avg_pool = shared_dense_1(avg_pool)
        max_pool = shared_dense_1(max_pool)
        
        avg_pool = shared_dense_2(avg_pool)
        max_pool = shared_dense_2(max_pool)
        
        channel_attention = layers.Add(dtype=dtype)([avg_pool, max_pool])
        channel_attention = layers.Activation('sigmoid', dtype=dtype)(channel_attention)
        
        # Apply channel attention
        x = layers.Multiply(dtype=dtype)([x, channel_attention])
        
        # Spatial Attention
        avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True), dtype=dtype)(x)
        max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True), dtype=dtype)(x)
        
        spatial_attention = layers.Concatenate(axis=3, dtype=dtype)([avg_pool, max_pool])
        
        spatial_attention = layers.Conv2D(filters=1, 
                                        kernel_size=kernel_size,
                                        strides=1,
                                        padding='same',
                                        activation='sigmoid',
                                        kernel_initializer='he_normal',
                                        use_bias=False,
                                        dtype=dtype)(spatial_attention)
        
        # Apply spatial attention
        x = layers.Multiply(dtype=dtype)([x, spatial_attention])
        
        return x
    
    return apply


def FeaturePyramidAttention(filters=256, dtype=None):
    """
    Feature Pyramid Attention module to enhance multi-scale feature fusion
    """
    def apply(x):
        # Global pooling branch
        global_features = layers.GlobalAveragePooling2D(keepdims=True, dtype=dtype)(x)
        global_features = ConvBlock(filters, 1, name='fpa_global', dtype=dtype)(global_features)
        
        # Get input shape for proper upsampling
        input_shape = tf.keras.backend.int_shape(x)
        
        # Different receptive field branches
        branch1 = ConvBlock(filters, 1, name='fpa_branch1', dtype=dtype)(x)
        
        branch2 = ConvBlock(filters//2, 1, name='fpa_branch2_1', dtype=dtype)(x)
        branch2 = ConvBlock(filters//2, 3, name='fpa_branch2_2', dtype=dtype)(branch2)
        branch2 = ConvBlock(filters, 1, name='fpa_branch2_3', dtype=dtype)(branch2)
        
        branch3 = ConvBlock(filters//2, 1, name='fpa_branch3_1', dtype=dtype)(x)
        branch3 = ConvBlock(filters//2, 5, name='fpa_branch3_2', dtype=dtype)(branch3)
        branch3 = ConvBlock(filters, 1, name='fpa_branch3_3', dtype=dtype)(branch3)
        
        # Upsample global features
        global_features = layers.UpSampling2D(
            size=(input_shape[1], input_shape[2]),
            interpolation='bilinear',
            dtype=dtype
        )(global_features)
        
        # Fuse all features
        fusion = layers.Concatenate(dtype=dtype)([global_features, branch1, branch2, branch3])
        
        # Compress channels
        output = ConvBlock(filters, 1, name='fpa_output', dtype=dtype)(fusion)
        
        # Apply attention
        output = CBAM(filters, dtype=dtype)(output)
        
        return output
    
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


def ShipDetectionBranch(filters=128, dtype=None):
    """Specialized branch for detecting small objects like ships"""
    def apply(low_level_features, mid_level_features=None):
        # Process low-level features for ship detection
        # Low-level features have higher resolution and better for small objects
        x = ConvBlock(filters, 1, name='ship_branch_reduce', dtype=dtype)(low_level_features)

        # Apply atrous convolutions with smaller dilation rates for small objects
        atrous1 = ConvBlock(filters, 3, dilation_rate=2, name='ship_branch_atrous1', dtype=dtype)(x)
        atrous2 = ConvBlock(filters, 3, dilation_rate=4, name='ship_branch_atrous2', dtype=dtype)(x)

        # Concatenate atrous features
        x = layers.Concatenate(name='ship_branch_concat', dtype=dtype)([x, atrous1, atrous2])

        # If mid-level features are provided, incorporate them
        if mid_level_features is not None:
            # Upsample mid-level features to match low-level features
            mid_shape = tf.keras.backend.int_shape(mid_level_features)
            low_shape = tf.keras.backend.int_shape(low_level_features)

            upsampled_mid = layers.UpSampling2D(
                size=(low_shape[1] // mid_shape[1], low_shape[2] // mid_shape[2]),
                interpolation='bilinear',
                name='ship_branch_upsample_mid',
                dtype=dtype
            )(mid_level_features)

            # Reduce channels and concatenate
            upsampled_mid = ConvBlock(filters, 1, name='ship_branch_mid_reduce', dtype=dtype)(upsampled_mid)
            x = layers.Concatenate(name='ship_branch_mid_concat', dtype=dtype)([x, upsampled_mid])

        # Apply additional convolutions
        x = ConvBlock(filters, 3, name='ship_branch_conv1', dtype=dtype)(x)
        x = ConvBlock(filters, 3, name='ship_branch_conv2', dtype=dtype)(x)

        # Apply spatial attention to focus on object-like structures
        attention = layers.Conv2D(
            filters=1,
            kernel_size=1,
            padding='same',
            activation='sigmoid',
            name='ship_branch_attention',
            dtype=dtype
        )(x)

        # Apply attention mechanism
        x = layers.Multiply(name='ship_branch_attention_multiply', dtype=dtype)([x, attention])

        return x

    return apply


def MultiScaleDecoder(filters=256, num_classes=5, output_size=(320, 320), ship_enhanced=True, dtype=None):
    """Enhanced decoder with a specific branch for ship detection"""
    def apply(aspp_output, low_level_features, mid_level_features=None):
        # Process main segmentation path (standard decoder)
        input_shape = tf.keras.backend.int_shape(aspp_output)
        encoder_shape = tf.keras.backend.int_shape(low_level_features)

        # Upsample ASPP output to match encoder features
        x = layers.UpSampling2D(
            size=(encoder_shape[1] // input_shape[1], encoder_shape[2] // input_shape[2]),
            interpolation='bilinear',
            name='decoder_upsample_to_encoder',
            dtype=dtype
        )(aspp_output)

        # Process encoder features
        encoder_features_processed = ConvBlock(48, 1, name='decoder_encoder_features', dtype=dtype)(low_level_features)

        # Concatenate upsampled features with encoder features
        x = layers.Concatenate(name='decoder_concat', dtype=dtype)([x, encoder_features_processed])

        # Main decoder path
        x = ConvBlock(filters, 3, name='decoder_conv1', dtype=dtype)(x)
        x = ConvBlock(filters, 3, name='decoder_conv2', dtype=dtype)(x)
        decoder_features = ConvBlock(filters, 3, name='decoder_conv3', dtype=dtype)(x)

        # Apply Feature Pyramid Attention
        decoder_features = FeaturePyramidAttention(filters, dtype=dtype)(decoder_features)

        # Ship detection specific path if enabled
        if ship_enhanced:
            ship_features = ShipDetectionBranch(filters//2, dtype=dtype)(
                low_level_features,
                mid_level_features
            )

            # Ensure ship_features are at the same resolution as decoder_features
            ship_shape = tf.keras.backend.int_shape(ship_features)
            decoder_shape = tf.keras.backend.int_shape(decoder_features)

            if ship_shape[1] != decoder_shape[1] or ship_shape[2] != decoder_shape[2]:
                ship_features = layers.UpSampling2D(
                    size=(decoder_shape[1] // ship_shape[1], decoder_shape[2] // ship_shape[2]),
                    interpolation='bilinear',
                    name='ship_branch_match_decoder',
                    dtype=dtype
                )(ship_features)

            # Fuse features from both paths
            combined_features = layers.Concatenate(name='combined_features', dtype=dtype)([
                decoder_features, ship_features
            ])

            # Process combined features
            combined_features = ConvBlock(filters, 3, name='combined_conv', dtype=dtype)(combined_features)

            # Final classification layer for standard segmentation
            main_logits = layers.Conv2D(
                filters=num_classes,
                kernel_size=1,
                padding='same',
                name='main_logits',
                dtype=dtype
            )(combined_features)
        else:
            # If ship enhancement is disabled, use standard decoder path
            main_logits = layers.Conv2D(
                filters=num_classes,
                kernel_size=1,
                padding='same',
                name='main_logits',
                dtype=dtype
            )(decoder_features)

        # Calculate upsampling size to reach the target output size
        current_shape = tf.keras.backend.int_shape(main_logits)
        upsampling_factor_h = output_size[0] // current_shape[1]
        upsampling_factor_w = output_size[1] // current_shape[2]

        if upsampling_factor_h > 1 or upsampling_factor_w > 1:
            # Upsample to original resolution
            main_logits = layers.UpSampling2D(
                size=(upsampling_factor_h, upsampling_factor_w),
                interpolation='bilinear',
                name='main_logits_upsample',
                dtype=dtype
            )(main_logits)

        return main_logits

    return apply


def DeepLabv3Plus(input_shape=(320, 320, 3), num_classes=5, output_stride=16, ship_enhanced=True):
    """
    Enhanced DeepLabv3+ model with EfficientNetV2-B3 backbone and attention mechanisms
    for improved oil spill and ship detection
    """
    # Use float32 for stable training
    compute_dtype = tf.float32
    print(f"Building enhanced DeepLabv3+ model with compute dtype: {compute_dtype}")

    # Input layer
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # Load EfficientNetV2-B3 backbone (better performance than B4 with similar compute cost)
    try:
        base_model = applications.EfficientNetV2B3(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        print("Using EfficientNetV2-B3 backbone")
    except AttributeError:
        # Fall back to EfficientNetB4 if V2 is not available
        print("EfficientNetV2-B3 not available, falling back to EfficientNetB4")
        base_model = applications.EfficientNetB4(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
    
    # Extract features at different levels for multi-scale processing
    try:
        # Map appropriate layer names based on the backbone used
        if 'EfficientNetV2' in base_model.name:
            # EfficientNetV2 layer names
            low_level_layer = 'block2c_add'
            mid_level_layer = 'block4c_add'
        else:
            # EfficientNetB4 layer names
            low_level_layer = 'block3b_add'
            mid_level_layer = 'block5f_add'
        
        # Extract feature maps
        low_level_features = base_model.get_layer(low_level_layer).output
        mid_level_features = base_model.get_layer(mid_level_layer).output
        high_level_features = base_model.output
        
        print(f"Feature extraction: Low-level from {low_level_layer}, Mid-level from {mid_level_layer}")
    except ValueError as e:
        # Handle layer name errors
        print(f"Error: Could not find the specified layer in {base_model.name}.")
        print(f"Original error: {str(e)}")
        
        # List some available layers to help with debugging
        print("Available layers (showing first 10):")
        for i, layer in enumerate(base_model.layers[:10]):
            print(f"  - {layer.name}")
        print("  ... and more")
        
        raise ValueError(f"Layer not found in {base_model.name}. Check the layer names.")

    # Apply Feature Pyramid Attention to high-level features
    fpa_output = FeaturePyramidAttention(256, dtype=compute_dtype)(high_level_features)
    
    # Apply ASPP to the FPA output for multi-scale context
    aspp_output = ASPP(256, dtype=compute_dtype)(fpa_output)
    
    # Apply CBAM attention to the ASPP output
    aspp_output = CBAM(256, dtype=compute_dtype)(aspp_output)

    # Apply enhanced multi-scale decoder
    output = MultiScaleDecoder(
        256,
        num_classes,
        output_size=(input_shape[0], input_shape[1]),
        ship_enhanced=ship_enhanced,
        dtype=compute_dtype
    )(aspp_output, low_level_features, mid_level_features)

    # Create model
    model = models.Model(inputs=inputs, outputs=output, name='EnhancedDeepLabv3Plus_EfficientNetV2B3')

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
