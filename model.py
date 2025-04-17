"""
Oil Spill Detection Model - DeepLabv3+ with EfficientNet-B4 Backbone

This module implements a DeepLabv3+ model for semantic segmentation of oil spill images.
The model uses:
1. EfficientNet-B4 backbone (pre-trained on ImageNet)
2. ASPP with atrous rates [6, 12, 18] and global average pooling (256 filters each)
3. Decoder with three 3x3 convolutions (256 filters each, ReLU activation)

Input: 320x320x3 RGB images
Output: 320x320x5 logits for 5 classes
    0 - Sea Surface
    1 - Oil Spill
    2 - Look-alike
    3 - Ship
    4 - Land
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def ConvBlock(filters, kernel_size, dilation_rate=1, use_bias=False, name=None):
    """
    Convolution block with BatchNorm and ReLU.

    Args:
        filters: Number of output filters
        kernel_size: Size of convolution kernel
        dilation_rate: Dilation rate for atrous convolution
        use_bias: Whether to use bias in the convolution
        name: Optional name for the block

    Returns:
        A sequential model representing the conv block
    """
    def apply(x):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            use_bias=use_bias,
            dilation_rate=dilation_rate,
            name=f"{name}_conv" if name else None
        )(x)
        x = layers.BatchNormalization(
            name=f"{name}_bn" if name else None
        )(x)
        x = layers.ReLU(
            name=f"{name}_relu" if name else None
        )(x)
        return x

    return apply


def ASPP(filters=256):
    """
    Atrous Spatial Pyramid Pooling module.

    Args:
        filters: Number of filters for each branch

    Returns:
        A function that applies ASPP to an input tensor
    """
    def apply(inputs):
        input_shape = tf.keras.backend.int_shape(inputs)

        # 1x1 convolution branch
        branch1 = ConvBlock(filters, 1, name='aspp_branch1')(inputs)

        # 3x3 convolutions with different atrous rates
        branch2 = ConvBlock(filters, 3, dilation_rate=6, name='aspp_branch2')(inputs)
        branch3 = ConvBlock(filters, 3, dilation_rate=12, name='aspp_branch3')(inputs)
        branch4 = ConvBlock(filters, 3, dilation_rate=18, name='aspp_branch4')(inputs)

        # Global average pooling branch
        branch5 = layers.GlobalAveragePooling2D()(inputs)
        branch5 = layers.Reshape((1, 1, input_shape[3]))(branch5)
        branch5 = ConvBlock(filters, 1, name='aspp_branch5')(branch5)
        branch5 = layers.UpSampling2D(
            size=(input_shape[1], input_shape[2]),
            interpolation='bilinear'
        )(branch5)

        # Concatenate all branches
        concat = layers.Concatenate(name='aspp_concat')([branch1, branch2, branch3, branch4, branch5])

        # Final 1x1 convolution
        output = ConvBlock(filters, 1, name='aspp_output')(concat)

        return output

    return apply


def Decoder(filters=256, num_classes=5, output_size=(320, 320)):
    """
    DeepLabv3+ decoder with three 3x3 convolutions.

    Args:
        filters: Number of filters for each convolution
        num_classes: Number of output classes
        output_size: Final output size (height, width)

    Returns:
        A function that applies the decoder to encoder outputs
    """
    def apply(inputs, encoder_features):
        # Get dimensions for proper upsampling
        input_shape = tf.keras.backend.int_shape(inputs)
        encoder_shape = tf.keras.backend.int_shape(encoder_features)

        # Upsample ASPP output to match encoder features
        x = layers.UpSampling2D(
            size=(encoder_shape[1] // input_shape[1], encoder_shape[2] // input_shape[2]),
            interpolation='bilinear',
            name='decoder_upsample_to_encoder'
        )(inputs)

        # Process encoder features with 1x1 convolution
        encoder_features = ConvBlock(48, 1, name='decoder_encoder_features')(encoder_features)

        # Concatenate upsampled features with encoder features
        x = layers.Concatenate(name='decoder_concat')([x, encoder_features])

        # Apply three 3x3 convolutions as requested (instead of two in the original DeepLabv3+)
        x = ConvBlock(filters, 3, name='decoder_conv1')(x)
        x = ConvBlock(filters, 3, name='decoder_conv2')(x)
        x = ConvBlock(filters, 3, name='decoder_conv3')(x)

        # Final convolution to get logits
        x = layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            padding='same',
            name='decoder_output'
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
                name='decoder_final_upsample'
            )(x)

        # Add a final resize operation to ensure exact output size
        x = layers.Lambda(lambda x: tf.image.resize(x, output_size), name='exact_size_resize')(x)

        return x

    return apply


def DeepLabv3Plus(input_shape=(320, 320, 3), num_classes=5, output_stride=16):
    """
    DeepLabv3+ model with EfficientNet-B4 backbone.

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        output_stride: Output stride of the backbone (8 or 16)

    Returns:
        A tf.keras.Model instance of DeepLabv3+ model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input_image')

    # EfficientNet-B4 backbone with specific output stride
    base_model = applications.EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    # Extract features from different levels of the backbone
    # For EfficientNetB4:
    # - Low-level features from block3b (stride 4)
    # - High-level features from the last block (stride 16)
    low_level_features = base_model.get_layer('block3b_add').output
    high_level_features = base_model.output

    # Apply ASPP to high-level features
    aspp_output = ASPP(256)(high_level_features)

    # Apply decoder to combine features and get final predictions
    output = Decoder(256, num_classes, output_size=(input_shape[0], input_shape[1]))(aspp_output, low_level_features)

    # Create model
    model = models.Model(inputs=inputs, outputs=output, name='DeepLabv3Plus_EfficientNetB4')

    return model


if __name__ == "__main__":
    # Create a test model and print summary
    model = DeepLabv3Plus()
    model.summary()

    # Plot the model architecture
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file='deeplabv3plus_model.png', show_shapes=True, show_layer_names=True)
        print("Model architecture saved to deeplabv3plus_model.png")
    except ImportError:
        print("Could not import plot_model. Model architecture diagram not saved.")

    # Test the model with a sample input
    import numpy as np

    # Create a random input image
    test_input = np.random.random((1, 320, 320, 3)).astype(np.float32)

    # Get model prediction
    test_output = model.predict(test_input)

    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")
    print("Model created successfully!")
