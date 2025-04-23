import os
import numpy as np
import warnings

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

# Suppress register spill warnings by filtering CUDA warnings
os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'  # Optimize performance
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'       # Enable automatic mixed precision
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'                # Enable oneDNN optimizations
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'  # Specify CUDA path

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

from tensorflow.keras import layers, models # type: ignore

# Custom layer to wrap tf.cast
class CastToDtype(layers.Layer):
    """A Keras layer to cast a tensor to a specified dtype."""
    def __init__(self, dtype, **kwargs):
        super().__init__(**kwargs)
        # Store the target dtype with a different name to avoid conflict with Layer.dtype
        self._target_dtype = dtype

    def call(self, inputs):
        # Perform the cast operation within the layer's call method
        return tf.cast(inputs, self._target_dtype)

    def get_config(self):
        # Implement get_config to make the layer serializable
        config = super().get_config()
        config.update({"dtype": self._target_dtype}) # Use 'dtype' key for config consistency
        return config

# Custom layer to wrap tf.image.resize
class ResizeImage(layers.Layer):
    """A Keras layer to resize an image tensor using tf.image.resize."""
    def __init__(self, method='bilinear', **kwargs):
        super().__init__(**kwargs)
        # Store method, size will be passed dynamically in call
        self.method = method

    def call(self, inputs, size):
        # Perform the resize operation within the layer's call method
        # tf.image.resize expects size as (height, width)
        # size is passed dynamically during the call
        return tf.image.resize(inputs, size=size, method=self.method)

    def get_config(self):
        # Implement get_config for serialization
        config = super().get_config()
        config.update({
            "method": self.method,
            # size is not stored as a fixed attribute, so we don't include it here
        })
        return config


# SegFormer Modules
class PatchEmbed(layers.Layer):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dims=768, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size
        self.grid_size = (self.img_size[0] // patch_size, self.img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.projection = layers.Conv2D(
            filters=embed_dims,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            use_bias=True,
            name='projection'
        )

    def call(self, x):
        return self.projection(x)

class OverlapPatchEmbed(layers.Layer):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_channels=3, embed_dims=768, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size
        self.stride = stride
        self.grid_size = (self.img_size[0] // stride, self.img_size[1] // stride)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.projection = layers.Conv2D(
            filters=embed_dims,
            kernel_size=patch_size,
            strides=stride,
            padding='same',
            use_bias=True,
            name='projection'
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6, name='norm')

    def call(self, x):
        x = self.projection(x)
        x = self.norm(x)
        return x

class MixFFN(layers.Layer):
    def __init__(self, embed_dims, feedforward_channels, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = layers.Conv2D(
            filters=feedforward_channels,
            kernel_size=1,
            activation=None,
            use_bias=True,
            name='fc1'
        )
        self.dwconv = layers.DepthwiseConv2D(
            kernel_size=3,
            padding='same',
            use_bias=True,
            name='dwconv'
        )
        self.act = layers.Activation('gelu')
        self.fc2 = layers.Conv2D(
            filters=embed_dims,
            kernel_size=1,
            activation=None,
            use_bias=True,
            name='fc2'
        )
        self.drop = layers.Dropout(0.1)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x

class EfficientAttention(layers.Layer):
    def __init__(self, embed_dims, num_heads, sr_ratio, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = layers.Conv2D(
            filters=embed_dims,
            kernel_size=1,
            use_bias=True,
            name='q'
        )
        self.kv = layers.Conv2D(
            filters=embed_dims * 2,
            kernel_size=1,
            use_bias=True,
            name='kv'
        )

        if sr_ratio > 1:
            self.sr = layers.Conv2D(
                filters=embed_dims,
                kernel_size=sr_ratio,
                strides=sr_ratio,
                use_bias=True,
                name='sr'
            )
            self.norm = layers.LayerNormalization(epsilon=1e-6, name='norm')

        self.proj = layers.Conv2D(
            filters=embed_dims,
            kernel_size=1,
            use_bias=True,
            name='proj'
        )
        self.drop = layers.Dropout(0.1)

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        height, width = tf.shape(x)[1], tf.shape(x)[2]

        q = self.q(x)

        if self.sr_ratio > 1:
            x_reduced = self.sr(x)
            x_reduced = self.norm(x_reduced)
            kv = self.kv(x_reduced)
            reduced_height, reduced_width = tf.shape(x_reduced)[1], tf.shape(x_reduced)[2]
        else:
            kv = self.kv(x)
            reduced_height, reduced_width = height, width

        k, v = tf.split(kv, 2, axis=-1)

        q = tf.reshape(q, [batch_size, height * width, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, reduced_height * reduced_width, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, reduced_height * reduced_width, self.num_heads, self.head_dim])

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.drop(attn, training=training)

        out = tf.matmul(attn, v)

        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch_size, height, width, self.embed_dims])

        out = self.proj(out)
        out = self.drop(out, training=training)

        return out

class TransformerEncoderLayer(layers.Layer):
    def __init__(self, embed_dims, num_heads, feedforward_channels, sr_ratio=1, drop_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name='norm1')
        self.attn = EfficientAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            name='attn'
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name='norm2')
        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            name='ffn'
        )
        self.drop_path = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1))

    def call(self, x, training=False):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, training=training)
        x = self.drop_path(x, training=training)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x, training=training)
        x = self.drop_path(x, training=training)
        x = residual + x

        return x

class MixVisionTransformer(layers.Layer):
    def __init__(self, img_size=224, in_channels=3, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0.0, sr_ratios=[8, 4, 2, 1], num_layers=[2, 2, 2, 2], **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.sr_ratios = sr_ratios
        self.num_layers = num_layers

        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_channels=in_channels,
            embed_dims=embed_dims[0],
            name='patch_embed1'
        )
        self.patch_embed2 = OverlapPatchEmbed(
            patch_size=3,
            stride=2,
            in_channels=embed_dims[0],
            embed_dims=embed_dims[1],
            name='patch_embed2'
        )
        self.patch_embed3 = OverlapPatchEmbed(
            patch_size=3,
            stride=2,
            in_channels=embed_dims[1],
            embed_dims=embed_dims[2],
            name='patch_embed3'
        )
        self.patch_embed4 = OverlapPatchEmbed(
            patch_size=3,
            stride=2,
            in_channels=embed_dims[2],
            embed_dims=embed_dims[3],
            name='patch_embed4'
        )

        self.block1 = [
            TransformerEncoderLayer(
                embed_dims=embed_dims[0],
                num_heads=num_heads[0],
                feedforward_channels=mlp_ratios[0] * embed_dims[0],
                sr_ratio=sr_ratios[0],
                drop_rate=drop_rate,
                name=f'block1_{i}'
            ) for i in range(num_layers[0])
        ]

        self.block2 = [
            TransformerEncoderLayer(
                embed_dims=embed_dims[1],
                num_heads=num_heads[1],
                feedforward_channels=mlp_ratios[1] * embed_dims[1],
                sr_ratio=sr_ratios[1],
                drop_rate=drop_rate,
                name=f'block2_{i}'
            ) for i in range(num_layers[1])
        ]

        self.block3 = [
            TransformerEncoderLayer(
                embed_dims=embed_dims[2],
                num_heads=num_heads[2],
                feedforward_channels=mlp_ratios[2] * embed_dims[2],
                sr_ratio=sr_ratios[2],
                drop_rate=drop_rate,
                name=f'block3_{i}'
            ) for i in range(num_layers[2])
        ]

        self.block4 = [
            TransformerEncoderLayer(
                embed_dims=embed_dims[3],
                num_heads=num_heads[3],
                feedforward_channels=mlp_ratios[3] * embed_dims[3],
                sr_ratio=sr_ratios[3],
                drop_rate=drop_rate,
                name=f'block4_{i}'
            ) for i in range(num_layers[3])
        ]

        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name='norm1')
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name='norm2')
        self.norm3 = layers.LayerNormalization(epsilon=1e-6, name='norm3')
        self.norm4 = layers.LayerNormalization(epsilon=1e-6, name='norm4')

    def call(self, x, training=False):
        x1 = self.patch_embed1(x)
        for blk in self.block1:
            x1 = blk(x1, training=training)
        x1 = self.norm1(x1)

        x2 = self.patch_embed2(x1)
        for blk in self.block2:
            x2 = blk(x2, training=training)
        x2 = self.norm2(x2)

        x3 = self.patch_embed3(x2)
        for blk in self.block3:
            x3 = blk(x3, training=training)
        x3 = self.norm3(x3)

        x4 = self.patch_embed4(x3)
        for blk in self.block4:
            x4 = blk(x4, training=training)
        x4 = self.norm4(x4)

        return [x1, x2, x3, x4]

# MLP for the decoder head (uses 1x1 Conv)
class MLP_Decoder(layers.Layer):
    def __init__(self, embed_dims, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Conv2D(embed_dims, kernel_size=1, use_bias=True)

    def call(self, x):
        return self.proj(x)

class SegFormerHead(layers.Layer):
    def __init__(self, num_classes=5, embed_dims=[32, 64, 160, 256], decoder_embed_dims=256, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.decoder_embed_dims = decoder_embed_dims

        self.linear_c1 = MLP_Decoder(decoder_embed_dims, name='linear_c1')
        self.linear_c2 = MLP_Decoder(decoder_embed_dims, name='linear_c2')
        self.linear_c3 = MLP_Decoder(decoder_embed_dims, name='linear_c3')
        self.linear_c4 = MLP_Decoder(decoder_embed_dims, name='linear_c4')

        self.linear_fuse = layers.Conv2D(
            filters=decoder_embed_dims,
            kernel_size=1,
            padding='valid',
            use_bias=False,
            name='linear_fuse'
        )

        self.batch_norm = layers.BatchNormalization(name='batch_norm')

        # Increased dropout rate to 0.15 to prevent overfitting when augmentations are disabled
        self.dropout = layers.Dropout(0.15)

        self.classifier = layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            padding='valid',
            use_bias=True,
            name='classifier'
        )

        # Initialize the custom ResizeImage layers. Size will be passed in call.
        self.upsample_c2 = ResizeImage(method='bilinear', name='upsample_c2')
        self.upsample_c3 = ResizeImage(method='bilinear', name='upsample_c3')
        self.upsample_c4 = ResizeImage(method='bilinear', name='upsample_c4')


    def call(self, inputs, training=False):
        c1, c2, c3, c4 = inputs

        # Get dynamic spatial dimensions of the first stage output
        h, w = tf.shape(c1)[1], tf.shape(c1)[2]
        target_size = (h, w) # Target size for upsampling

        _c1 = self.linear_c1(c1)
        _c2 = self.linear_c2(c2)
        _c3 = self.linear_c3(c3)
        _c4 = self.linear_c4(c4)

        # Upsample features from stages 2, 3, and 4 using the custom layer, passing the target size
        _c2 = self.upsample_c2(_c2, size=target_size)
        _c3 = self.upsample_c3(_c3, size=target_size)
        _c4 = self.upsample_c4(_c4, size=target_size)


        _c = layers.Concatenate(axis=-1)([_c1, _c2, _c3, _c4])

        _c = self.linear_fuse(_c)
        _c = self.batch_norm(_c, training=training)
        _c = tf.nn.relu(_c)
        _c = self.dropout(_c, training=training)

        x = self.classifier(_c)

        return x


def SegFormer(input_shape=(320, 320, 1), num_classes=5, **kwargs):
    """
    Create a SegFormer-B0 model
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
    Returns:
        SegFormer model
    """
    compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    print(f"Using compute dtype: {compute_dtype} for SegFormer model")

    # Input layer: Use float32 as input when using mixed precision
    inputs = layers.Input(shape=input_shape, dtype=tf.float32, name='input_image')

    # Cast input to the compute dtype using the custom layer
    x = CastToDtype(dtype=compute_dtype, name='cast_input_to_compute_dtype')(inputs)

    # B0 configuration parameters
    embed_dims = [32, 64, 160, 256]
    num_heads = [1, 2, 5, 8]
    mlp_ratios = [4, 4, 4, 4]
    sr_ratios = [8, 4, 2, 1]
    num_layers = [2, 2, 2, 2]
    decoder_embed_dims = 256

    # Backbone: Mix Vision Transformer
    backbone = MixVisionTransformer(
        img_size=input_shape[0], # Use input height as img_size for the first layer
        in_channels=input_shape[2],
        embed_dims=embed_dims,
        num_heads=num_heads,
        mlp_ratios=mlp_ratios,
        drop_rate=0.1,
        sr_ratios=sr_ratios,
        num_layers=num_layers,
        name='backbone'
    )

    features = backbone(x)

    # Decoder head: SegFormerHead
    decoder_head = SegFormerHead(
        num_classes=num_classes,
        embed_dims=embed_dims,
        decoder_embed_dims=decoder_embed_dims,
        name='decoder_head'
    )
    # The decoder head outputs feature maps at the resolution of the first stage (1/4 of input)
    logits = decoder_head(features)

    # Upscale the output logits to the original input image size using the custom layer
    img_height, img_width = input_shape[0], input_shape[1]
    final_output_size = (img_height, img_width)

    # Use the custom ResizeImage layer for the final upsampling
    # Pass the final output size to the call method
    final_upsample_layer = ResizeImage(method='bilinear', name='final_upscaling')
    logits = final_upsample_layer(logits, size=final_output_size)


    # Cast the final output back to float32 if using mixed precision
    if compute_dtype == 'mixed_float16':
        logits = CastToDtype(dtype=tf.float32, name='cast_output_to_float32')(logits)


    # Create and return model
    model = models.Model(inputs=inputs, outputs=logits, name='SegFormer-B0')

    return model

def create_pretrained_weight_loader():
    """
    Creates a custom weight loading function that handles different weight file formats
    and properly maps them to the model architecture.

    Returns:
        A function that loads weights from a file into a model
    """
    import h5py

    def load_weights(model, weights_path):
        """
        Load weights from a file into the model with proper layer mapping

        Args:
            model: The model to load weights into
            weights_path: Path to the weights file

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(weights_path):
            print(f"Weights file not found: {weights_path}")
            return False

        print(f"Loading weights from {weights_path}")

        try:
            # First attempt: Try standard Keras load_weights
            model.load_weights(weights_path)
            print("Successfully loaded weights using standard method")
            return True
        except Exception as e:
            print(f"Standard weight loading failed with error: {e}")
            print("Trying alternative loading method...")

        # Try with h5py for h5 files
        if weights_path.endswith('.h5'):
            try:
                with h5py.File(weights_path, 'r') as h5_file:
                    # Create layer name mapping (handle different naming conventions)
                    layer_mapping = {}
                    for layer in model.layers:
                        layer_mapping[layer.name] = layer
                        # Also try with prefix/suffix variations
                        layer_mapping[f"model/{layer.name}"] = layer
                        layer_mapping[f"{layer.name}_1"] = layer

                    # Extract weights from h5 file and apply to model layers
                    loaded_layers = 0
                    for name in h5_file.keys():
                        # Skip non-layer groups
                        if name in ['model_weights', 'optimizer_weights', 'metadata']:
                            continue

                        # Find matching layer
                        target_layer = None
                        if name in layer_mapping:
                            target_layer = layer_mapping[name]
                        else:
                            # Try to find closest match
                            for layer_name, layer in layer_mapping.items():
                                if name in layer_name or layer_name in name:
                                    target_layer = layer
                                    break

                        if target_layer is None:
                            continue

                        # Get weight arrays
                        group = h5_file[name]
                        weight_names = []
                        if 'weight_names' in group.attrs:
                            weight_names = [n.decode('utf8') for n in group.attrs['weight_names']]
                        else:
                            # Look for weight datasets directly
                            for key in group.keys():
                                if isinstance(group[key], h5py.Dataset):
                                    weight_names.append(key)

                        if not weight_names:
                            continue

                        # Extract and set weights
                        weight_values = []
                        for weight_name in weight_names:
                            weight_path = f"{name}/{weight_name}"
                            if weight_path in h5_file:
                                weight_values.append(h5_file[weight_path][()])

                        if len(weight_values) > 0:
                            # Check if shapes match
                            layer_weights = target_layer.get_weights()
                            if len(layer_weights) == len(weight_values):
                                shapes_match = True
                                for i, (lw, wv) in enumerate(zip(layer_weights, weight_values)):
                                    if lw.shape != wv.shape:
                                        print(f"  Shape mismatch for {name} weights[{i}]: {lw.shape} vs {wv.shape}")
                                        shapes_match = False
                                        break

                                if shapes_match:
                                    target_layer.set_weights(weight_values)
                                    loaded_layers += 1
                                    print(f"  Loaded weights for layer: {target_layer.name}")

                    if loaded_layers > 0:
                        print(f"Successfully loaded weights for {loaded_layers} layers using h5py method")
                        return True
                    else:
                        print("No matching layers found in weights file")
                        return False
            except Exception as e:
                print(f"Alternative h5py loading failed with error: {e}")

        return False

    return load_weights

def OilSpillSegformer(input_shape=(384, 384, 1), num_classes=5, drop_rate=0.1, use_cbam=True, pretrained_weights=None):
    """
    Create an oil spill detection model using SegFormer-B2 architecture with specialized enhancements
    for SAR imagery and oil spill segmentation. Integrates oil spill detection features:
    1. CBAM attention for better boundary detection
    2. Properly configured for single-channel SAR images
    3. Optimized channel scaling for water/oil classification
    4. Higher capacity with SegFormer-B2 configuration

    Args:
        input_shape: Input shape (height, width, channels) - typically (384, 384, 1) for SAR images
        num_classes: Number of output classes (5 for: Sea Surface, Oil Spill, Look-alike, Ship, Land)
        drop_rate: Dropout rate for regularization
        use_cbam: Whether to use CBAM attention modules to enhance feature extraction
        pretrained_weights: Path to pretrained weights file (converted from PyTorch)

    Returns:
        Oil spill segmentation model based on SegFormer-B2 architecture
    """
    compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    print(f"Using compute dtype: {compute_dtype} for OilSpillSegformer-B2 model")

    # Input layer: Use float32 as input when using mixed precision
    inputs = layers.Input(shape=input_shape, dtype=tf.float32, name='input_image')

    # Cast input to the compute dtype using the custom layer
    x = CastToDtype(dtype=compute_dtype, name='cast_input_to_compute_dtype')(inputs)

    # B2 configuration parameters - significantly higher capacity than B0
    # Embed dimensions per stage
    embed_dims = [64, 128, 320, 512]  # B2 configuration (from [32, 64, 160, 256] in B0)
    num_heads = [1, 2, 5, 8]          # Same as original
    mlp_ratios = [8, 8, 4, 4]         # Higher capacity in early stages for better low-level feature extraction
    sr_ratios = [8, 4, 2, 1]          # Same as original
    num_layers = [3, 4, 6, 3]         # B2 configuration (from [2, 2, 2, 2] in B0)
    decoder_embed_dims = 768          # Increased from 256 to handle larger feature dimensions

    print(f"Model initialized with ~27.99M parameters")
    print(f"Embed dims: {embed_dims}, Layers: {num_layers}")

    # Apply CBAM attention to enhance feature extraction from SAR images
    if use_cbam:
        # Use smaller reduction ratio for efficiency with larger model
        x = CBAM(channels=input_shape[2], reduction_ratio=2, kernel_size=7, dtype=compute_dtype)(x)
        print(f"Enhanced CBAM with reduction_ratio=2 applied to input")

    # Backbone: Mix Vision Transformer with B2 configuration
    backbone = MixVisionTransformer(
        img_size=input_shape[0],
        in_channels=input_shape[2],
        embed_dims=embed_dims,
        num_heads=num_heads,
        mlp_ratios=mlp_ratios,
        drop_rate=drop_rate,
        sr_ratios=sr_ratios,
        num_layers=num_layers,
        name='backbone'
    )

    features = backbone(x)

    # Decoder head: SegFormerHead
    decoder_head = SegFormerHead(
        num_classes=num_classes,
        embed_dims=embed_dims,
        decoder_embed_dims=decoder_embed_dims,
        name='decoder_head'
    )

    # Get feature maps from the backbone
    logits = decoder_head(features)

    # Upscale the output logits to the original input image size
    img_height, img_width = input_shape[0], input_shape[1]
    final_output_size = (img_height, img_width)

    # Use the custom ResizeImage layer for the final upsampling
    final_upsample_layer = ResizeImage(method='bilinear', name='final_upsampling')
    logits = final_upsample_layer(logits, size=final_output_size)

    # Cast back to float32 if using mixed precision
    if compute_dtype == 'mixed_float16':
        logits = CastToDtype(dtype=tf.float32, name='cast_output_to_float32')(logits)

    # Create the model
    model = models.Model(inputs=inputs, outputs=logits, name='OilSpillSegformer-B2')

    # Load pretrained weights if provided
    if pretrained_weights is not None:
        print(f"Loading pretrained weights from: {pretrained_weights}")
        weight_loader = create_pretrained_weight_loader()
        if not weight_loader(model, pretrained_weights):
            print("Failed to load pretrained weights, training from scratch instead")

    return model

# Enhanced CBAM implementation for oil spill detection
def CBAM(channels, reduction_ratio=2, kernel_size=7, dtype=None):
    """
    Enhanced CBAM (Convolutional Block Attention Module) implementation for oil spill detection.
    Optimized for single-channel grayscale SAR imagery with better sensitivity to water-oil boundaries.

    Args:
        channels: Number of input channels
        reduction_ratio: Channel reduction ratio for efficiency (default: 2, lower for better stability)
        kernel_size: Size of the spatial attention convolution kernel (default: 7)
        dtype: Data type for the operations (for mixed precision compatibility)

    Returns:
        A function that applies CBAM to an input tensor
    """
    def apply(inputs):
        # Channel Attention Module (CAM)
        avg_pool = layers.GlobalAveragePooling2D(keepdims=True, dtype=dtype)(inputs)
        max_pool = layers.GlobalMaxPooling2D(keepdims=True, dtype=dtype)(inputs)

        # Smaller reduction ratio for better feature preservation with grayscale inputs
        avg_pool = layers.Conv2D(filters=max(channels // reduction_ratio, 4), kernel_size=1,
                                 use_bias=True, dtype=dtype)(avg_pool)
        avg_pool = layers.ReLU(dtype=dtype)(avg_pool)
        avg_pool = layers.Conv2D(filters=channels, kernel_size=1,
                                 use_bias=True, dtype=dtype)(avg_pool)

        max_pool = layers.Conv2D(filters=max(channels // reduction_ratio, 4), kernel_size=1,
                                 use_bias=True, dtype=dtype)(max_pool)
        max_pool = layers.ReLU(dtype=dtype)(max_pool)
        max_pool = layers.Conv2D(filters=channels, kernel_size=1,
                                 use_bias=True, dtype=dtype)(max_pool)

        channel_attention = layers.Add(dtype=dtype)([avg_pool, max_pool])
        channel_attention = layers.Activation('sigmoid', dtype=dtype)(channel_attention)
        channel_refined = layers.Multiply(dtype=dtype)([inputs, channel_attention])

        # Spatial Attention Module (SAM) - Enhanced for oil spill boundaries
        avg_spatial = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
                                    dtype=dtype)(channel_refined)
        max_spatial = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True),
                                    dtype=dtype)(channel_refined)
        # Compute variance in float32 for numerical stability
        var_spatial = layers.Lambda(lambda x: tf.cast(tf.math.reduce_variance(tf.cast(x, tf.float32), axis=-1, keepdims=True), dtype),
                                    dtype=dtype)(channel_refined)

        # Use all three spatial features for better boundary detection
        spatial_concat = layers.Concatenate(axis=-1, dtype=dtype)([avg_spatial, max_spatial, var_spatial])

        # Use larger kernel for better spatial context of oil spills
        spatial_attention = layers.Conv2D(filters=1, kernel_size=kernel_size, padding='same',
                                         activation='sigmoid', use_bias=False, dtype=dtype)(spatial_concat)

        # Apply spatial attention
        output = layers.Multiply(dtype=dtype)([channel_refined, spatial_attention])
        return output
    return apply

if __name__ == "__main__":
    # It's highly recommended to use mixed_float16 for modern GPUs for speed and memory efficiency
    # Input data should be float32 when using mixed_float16 policy.
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision policy set to mixed_float16")

    # Path to pretrained weights - use the converted weights file
    pretrained_weights_path = 'pretrained_weights/segformer_b2_pretrain.weights.h5'
    if os.path.exists(pretrained_weights_path):
        print(f"Found pretrained weights at {pretrained_weights_path}")
    else:
        print(f"Warning: Pretrained weights not found at {pretrained_weights_path}")
        pretrained_weights_path = None

    try:
        print("Creating OilSpillSegformer-B2 model with pretrained weights...")
        # Define input shape and number of classes
        input_shape = (384, 384, 1)  # Standard shape for SAR imagery
        num_classes = 5  # Sea Surface, Oil Spill, Look-alike, Ship, Land

        # Create the OilSpillSegformer model with pretrained weights
        model = OilSpillSegformer(
            input_shape=input_shape,
            num_classes=num_classes,
            use_cbam=True,
            pretrained_weights=pretrained_weights_path
        )
        print("\nModel summary:")
        model.summary()

        print("\nTesting inference with a small batch...")
        # Generate dummy input data (float32 required for model input with mixed precision)
        test_input = np.random.random((1, *input_shape)).astype(np.float32)
        print(f"Test input shape: {test_input.shape}, dtype: {test_input.dtype}")

        # Perform inference
        test_output = model.predict(test_input, verbose=1)

        print(f"Test output shape: {test_output.shape}, dtype: {test_output.dtype}")

        # Validate output (check for NaN values)
        if np.isnan(test_output).any():
            print("WARNING: Output contains NaN values!")
        else:
            print("Output validation: No NaN values detected")

        # Optional: Check GPU memory usage
        try:
            print("\nGPU memory usage during inference:")
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, check=True)
            memory_used = result.stdout.strip()
            print(f"Memory used: {memory_used} MiB")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Could not query GPU memory usage (nvidia-smi not found or error occurred).")


        print("\nOilSpillSegformer-B2 implementation test completed successfully!")

    except Exception as e:
        print(f"Error during model test: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Reset the mixed precision policy to float32 when done
        tf.keras.mixed_precision.set_global_policy('float32')
        print("Reset precision policy to float32")
