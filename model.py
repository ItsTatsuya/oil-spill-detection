import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

class CastToDtype(layers.Layer):
    def __init__(self, dtype, **kwargs):
        super().__init__(**kwargs)
        self._target_dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, self._target_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"dtype": self._target_dtype})
        return config

class ResizeImage(layers.Layer):
    def __init__(self, method='bilinear', **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def call(self, inputs, size):
        return tf.image.resize(inputs, size=size, method=self.method)

    def get_config(self):
        config = super().get_config()
        config.update({"method": self.method})
        return config


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
        self.dropout = layers.Dropout(0.15)
        self.classifier = layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            padding='valid',
            use_bias=True,
            name='classifier'
        )

        self.upsample_c2 = ResizeImage(method='bilinear', name='upsample_c2')
        self.upsample_c3 = ResizeImage(method='bilinear', name='upsample_c3')
        self.upsample_c4 = ResizeImage(method='bilinear', name='upsample_c4')

    def call(self, inputs, training=False):
        c1, c2, c3, c4 = inputs
        h, w = tf.shape(c1)[1], tf.shape(c1)[2]
        target_size = (h, w)

        _c1 = self.linear_c1(c1)
        _c2 = self.linear_c2(c2)
        _c3 = self.linear_c3(c3)
        _c4 = self.linear_c4(c4)

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
    compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

    inputs = layers.Input(shape=input_shape, dtype=tf.float32, name='input_image')
    x = CastToDtype(dtype=compute_dtype, name='cast_input_to_compute_dtype')(inputs)

    embed_dims = [32, 64, 160, 256]
    num_heads = [1, 2, 5, 8]
    mlp_ratios = [4, 4, 4, 4]
    sr_ratios = [8, 4, 2, 1]
    num_layers = [2, 2, 2, 2]
    decoder_embed_dims = 256

    backbone = MixVisionTransformer(
        img_size=input_shape[0],
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

    decoder_head = SegFormerHead(
        num_classes=num_classes,
        embed_dims=embed_dims,
        decoder_embed_dims=decoder_embed_dims,
        name='decoder_head'
    )
    logits = decoder_head(features)

    img_height, img_width = input_shape[0], input_shape[1]
    final_output_size = (img_height, img_width)

    final_upsample_layer = ResizeImage(method='bilinear', name='final_upscaling')
    logits = final_upsample_layer(logits, size=final_output_size)

    if compute_dtype == 'mixed_float16':
        logits = CastToDtype(dtype=tf.float32, name='cast_output_to_float32')(logits)

    model = models.Model(inputs=inputs, outputs=logits, name='SegFormer-B0')

    return model

def create_pretrained_weight_loader():
    import h5py

    def load_weights(model, weights_path):
        if not os.path.exists(weights_path):
            print(f"Weights file not found: {weights_path}")
            return False

        print(f"Loading weights from {weights_path}")

        try:
            model.load_weights(weights_path)
            print("Successfully loaded weights using standard method")
            return True
        except Exception as e:
            print(f"Standard weight loading failed with error: {e}")
            print("Trying alternative loading method...")

        if weights_path.endswith('.h5'):
            try:
                with h5py.File(weights_path, 'r') as h5_file:
                    layer_mapping = {}
                    for layer in model.layers:
                        layer_mapping[layer.name] = layer
                        layer_mapping[f"model/{layer.name}"] = layer
                        layer_mapping[f"{layer.name}_1"] = layer

                    loaded_layers = 0
                    for name in h5_file.keys():
                        if name in ['model_weights', 'optimizer_weights', 'metadata']:
                            continue

                        target_layer = None
                        if name in layer_mapping:
                            target_layer = layer_mapping[name]
                        else:
                            for layer_name, layer in layer_mapping.items():
                                if name in layer_name or layer_name in name:
                                    target_layer = layer
                                    break

                        if target_layer is None:
                            continue

                        group = h5_file[name]
                        weight_names = []
                        if 'weight_names' in group.attrs:
                            weight_names = [n.decode('utf8') for n in group.attrs['weight_names']]
                        else:
                            for key in group.keys():
                                if isinstance(group[key], h5py.Dataset):
                                    weight_names.append(key)

                        if not weight_names:
                            continue

                        weight_values = []
                        for weight_name in weight_names:
                            weight_path = f"{name}/{weight_name}"
                            if weight_path in h5_file:
                                weight_values.append(h5_file[weight_path][()])

                        if len(weight_values) > 0:
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
    compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

    inputs = layers.Input(shape=input_shape, dtype=tf.float32, name='input_image')
    x = CastToDtype(dtype=compute_dtype, name='cast_input_to_compute_dtype')(inputs)

    embed_dims = [64, 128, 320, 512]
    num_heads = [1, 2, 5, 8]
    mlp_ratios = [8, 8, 4, 4]
    sr_ratios = [8, 4, 2, 1]
    num_layers = [3, 4, 6, 3]
    decoder_embed_dims = 768

    if use_cbam:
        x = CBAM(channels=input_shape[2], reduction_ratio=2, kernel_size=7, dtype=compute_dtype)(x)

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

    decoder_head = SegFormerHead(
        num_classes=num_classes,
        embed_dims=embed_dims,
        decoder_embed_dims=decoder_embed_dims,
        name='decoder_head'
    )

    logits = decoder_head(features)

    img_height, img_width = input_shape[0], input_shape[1]
    final_output_size = (img_height, img_width)

    final_upsample_layer = ResizeImage(method='bilinear', name='final_upsampling')
    logits = final_upsample_layer(logits, size=final_output_size)

    if compute_dtype == 'mixed_float16':
        logits = CastToDtype(dtype=tf.float32, name='cast_output_to_float32')(logits)

    model = models.Model(inputs=inputs, outputs=logits, name='OilSpillSegformer-B2')

    if pretrained_weights is not None:
        print(f"Loading pretrained weights from: {pretrained_weights}")
        weight_loader = create_pretrained_weight_loader()
        if not weight_loader(model, pretrained_weights):
            print("Failed to load pretrained weights, training from scratch instead")

    return model

def CBAM(channels, reduction_ratio=2, kernel_size=7, dtype=None):
    def apply(inputs):
        avg_pool = layers.GlobalAveragePooling2D(keepdims=True, dtype=dtype)(inputs)
        max_pool = layers.GlobalMaxPooling2D(keepdims=True, dtype=dtype)(inputs)

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

        avg_spatial = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
                                    dtype=dtype)(channel_refined)
        max_spatial = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True),
                                    dtype=dtype)(channel_refined)
        var_spatial = layers.Lambda(lambda x: tf.cast(tf.math.reduce_variance(tf.cast(x, tf.float32), axis=-1, keepdims=True), dtype),
                                    dtype=dtype)(channel_refined)

        spatial_concat = layers.Concatenate(axis=-1, dtype=dtype)([avg_spatial, max_spatial, var_spatial])

        spatial_attention = layers.Conv2D(filters=1, kernel_size=kernel_size, padding='same',
                                         activation='sigmoid', use_bias=False, dtype=dtype)(spatial_concat)

        output = layers.Multiply(dtype=dtype)([channel_refined, spatial_attention])
        return output
    return apply
