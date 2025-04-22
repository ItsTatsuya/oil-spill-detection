#!/usr/bin/env python3
"""
Script to convert MMSegmentation SegFormer-B2 weights (PyTorch) to TensorFlow format
for use with OilSpillSegformer-B2 model.

Requirements:
- torch
- mmcv
- mmsegmentation
- tensorflow
"""

import os
import sys
import numpy as np
import argparse
from collections import OrderedDict

def silent_tf_import():
    import os
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

def silent_torch_import():
    import os
    import sys
    orig_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(orig_stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, orig_stderr_fd)
    os.close(devnull_fd)

    import torch

    os.dup2(saved_stderr_fd, orig_stderr_fd)
    os.close(saved_stderr_fd)

    return torch

try:
    tf = silent_tf_import()
    torch = silent_torch_import()
    print("Successfully imported TensorFlow and PyTorch")
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please install required packages: torch, mmcv, mmsegmentation, tensorflow")
    sys.exit(1)

def convert_segformer_b2_backbone(pytorch_state_dict):
    """
    Convert SegFormer-B2 backbone weights from PyTorch to TensorFlow format.

    Args:
        pytorch_state_dict: PyTorch state dictionary

    Returns:
        Dictionary mapping TensorFlow layer names to numpy arrays
    """
    tf_weights = {}

    # Parse PyTorch state_dict
    print("Converting backbone layers...")

    # Layer name mapping from PyTorch to TensorFlow
    backbone_mapping = {
        # First stage
        "backbone.patch_embed1.projection.weight": "backbone/patch_embed1/projection/kernel",
        "backbone.patch_embed1.projection.bias": "backbone/patch_embed1/projection/bias",
        "backbone.patch_embed1.norm.weight": "backbone/patch_embed1/norm/gamma",
        "backbone.patch_embed1.norm.bias": "backbone/patch_embed1/norm/beta",

        # Second stage
        "backbone.patch_embed2.projection.weight": "backbone/patch_embed2/projection/kernel",
        "backbone.patch_embed2.projection.bias": "backbone/patch_embed2/projection/bias",
        "backbone.patch_embed2.norm.weight": "backbone/patch_embed2/norm/gamma",
        "backbone.patch_embed2.norm.bias": "backbone/patch_embed2/norm/beta",

        # Third stage
        "backbone.patch_embed3.projection.weight": "backbone/patch_embed3/projection/kernel",
        "backbone.patch_embed3.projection.bias": "backbone/patch_embed3/projection/bias",
        "backbone.patch_embed3.norm.weight": "backbone/patch_embed3/norm/gamma",
        "backbone.patch_embed3.norm.bias": "backbone/patch_embed3/norm/beta",

        # Fourth stage
        "backbone.patch_embed4.projection.weight": "backbone/patch_embed4/projection/kernel",
        "backbone.patch_embed4.projection.bias": "backbone/patch_embed4/projection/bias",
        "backbone.patch_embed4.norm.weight": "backbone/patch_embed4/norm/gamma",
        "backbone.patch_embed4.norm.bias": "backbone/patch_embed4/norm/beta",

        # Output norms for each stage
        "backbone.norm1.weight": "backbone/norm1/gamma",
        "backbone.norm1.bias": "backbone/norm1/beta",
        "backbone.norm2.weight": "backbone/norm2/gamma",
        "backbone.norm2.bias": "backbone/norm2/beta",
        "backbone.norm3.weight": "backbone/norm3/gamma",
        "backbone.norm3.bias": "backbone/norm3/beta",
        "backbone.norm4.weight": "backbone/norm4/gamma",
        "backbone.norm4.bias": "backbone/norm4/beta",
    }

    # Handle encoder blocks - SegFormer-B2 has [3, 4, 6, 3] layers
    block_counts = [3, 4, 6, 3]  # B2 configuration

    # Define block mappings for different parts of a transformer encoder block
    for stage_idx in range(4):
        stage_num = stage_idx + 1
        for block_idx in range(block_counts[stage_idx]):
            # PyTorch prefix for this block
            pt_block_prefix = f"backbone.block{stage_num}.{block_idx}."
            # TensorFlow prefix for this block
            tf_block_prefix = f"backbone/block{stage_num}_{block_idx}/"

            # Attention layers
            block_mapping = {
                f"{pt_block_prefix}attn.q.weight": f"{tf_block_prefix}attn/q/kernel",
                f"{pt_block_prefix}attn.q.bias": f"{tf_block_prefix}attn/q/bias",
                f"{pt_block_prefix}attn.kv.weight": f"{tf_block_prefix}attn/kv/kernel",
                f"{pt_block_prefix}attn.kv.bias": f"{tf_block_prefix}attn/kv/bias",
                f"{pt_block_prefix}attn.proj.weight": f"{tf_block_prefix}attn/proj/kernel",
                f"{pt_block_prefix}attn.proj.bias": f"{tf_block_prefix}attn/proj/bias",

                # Spatial reduction for efficient attention
                f"{pt_block_prefix}attn.sr.weight": f"{tf_block_prefix}attn/sr/kernel",
                f"{pt_block_prefix}attn.sr.bias": f"{tf_block_prefix}attn/sr/bias",
                f"{pt_block_prefix}attn.norm.weight": f"{tf_block_prefix}attn/norm/gamma",
                f"{pt_block_prefix}attn.norm.bias": f"{tf_block_prefix}attn/norm/beta",

                # Layer norms
                f"{pt_block_prefix}norm1.weight": f"{tf_block_prefix}norm1/gamma",
                f"{pt_block_prefix}norm1.bias": f"{tf_block_prefix}norm1/beta",
                f"{pt_block_prefix}norm2.weight": f"{tf_block_prefix}norm2/gamma",
                f"{pt_block_prefix}norm2.bias": f"{tf_block_prefix}norm2/beta",

                # MixFFN layers
                f"{pt_block_prefix}ffn.fc1.weight": f"{tf_block_prefix}ffn/fc1/kernel",
                f"{pt_block_prefix}ffn.fc1.bias": f"{tf_block_prefix}ffn/fc1/bias",
                f"{pt_block_prefix}ffn.fc2.weight": f"{tf_block_prefix}ffn/fc2/kernel",
                f"{pt_block_prefix}ffn.fc2.bias": f"{tf_block_prefix}ffn/fc2/bias",
                f"{pt_block_prefix}ffn.dwconv.weight": f"{tf_block_prefix}ffn/dwconv/depthwise_kernel",
                f"{pt_block_prefix}ffn.dwconv.bias": f"{tf_block_prefix}ffn/dwconv/bias",
            }

            # Update the mapping dictionary
            backbone_mapping.update(block_mapping)

    # Apply the mapping to convert weights
    for pt_name, tf_name in backbone_mapping.items():
        if pt_name in pytorch_state_dict:
            weight = pytorch_state_dict[pt_name].numpy()

            # Handle convolution kernels - PyTorch uses (out_channels, in_channels, h, w)
            # TensorFlow uses (h, w, in_channels, out_channels)
            if "projection" in pt_name and "weight" in pt_name:
                weight = np.transpose(weight, (2, 3, 1, 0))
            elif "conv" in pt_name and "weight" in pt_name:
                if "dwconv" in pt_name:  # Depthwise conv
                    weight = np.transpose(weight, (2, 3, 0, 1))
                else:  # Regular conv
                    weight = np.transpose(weight, (2, 3, 1, 0))
            elif "fc" in pt_name and "weight" in pt_name:  # FC layers implemented as 1x1 convs
                weight = np.transpose(weight, (2, 3, 1, 0)) if len(weight.shape) == 4 else weight.T

            tf_weights[tf_name] = weight

    return tf_weights

def convert_segformer_b2_decoder(pytorch_state_dict):
    """
    Convert SegFormer-B2 decoder head weights from PyTorch to TensorFlow format.

    Args:
        pytorch_state_dict: PyTorch state dictionary

    Returns:
        Dictionary mapping TensorFlow layer names to numpy arrays
    """
    tf_weights = {}

    # Parse PyTorch state_dict
    print("Converting decoder head layers...")

    # Define decoder mapping
    decoder_mapping = {
        # Linear projections for each stage
        "decode_head.linear_c1.0.weight": "decoder_head/linear_c1/kernel",
        "decode_head.linear_c1.0.bias": "decoder_head/linear_c1/bias",
        "decode_head.linear_c2.0.weight": "decoder_head/linear_c2/kernel",
        "decode_head.linear_c2.0.bias": "decoder_head/linear_c2/bias",
        "decode_head.linear_c3.0.weight": "decoder_head/linear_c3/kernel",
        "decode_head.linear_c3.0.bias": "decoder_head/linear_c3/bias",
        "decode_head.linear_c4.0.weight": "decoder_head/linear_c4/kernel",
        "decode_head.linear_c4.0.bias": "decoder_head/linear_c4/bias",

        # Linear fusion
        "decode_head.linear_fuse.weight": "decoder_head/linear_fuse/kernel",

        # Batch norm
        "decode_head.batch_norm.weight": "decoder_head/batch_norm/gamma",
        "decode_head.batch_norm.bias": "decoder_head/batch_norm/beta",
        "decode_head.batch_norm.running_mean": "decoder_head/batch_norm/moving_mean",
        "decode_head.batch_norm.running_var": "decoder_head/batch_norm/moving_variance",
        "decode_head.batch_norm.num_batches_tracked": None,  # Not used in TF

        # Final classifier
        "decode_head.conv_seg.weight": "decoder_head/classifier/kernel",
        "decode_head.conv_seg.bias": "decoder_head/classifier/bias",
    }

    # Apply the mapping to convert weights
    for pt_name, tf_name in decoder_mapping.items():
        if pt_name in pytorch_state_dict and tf_name is not None:
            weight = pytorch_state_dict[pt_name].numpy()

            # Handle convolution kernels
            if ("conv" in pt_name or "linear" in pt_name) and "weight" in pt_name:
                if len(weight.shape) == 4:  # 2D conv
                    weight = np.transpose(weight, (2, 3, 1, 0))
                else:  # 1D conv
                    weight = weight.T

            tf_weights[tf_name] = weight

    return tf_weights

def convert_weights_to_tf_format(input_file, output_file, single_channel=True):
    """
    Convert MMSegmentation SegFormer-B2 weights to TensorFlow format

    Args:
        input_file: PyTorch checkpoint file (.pth)
        output_file: Output TensorFlow checkpoint file (.h5)
        single_channel: Whether to adapt weights for single-channel input
    """
    print(f"Loading PyTorch weights from: {input_file}")

    # Load PyTorch weights
    state_dict = torch.load(input_file, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # Convert backbone weights
    backbone_weights = convert_segformer_b2_backbone(state_dict)

    # Convert decoder weights
    decoder_weights = convert_segformer_b2_decoder(state_dict)

    # Combine all weights
    tf_weights = {}
    tf_weights.update(backbone_weights)
    tf_weights.update(decoder_weights)

    # If single-channel input, adapt the first convolution layer
    if single_channel:
        print("Adapting weights for single-channel input...")

        # In PyTorch SegFormer-B2, the first conv is backbone.patch_embed1.projection.weight
        # In TF, it's backbone/patch_embed1/projection/kernel
        if "backbone/patch_embed1/projection/kernel" in tf_weights:
            # Get the weight
            weight = tf_weights["backbone/patch_embed1/projection/kernel"]

            # Original weight shape: [kernel_h, kernel_w, in_channels, out_channels]
            # We need to convert from 3 channels to 1 channel
            if weight.shape[2] == 3:  # RGB input
                # Average the weights across RGB channels
                single_channel_weight = np.mean(weight, axis=2, keepdims=True)
                tf_weights["backbone/patch_embed1/projection/kernel"] = single_channel_weight
                print(f"First layer weights adapted from shape {weight.shape} to {single_channel_weight.shape}")

    # Create a TensorFlow model and assign weights
    print(f"Creating temporary TensorFlow model for saving weights...")

    # Import the model module
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model import OilSpillSegformer

    # Create a model with B2 configuration
    input_shape = (384, 384, 1)
    num_classes = 5  # For oil spill detection
    model = OilSpillSegformer(
        input_shape=input_shape,
        num_classes=num_classes,
        drop_rate=0.0,  # Set to 0 for inference
        use_cbam=False  # Don't use CBAM for the weight conversion
    )

    # Compile the model (required to build)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Set the weights
    for layer in model.layers:
        set_layer_weights_recursive(layer, tf_weights)

    # Save the weights
    print(f"Saving TensorFlow weights to: {output_file}")
    model.save_weights(output_file)
    print("Conversion completed successfully!")

    # Count the number of matched weights
    matched_count = len([k for k in tf_weights.keys() if any(k in w.name for w in model.weights)])
    total_weights = len(model.weights)
    print(f"Matched {matched_count} out of {total_weights} model weights")

    return model

def set_layer_weights_recursive(layer, weight_dict):
    """
    Recursively set weights for all sub-layers

    Args:
        layer: TensorFlow layer
        weight_dict: Dictionary of weights
    """
    if hasattr(layer, 'layers'):
        for sublayer in layer.layers:
            set_layer_weights_recursive(sublayer, weight_dict)

    # Try to find matching weights for this layer
    weights_to_set = []
    for weight in layer.weights:
        # Extract name without the model prefix
        name_parts = weight.name.split('/')
        name = '/'.join(name_parts[1:])  # Remove the first part (model name)
        name = name.split(':')[0]  # Remove the trailing :0

        if name in weight_dict:
            weights_to_set.append(weight_dict[name])
        else:
            # If no exact match, keep the current weight
            weights_to_set.append(weight.numpy())

    # Only set weights if we have the right number
    if weights_to_set and len(weights_to_set) == len(layer.weights):
        layer.set_weights(weights_to_set)

def get_pretrained_urls():
    """
    Returns URLs for pretrained SegFormer weights from MMSegmentation
    """
    return {
        'segformer_b0': 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth',
        'segformer_b1': 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth',
        'segformer_b2': 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth',
        'segformer_b3': 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_8x1_1024x1024_160k_cityscapes/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8a177.pth',
        'segformer_b4': 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_8x1_1024x1024_160k_cityscapes/segformer_mit-b4_8x1_1024x1024_160k_cityscapes_20211207_080709-07f6c333.pth',
        'segformer_b5': 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth',
    }

def download_pretrained_weights(model_type, output_dir):
    """
    Download pretrained weights from MMSegmentation

    Args:
        model_type: Model type (e.g., 'segformer_b2')
        output_dir: Output directory

    Returns:
        Path to downloaded weights
    """
    import requests
    from tqdm import tqdm

    urls = get_pretrained_urls()
    if model_type not in urls:
        print(f"Error: Unknown model type '{model_type}'. Available models: {list(urls.keys())}")
        return None

    url = urls[model_type]
    output_file = os.path.join(output_dir, os.path.basename(url))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download if file doesn't exist
    if not os.path.exists(output_file):
        print(f"Downloading {url} to {output_file}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(output_file, 'wb') as f, tqdm(
            desc=os.path.basename(url),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    else:
        print(f"Using existing file: {output_file}")

    return output_file

def main():
    parser = argparse.ArgumentParser(description='Convert MMSegmentation SegFormer weights to TensorFlow format')
    parser.add_argument('--input', type=str, help='Input PyTorch checkpoint file (.pth)')
    parser.add_argument('--output', type=str, default='segformer_b2.weights.h5', help='Output TensorFlow weights file (.h5)')
    parser.add_argument('--download', type=str, choices=list(get_pretrained_urls().keys()),
                        default='segformer_b2', help='Download and convert pretrained weights')
    parser.add_argument('--output-dir', type=str, default='pretrained_weights', help='Directory to save downloaded weights')
    parser.add_argument('--single-channel', action='store_true', default=True,
                        help='Adapt weights for single-channel input (default: True)')

    args = parser.parse_args()

    # If input file is not provided, download pretrained weights
    input_file = args.input
    if input_file is None:
        input_file = download_pretrained_weights(args.download, args.output_dir)
        if input_file is None:
            print("Error: Failed to download pretrained weights.")
            return

    # Convert weights to TF format
    model = convert_weights_to_tf_format(input_file, args.output, args.single_channel)

    print(f"\nModel summary:")
    model.summary()

    print(f"\nConverted weights saved to: {args.output}")
    print(f"You can now use them with OilSpillSegformer with the parameter: pretrained_weights='{args.output}'")

if __name__ == '__main__':
    main()
