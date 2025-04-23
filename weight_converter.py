import torch
import tensorflow as tf
import numpy as np
from model import OilSpillSegformer  # Your custom model
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weight_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_pytorch_weights(pth_path):
    """Load PyTorch weights from a .pth file."""
    try:
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"Pretrained weights not found at {pth_path}")
        state_dict = torch.load(pth_path, map_location='cpu')
        # MMSegmentation checkpoints store weights under 'state_dict'
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        logger.info(f"Loaded PyTorch weights from {pth_path}")

        # Print keys for debugging
        logger.info(f"PyTorch state_dict keys (first 5): {list(state_dict.keys())[:5]}")

        return state_dict
    except Exception as e:
        logger.error(f"Failed to load PyTorch weights: {e}")
        raise

def adapt_first_conv_weight(pytorch_weight):
    """Adapt the first conv layer's weight from 3 channels (RGB) to 1 channel (SAR)."""
    # PyTorch: [out_channels, in_channels=3, kernel_h, kernel_w]
    if pytorch_weight.shape[1] != 3:
        logger.warning(f"Expected 3 input channels, got {pytorch_weight.shape[1]}. Skipping adaptation.")
        return pytorch_weight
    # Average over RGB channels
    single_channel_weight = torch.mean(pytorch_weight, dim=1, keepdim=True)  # [out_channels, 1, kernel_h, kernel_w]
    logger.info("Adapted first conv layer from 3 channels to 1 channel")
    return single_channel_weight

def map_pytorch_to_tensorflow(state_dict, tf_model):
    """Map PyTorch weights to TensorFlow model layers."""
    weight_map = []
    tf_layers = {layer.name: layer for layer in tf_model.layers}

    # Find the backbone layer to access num_layers
    backbone = None
    for layer in tf_model.layers:
        if hasattr(layer, 'name') and layer.name == 'backbone':
            backbone = layer
            break

    # Get num_layers from backbone if found, otherwise use default values
    num_encoder_blocks = [3, 4, 6, 3]  # Default for SegFormer-B2
    if backbone and hasattr(backbone, 'num_layers'):
        num_encoder_blocks = backbone.num_layers

    # Get all keys from state_dict for reference
    all_keys = list(state_dict.keys())
    logger.info(f"Total number of keys in PyTorch state dict: {len(all_keys)}")

    # Create an accurate mapping based on the actual keys in the state dict
    # The structure follows: backbone.layers.stage_idx.block_idx...
    pytorch_to_tf = {}

    # Stage 0 - First stage (patch_embed1) - Maps to backbone.patch_embed1
    stage0_keys = [k for k in all_keys if 'backbone.layers.0.0' in k]
    if stage0_keys:
        pytorch_to_tf.update({
            'backbone.layers.0.0.projection.weight': ('backbone/patch_embed1/projection/kernel',
                lambda x: adapt_first_conv_weight(x).permute(2, 3, 1, 0).numpy()),
            'backbone.layers.0.0.projection.bias': ('backbone/patch_embed1/projection/bias',
                lambda x: x.numpy()),
            'backbone.layers.0.0.norm.weight': ('backbone/patch_embed1/norm/gamma',
                lambda x: x.numpy()),
            'backbone.layers.0.0.norm.bias': ('backbone/patch_embed1/norm/beta',
                lambda x: x.numpy()),
        })

    # Map transformer blocks for each stage
    for stage in range(4):
        for block in range(num_encoder_blocks[stage]):
            pt_prefix = f'backbone.layers.{stage}.{block+1}'
            tf_prefix = f'backbone/block{stage+1}_{block}'

            # Map attention module
            pytorch_to_tf.update({
                f'{pt_prefix}.1.norm1.weight': (f'{tf_prefix}/norm1/gamma', lambda x: x.numpy()),
                f'{pt_prefix}.1.norm1.bias': (f'{tf_prefix}/norm1/beta', lambda x: x.numpy()),
                f'{pt_prefix}.1.attn.q.weight': (f'{tf_prefix}/attn/q/kernel',
                    lambda x: x.permute(2, 3, 1, 0).numpy()),
                f'{pt_prefix}.1.attn.q.bias': (f'{tf_prefix}/attn/q/bias', lambda x: x.numpy()),
                f'{pt_prefix}.1.attn.kv.weight': (f'{tf_prefix}/attn/kv/kernel',
                    lambda x: x.permute(2, 3, 1, 0).numpy()),
                f'{pt_prefix}.1.attn.kv.bias': (f'{tf_prefix}/attn/kv/bias', lambda x: x.numpy()),
                f'{pt_prefix}.1.attn.proj.weight': (f'{tf_prefix}/attn/proj/kernel',
                    lambda x: x.permute(2, 3, 1, 0).numpy()),
                f'{pt_prefix}.1.attn.proj.bias': (f'{tf_prefix}/attn/proj/bias', lambda x: x.numpy()),

                # Map FFN module
                f'{pt_prefix}.1.norm2.weight': (f'{tf_prefix}/norm2/gamma', lambda x: x.numpy()),
                f'{pt_prefix}.1.norm2.bias': (f'{tf_prefix}/norm2/beta', lambda x: x.numpy()),
                f'{pt_prefix}.1.ffn.fc1.weight': (f'{tf_prefix}/ffn/fc1/kernel',
                    lambda x: x.permute(2, 3, 1, 0).numpy()),
                f'{pt_prefix}.1.ffn.fc1.bias': (f'{tf_prefix}/ffn/fc1/bias', lambda x: x.numpy()),
                f'{pt_prefix}.1.ffn.dwconv.weight': (f'{tf_prefix}/ffn/dwconv/depthwise_kernel',
                    lambda x: x.permute(2, 3, 0, 1).numpy()),
                f'{pt_prefix}.1.ffn.dwconv.bias': (f'{tf_prefix}/ffn/dwconv/bias', lambda x: x.numpy()),
                f'{pt_prefix}.1.ffn.fc2.weight': (f'{tf_prefix}/ffn/fc2/kernel',
                    lambda x: x.permute(2, 3, 1, 0).numpy()),
                f'{pt_prefix}.1.ffn.fc2.bias': (f'{tf_prefix}/ffn/fc2/bias', lambda x: x.numpy()),
            })

    # Stage transitions - patch embedding layers for stages 1-3
    stage_transitions = [
        ('backbone.layers.1.0', 'backbone/patch_embed2'),
        ('backbone.layers.2.0', 'backbone/patch_embed3'),
        ('backbone.layers.3.0', 'backbone/patch_embed4')
    ]

    for pt_prefix, tf_prefix in stage_transitions:
        pytorch_to_tf.update({
            f'{pt_prefix}.projection.weight': (f'{tf_prefix}/projection/kernel',
                lambda x: x.permute(2, 3, 1, 0).numpy()),
            f'{pt_prefix}.projection.bias': (f'{tf_prefix}/projection/bias', lambda x: x.numpy()),
            f'{pt_prefix}.norm.weight': (f'{tf_prefix}/norm/gamma', lambda x: x.numpy()),
            f'{pt_prefix}.norm.bias': (f'{tf_prefix}/norm/beta', lambda x: x.numpy()),
        })

    # Stage norms
    for stage in range(4):
        pytorch_to_tf.update({
            f'backbone.layers.{stage}.2.weight': (f'backbone/norm{stage+1}/gamma', lambda x: x.numpy()),
            f'backbone.layers.{stage}.2.bias': (f'backbone/norm{stage+1}/beta', lambda x: x.numpy()),
        })

    # Decoder head
    decoder_mappings = {
        'decode_head.conv_seg.weight': ('decoder_head/classifier/kernel',
            lambda x: x.permute(2, 3, 1, 0).numpy()),
        'decode_head.conv_seg.bias': ('decoder_head/classifier/bias', lambda x: x.numpy()),

        # Linear projections for each feature level
        'decode_head.linear_c1.conv.weight': ('decoder_head/linear_c1/proj/kernel',
            lambda x: x.permute(2, 3, 1, 0).numpy()),
        'decode_head.linear_c1.conv.bias': ('decoder_head/linear_c1/proj/bias', lambda x: x.numpy()),
        'decode_head.linear_c2.conv.weight': ('decoder_head/linear_c2/proj/kernel',
            lambda x: x.permute(2, 3, 1, 0).numpy()),
        'decode_head.linear_c2.conv.bias': ('decoder_head/linear_c2/proj/bias', lambda x: x.numpy()),
        'decode_head.linear_c3.conv.weight': ('decoder_head/linear_c3/proj/kernel',
            lambda x: x.permute(2, 3, 1, 0).numpy()),
        'decode_head.linear_c3.conv.bias': ('decoder_head/linear_c3/proj/bias', lambda x: x.numpy()),
        'decode_head.linear_c4.conv.weight': ('decoder_head/linear_c4/proj/kernel',
            lambda x: x.permute(2, 3, 1, 0).numpy()),
        'decode_head.linear_c4.conv.bias': ('decoder_head/linear_c4/proj/bias', lambda x: x.numpy()),

        # Fusion layer
        'decode_head.linear_fuse.weight': ('decoder_head/linear_fuse/kernel',
            lambda x: x.permute(2, 3, 1, 0).numpy()),
        'decode_head.linear_fuse.bias': ('decoder_head/linear_fuse/bias', lambda x: x.numpy()),

        # Batch norm in the decoder
        'decode_head.bn.weight': ('decoder_head/batch_norm/gamma', lambda x: x.numpy()),
        'decode_head.bn.bias': ('decoder_head/batch_norm/beta', lambda x: x.numpy()),
        'decode_head.bn.running_mean': ('decoder_head/batch_norm/moving_mean', lambda x: x.numpy()),
        'decode_head.bn.running_var': ('decoder_head/batch_norm/moving_variance', lambda x: x.numpy()),
    }

    pytorch_to_tf.update(decoder_mappings)

    # Log the keys that will be attempted to convert
    logger.info(f"Attempting to convert {len(pytorch_to_tf)} weights")
    matched_keys = set(pytorch_to_tf.keys()).intersection(set(all_keys))
    logger.info(f"Found {len(matched_keys)} matching keys in PyTorch state dict")

    # Process mapping
    for pt_key, (tf_key, transform) in pytorch_to_tf.items():
        if pt_key in state_dict:
            try:
                tf_weight = transform(state_dict[pt_key])
                weight_map.append((tf_key, tf_weight))
                logger.info(f"Successfully mapped {pt_key} to {tf_key}, shape: {tf_weight.shape}")
            except Exception as e:
                logger.error(f"Error transforming {pt_key}: {e}")
        else:
            # Only log this as debug since we expect many keys not to match
            logger.debug(f"Key {pt_key} not found in state_dict")

    # Check if we have some backup/alternative keys that might match
    if len(weight_map) < 10:  # Very few matches, try alternative naming
        logger.warning("Few keys matched. Trying alternative naming patterns...")
        # Log some of the actual keys for debugging
        sample_keys = all_keys[:20] if len(all_keys) > 20 else all_keys
        logger.info(f"Sample keys from state dict: {sample_keys}")

        # Try a simpler direct mapping approach as fallback
        for pt_key in all_keys:
            if 'backbone' in pt_key or 'decode_head' in pt_key:
                try:
                    # Simplify the TF key to match the general structure
                    tf_key = pt_key.replace('.', '/')
                    weight = state_dict[pt_key]

                    # Apply appropriate transformation based on key pattern
                    if 'weight' in pt_key and len(weight.shape) == 4:  # Conv weights
                        tf_weight = weight.permute(2, 3, 1, 0).numpy()
                    elif 'bias' in pt_key or 'norm' in pt_key:  # Biases and norms
                        tf_weight = weight.numpy()
                    else:
                        tf_weight = weight.numpy()

                    weight_map.append((tf_key, tf_weight))
                    logger.info(f"Fallback mapping: {pt_key} -> {tf_key}, shape: {tf_weight.shape}")
                except Exception as e:
                    logger.error(f"Fallback mapping error for {pt_key}: {e}")

    return weight_map

def apply_weights_to_model(tf_model, weight_map):
    """Apply mapped weights to the TensorFlow model."""
    successfully_applied = 0

    try:
        # Create a dictionary to store layer paths
        layer_dict = {}

        # First collect all top-level layers
        for layer in tf_model.layers:
            layer_dict[layer.name] = layer

            # For nested layers (like in backbone), try to access sublayers directly
            if hasattr(layer, 'layers') and isinstance(layer.layers, list):
                for sublayer in layer.layers:
                    layer_dict[f"{layer.name}/{sublayer.name}"] = sublayer

            # For custom layers with attributes that are layers
            for attr_name in dir(layer):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(layer, attr_name)
                    if isinstance(attr, tf.keras.layers.Layer):
                        layer_dict[f"{layer.name}/{attr_name}"] = attr
                except AttributeError:
                    # Skip attributes that can't be accessed (like input tensors before call)
                    pass

        logger.info(f"Found {len(layer_dict)} layers in the model")

        # Apply weights directly to matched layers
        for tf_path, weight_value in weight_map:
            path_parts = tf_path.split('/')

            # Try different combinations of the path to find matching layers
            for i in range(1, len(path_parts) + 1):
                # Try with progressively longer path prefixes
                current_path = '/'.join(path_parts[:i])
                if current_path in layer_dict:
                    target_layer = layer_dict[current_path]

                    # Try setting weights directly if this is a final layer
                    if i == len(path_parts) or path_parts[i] not in ['kernel', 'bias', 'gamma', 'beta', 'moving_mean', 'moving_variance']:
                        try:
                            # Get current weights to check shape compatibility
                            current_weights = target_layer.get_weights()

                            # Check if any of the current weights match our weight shape
                            for j, current_weight in enumerate(current_weights):
                                if current_weight.shape == weight_value.shape:
                                    # Replace the matching weight and set all weights
                                    new_weights = list(current_weights)
                                    new_weights[j] = weight_value

                                    target_layer.set_weights(new_weights)
                                    successfully_applied += 1
                                    logger.info(f"Applied weight to {current_path} (position {j}), shape: {weight_value.shape}")
                                    break
                        except Exception as e:
                            logger.debug(f"Could not set weights directly for {current_path}: {e}")
                            # Continue to try other approaches

                    # If this is a layer with specific weight parts (kernel, bias, etc.)
                    elif i < len(path_parts):
                        weight_type = path_parts[i]
                        try:
                            current_weights = target_layer.get_weights()

                            # Determine which position in weights corresponds to the weight type
                            position = -1
                            if weight_type in ['kernel', 'depthwise_kernel'] and len(current_weights) > 0:
                                position = 0
                            elif weight_type == 'bias' and len(current_weights) > 1:
                                position = 1
                            elif weight_type == 'gamma' and len(current_weights) >= 1:
                                position = 0
                            elif weight_type == 'beta' and len(current_weights) >= 2:
                                position = 1
                            elif weight_type == 'moving_mean' and len(current_weights) >= 3:
                                position = 2
                            elif weight_type == 'moving_variance' and len(current_weights) >= 4:
                                position = 3

                            if position >= 0 and position < len(current_weights):
                                # Verify shapes match
                                if current_weights[position].shape == weight_value.shape:
                                    new_weights = list(current_weights)
                                    new_weights[position] = weight_value

                                    target_layer.set_weights(new_weights)
                                    successfully_applied += 1
                                    logger.info(f"Applied {weight_type} to {current_path}, shape: {weight_value.shape}")
                                    break
                                else:
                                    logger.debug(f"Shape mismatch for {current_path}/{weight_type}: "
                                               f"expected {current_weights[position].shape}, got {weight_value.shape}")
                        except Exception as e:
                            logger.debug(f"Could not set {weight_type} for {current_path}: {e}")

        # Check the result
        if successfully_applied == 0:
            logger.warning("No weights were successfully applied to the model!")
        else:
            logger.info(f"Successfully applied {successfully_applied} weights to the model")

        return successfully_applied > 0

    except Exception as e:
        logger.error(f"Failed to apply weights: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def verify_weights(tf_model, sample_input):
    """Verify model with converted weights produces reasonable outputs."""
    try:
        output = tf_model(sample_input, training=False)
        logger.info(f"Verification output shape: {output.shape}")
        logger.info(f"Output stats - min: {tf.reduce_min(output).numpy()}, max: {tf.reduce_max(output).numpy()}")
        if tf.reduce_any(tf.math.is_nan(output)) or tf.reduce_any(tf.math.is_inf(output)):
            raise ValueError("Model output contains NaN or Inf values")
        logger.info("Weight verification passed")
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise

def convert_pretrained_weights(pth_path, output_path, input_shape=(384, 384, 1), num_classes=5):
    """Main function to convert PyTorch weights to TensorFlow."""
    logger.info("Starting weight conversion process")

    # Initialize TensorFlow model
    try:
        tf_model = OilSpillSegformer(
            input_shape=input_shape,
            num_classes=num_classes,
            drop_rate=0.1,
            use_cbam=True,  # Match your train.py configuration
            pretrained_weights=None
        )
        logger.info("Initialized OilSpillSegformer model")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    # Create dummy input for verification
    sample_input = tf.random.normal((1, *input_shape), dtype=tf.float32)

    # Load PyTorch weights
    state_dict = load_pytorch_weights(pth_path)

    # Map weights
    weight_map = map_pytorch_to_tensorflow(state_dict, tf_model)

    # Apply weights
    apply_weights_to_model(tf_model, weight_map)

    # Verify model
    verify_weights(tf_model, sample_input)

    # Save weights
    try:
        tf_model.save_weights(output_path)
        logger.info(f"Saved converted weights to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save weights: {e}")
        raise

if __name__ == "__main__":
    pth_path = "pretrained_weights/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth"
    output_path = "pretrained_weights/segformer_b2_pretrain.weights.h5"  # Changed to end with .weights.h5
    convert_pretrained_weights(pth_path, output_path)
