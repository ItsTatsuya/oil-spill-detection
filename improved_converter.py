import os
import numpy as np
import tensorflow as tf
import h5py
import re

def create_improved_weight_mapper():
    """
    Creates an improved weight mapping function that better handles the naming differences
    between the saved weights and the model layer names.
    """

    def map_weight_name(h5_name):
        """
        Maps a name from the h5 file to a name pattern that matches the model structure.

        Args:
            h5_name: Original name from the h5 file

        Returns:
            List of possible matching patterns in the model
        """
        # Remove common prefixes and paths
        name = h5_name.replace('layers/', '')

        # Extract components from the name
        patterns = []

        # Handle special cases

        # CBAM attention layers
        if any(name.startswith(p) for p in ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3']):
            cbam_parts = name.split('/')
            layer_name = cbam_parts[0]
            patterns.append(layer_name)
            return patterns

        # Specific handling for decoder head
        if 'seg_former_head' in name:
            decoder_name = name.replace('seg_former_head/', 'decoder_head/')
            patterns.append(decoder_name)
            # Also try without the path
            if 'linear_c' in name or 'batch_norm' in name or 'classifier' in name:
                component = name.split('/')[-2]
                patterns.append(f"decoder_head/{component}")
            return patterns

        # Backbone blocks (most complex part)
        if 'mix_vision_transformer' in name:
            # Extract the relevant parts
            name = name.replace('mix_vision_transformer/', '')

            if 'patch_embed' in name:
                embed_num = re.search(r'patch_embed(\d+)', name).group(1)
                patterns.append(f"backbone/patch_embed{embed_num}")
                var_part = name.split('/')[-2]
                patterns.append(f"backbone/patch_embed{embed_num}/{var_part}")
                return patterns

            if 'norm' in name and len(name.split('/')) <= 2:
                norm_num = re.search(r'norm(\d+)', name).group(1)
                patterns.append(f"backbone/norm{norm_num}")
                return patterns

            # Handle block structure
            if any(f"block{i}" in name for i in range(1, 5)):
                for i in range(1, 5):
                    if f"block{i}" in name:
                        block_name = f"block{i}"
                        rest = name.replace(f"{block_name}/", "")

                        # Extract block layer indices
                        layer_match = re.search(r'transformer_encoder_layer_?(\d*)', rest)
                        layer_idx = layer_match.group(1) if layer_match and layer_match.group(1) else "0"

                        # Create pattern with correct block/layer indexing
                        block_pattern = f"backbone/{block_name}_{layer_idx}"
                        patterns.append(block_pattern)

                        # Handle components within blocks
                        if 'attn' in rest:
                            attn_part = rest.split('attn/')[-1].split('/')[0]
                            patterns.append(f"{block_pattern}/attn/{attn_part}")

                        if 'ffn' in rest:
                            ffn_part = rest.split('ffn/')[-1].split('/')[0]
                            patterns.append(f"{block_pattern}/ffn/{ffn_part}")

                        if 'norm1' in rest:
                            patterns.append(f"{block_pattern}/norm1")

                        if 'norm2' in rest:
                            patterns.append(f"{block_pattern}/norm2")

                        # If it's a direct tensor
                        if 'vars' in rest:
                            var_idx = int(rest.split('vars/')[-1])
                            patterns.append((block_pattern, var_idx))

                        break

                return patterns

        # Default pattern - return as is for direct matching
        patterns.append(name)
        return patterns

    def map_weights(model, weights_path):
        """
        Load weights from an h5 file into the model with a more flexible mapping approach.

        Args:
            model: The model to load weights into
            weights_path: Path to the h5 weights file

        Returns:
            True if successful, False otherwise
        """
        print(f"Loading weights with improved mapping from: {weights_path}")

        if not os.path.exists(weights_path):
            print(f"Error: Weight file not found at {weights_path}")
            return False

        # Build a model layer registry for quick lookups
        layer_registry = {}

        def register_layer(layer, prefix=""):
            if hasattr(layer, 'name'):
                layer_path = f"{prefix}/{layer.name}" if prefix else layer.name
                layer_registry[layer_path] = layer

                # Register additional common aliases
                layer_registry[layer.name] = layer  # Direct name
                if '/' in layer_path:
                    layer_registry[layer_path.split('/')[-1]] = layer  # Just the last component

                # Check for sublayers
                if hasattr(layer, 'layers') and layer.layers:
                    for sublayer in layer.layers:
                        register_layer(sublayer, layer_path)

                # Check for blocks in backbone
                if layer.name == 'backbone':
                    # Register special nested structures in backbone
                    for block_attr in ['block1', 'block2', 'block3', 'block4']:
                        if hasattr(layer, block_attr):
                            blocks = getattr(layer, block_attr)
                            for i, block in enumerate(blocks):
                                block_path = f"{layer_path}/{block_attr}_{i}"
                                layer_registry[block_path] = block

                                # Register parts of the transformer encoder layer
                                for part in ['attn', 'ffn', 'norm1', 'norm2']:
                                    if hasattr(block, part):
                                        part_layer = getattr(block, part)
                                        part_path = f"{block_path}/{part}"
                                        layer_registry[part_path] = part_layer

                                        # For deeper parts like q, k, v in attention
                                        if part in ['attn', 'ffn'] and hasattr(part_layer, 'layers'):
                                            for subpart in part_layer.layers:
                                                subpart_path = f"{part_path}/{subpart.name}"
                                                layer_registry[subpart_path] = subpart

                # Register decoder head components
                if layer.name == 'decoder_head':
                    for head_part in ['linear_c1', 'linear_c2', 'linear_c3', 'linear_c4',
                                      'linear_fuse', 'batch_norm', 'classifier']:
                        if hasattr(layer, head_part):
                            part_layer = getattr(layer, head_part)
                            part_path = f"{layer_path}/{head_part}"
                            layer_registry[part_path] = part_layer

        # Build the layer registry
        for layer in model.layers:
            register_layer(layer)

        print(f"Built layer registry with {len(layer_registry)} entries")

        try:
            with h5py.File(weights_path, 'r') as h5_file:
                weights_loaded = 0
                weights_attempted = 0

                # Helper to recursively explore groups and datasets
                def process_group(group, path=""):
                    nonlocal weights_loaded, weights_attempted

                    for name, item in group.items():
                        item_path = f"{path}/{name}" if path else name

                        if isinstance(item, h5py.Group):
                            # If it's a group, recurse
                            process_group(item, item_path)
                        elif isinstance(item, h5py.Dataset) and 'vars' in item_path:
                            # Only process actual weight tensors
                            weights_attempted += 1

                            # Get parent path (the layer this belongs to)
                            parent_path = '/'.join(item_path.split('/')[:-2])  # Remove /vars/X

                            # Get the variable index
                            var_idx = int(item_path.split('/')[-1])

                            # Map the name to potential model paths
                            potential_paths = map_weight_name(parent_path)

                            weight_matched = False
                            weight_value = item[...]

                            # Try each potential path
                            for path_pattern in potential_paths:
                                if isinstance(path_pattern, tuple):
                                    # Handle special case where we have a direct (layer, idx) mapping
                                    layer_path, idx = path_pattern
                                    if layer_path in layer_registry:
                                        layer = layer_registry[layer_path]
                                        if idx < len(layer.weights):
                                            # Check shape compatibility
                                            if layer.weights[idx].shape == weight_value.shape:
                                                try:
                                                    # Use assign to set the weight value
                                                    layer.weights[idx].assign(weight_value)
                                                    weights_loaded += 1
                                                    weight_matched = True
                                                    break
                                                except Exception as e:
                                                    print(f"Failed to assign weight for {layer_path}: {e}")

                                elif path_pattern in layer_registry:
                                    layer = layer_registry[path_pattern]
                                    if var_idx < len(layer.weights):
                                        # Check shape compatibility
                                        if layer.weights[var_idx].shape == weight_value.shape:
                                            try:
                                                # Use assign to set the weight value
                                                layer.weights[var_idx].assign(weight_value)
                                                weights_loaded += 1
                                                weight_matched = True
                                                break
                                            except Exception as e:
                                                print(f"Failed to assign weight for {path_pattern}: {e}")

                            if not weight_matched and weights_attempted < 20:
                                print(f"No match found for {parent_path} with shape {weight_value.shape}")

                # Start processing from the root group
                process_group(h5_file)

                print(f"Successfully loaded {weights_loaded}/{weights_attempted} weights")

                # Check if we loaded a reasonable number of weights
                if weights_loaded > 0:
                    return True
                else:
                    print("No weights were successfully loaded")
                    return False

        except Exception as e:
            print(f"Error loading weights: {e}")
            import traceback
            traceback.print_exc()
            return False

    return map_weights

# Modified to update the train.py file to use the improved weight mapper
def update_train_py_with_improved_loader():
    """
    Update the weight loading logic in train.py to use the improved weight mapper.
    This function replaces the create_pretrained_weight_loader import with the improved version.
    """
    # Create the improved weight mapper to replace create_pretrained_weight_loader
    # Then update the train.py file to import and use it

    return create_improved_weight_mapper
