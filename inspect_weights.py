import h5py
import os
import numpy as np

def inspect_h5_file(file_path):
    print(f"Inspecting H5 file: {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Top-level keys: {list(f.keys())}")

            # Function to recursively explore groups
            def explore_group(group, prefix=''):
                print(f"\n{prefix}Group {group.name}:")
                for key, item in group.items():
                    if isinstance(item, h5py.Group):
                        explore_group(item, prefix + '  ')
                    elif isinstance(item, h5py.Dataset):
                        print(f"{prefix}  Dataset: {key}, Shape: {item.shape}, Dtype: {item.dtype}")
                        # For small arrays, print a sample
                        if item.size < 10:
                            print(f"{prefix}    Value: {item[...]}")

            # Explore the root groups
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    explore_group(f[key])
                elif isinstance(f[key], h5py.Dataset):
                    print(f"\nDataset {key}: Shape: {f[key].shape}, Dtype: {f[key].dtype}")
                    if f[key].size < 10:
                        print(f"  Value: {f[key][...]}")

            # Check for attributes in the file
            if f.attrs:
                print("\nFile attributes:")
                for key, value in f.attrs.items():
                    print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error inspecting H5 file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Inspect the pretrained weights file
    weights_path = 'pretrained_weights/segformer_b2_pretrain.weights.h5'
    inspect_h5_file(weights_path)

    # Also check if there are other weight files available
    pth_path = 'pretrained_weights/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth'
    if os.path.exists(pth_path):
        print(f"\nFound PyTorch weight file: {pth_path}")
        print("This file would need conversion using the weight_converter.py script")
