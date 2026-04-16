from typing import Mapping

import torch

NUM_CLASSES = 5
SHIP_CLASS_IDX = 3

CLASS_NAMES = ["sea_surface", "oil_spill", "look_alike", "ship", "land"]

CLASS_COLORS_RGB = [
    [0, 0, 0],  
    [0, 255, 255],  
    [255, 0, 0],  
    [153, 76, 0],  
    [0, 153, 0],  
]

CLASS_COLORS_RGB_DOC_ALTERNATE = {
    "ship": [139, 69, 19],
    "land": [0, 128, 0],
}

DEFAULT_CLASS_PIXEL_COUNTS = {
    "sea_surface": 797_700_000,
    "oil_spill": 9_100_000,
    "look_alike": 50_400_000,
    "ship": 300_000,
    "land": 45_700_000,
}

DEFAULT_PIXEL_COUNTS = [DEFAULT_CLASS_PIXEL_COUNTS[name] for name in CLASS_NAMES]
SHIP_CLASS_WEIGHT_CAP = 15.0


def compute_median_freq_weights(pixel_counts: Mapping[str, int]) -> torch.Tensor:
    missing = [name for name in CLASS_NAMES if name not in pixel_counts]
    if missing:
        raise ValueError(f"Missing pixel counts for classes: {missing}")

    counts = torch.tensor(
        [float(pixel_counts[name]) for name in CLASS_NAMES],
        dtype=torch.float32,
    )
    total = counts.sum().clamp(min=1.0)
    frequencies = counts / total
    median_frequency = frequencies.median()
    weights = median_frequency / frequencies.clamp(min=1e-12)
    weights = weights.clamp(max=SHIP_CLASS_WEIGHT_CAP)
    return weights.to(dtype=torch.float32)


DEFAULT_CLASS_WEIGHTS = compute_median_freq_weights(DEFAULT_CLASS_PIXEL_COUNTS)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
