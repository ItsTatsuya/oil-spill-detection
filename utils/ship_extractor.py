import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from constants import SHIP_CLASS_IDX

logger = logging.getLogger(__name__)


def _normalize_crop_source_image(image: np.ndarray) -> np.ndarray:
    img_float = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        max_value = float(np.iinfo(image.dtype).max)
        if max_value > 1.0:
            img_float = img_float / max_value
    return img_float


def extract_ship_crops_from_image(
    image: np.ndarray,
    mask: np.ndarray,
    min_pixel_count: int = 3,
    padding: int = 5,
) -> List[Dict]:
    H, W = mask.shape
    ship_binary = (mask == SHIP_CLASS_IDX).astype(np.uint8)

    if ship_binary.sum() == 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ship_binary, connectivity=8
    )

    crops = []
    img_float = _normalize_crop_source_image(image)

    for label_id in range(1, num_labels):  
        component = labels == label_id
        pixel_count = int(component.sum())

        if pixel_count < min_pixel_count:
            continue

        rows = np.where(component.any(axis=1))[0]
        cols = np.where(component.any(axis=0))[0]
        y1, y2 = int(rows.min()), int(rows.max()) + 1
        x1, x2 = int(cols.min()), int(cols.max()) + 1

        y1p = max(0, y1 - padding)
        y2p = min(H, y2 + padding)
        x1p = max(0, x1 - padding)
        x2p = min(W, x2 + padding)

        crops.append(
            {
                "image_patch": img_float[y1p:y2p, x1p:x2p].copy(),
                "mask_patch": component[y1p:y2p, x1p:x2p].copy(),
                "bbox": (y1p, x1p, y2p, x2p),
                "pixel_count": pixel_count,
            }
        )

    return crops


def compute_dataset_ship_statistics(
    dataset_root: str,
    split: str = "train",
) -> Dict:
    labels_1d_dir = Path(dataset_root) / "annotations"
    if not labels_1d_dir.exists():
        logger.warning(f"Labels directory not found: {labels_1d_dir}")
        return {}

    total_ship_images = 0
    total_instances = 0
    sizes = []

    for mask_path in sorted(labels_1d_dir.glob("*.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        ship_binary = (mask == SHIP_CLASS_IDX).astype(np.uint8)
        if ship_binary.sum() == 0:
            continue

        total_ship_images += 1
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            ship_binary, connectivity=8
        )

        for label_id in range(1, num_labels):
            size = int((labels == label_id).sum())
            if size >= 3:
                total_instances += 1
                sizes.append(size)

    return {
        "total_ship_images": total_ship_images,
        "total_ship_instances": total_instances,
        "size_distribution": sizes,
    }
