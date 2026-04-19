import logging
import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from constants import SHIP_CLASS_IDX
from utils.config import resolve_ship_library_path

logger = logging.getLogger(__name__)

SEA_SURFACE_CLASS_IDX = 0
OIL_SPILL_CLASS_IDX = 1
LOOK_ALIKE_CLASS_IDX = 2
LAND_CLASS_IDX = 4


class ShipCrop:

    def __init__(
        self,
        image_patch: np.ndarray,
        mask_patch: np.ndarray,
        source_path: str,
        bbox: Tuple[int, int, int, int],  
        pixel_count: int,
    ) -> None:
        self.image_patch = image_patch  
        self.mask_patch = mask_patch  
        self.source_path = source_path
        self.bbox = bbox
        self.pixel_count = pixel_count


class CopyPasteAugmentation:

    _LIBRARY_VERSION = 5

    def __init__(self, config: dict, library_path: Optional[str] = None) -> None:
        self.config = config
        copy_paste_cfg = (
            config.get("augmentation", {}).get("train", {}).get("copy_paste", {})
        )

        self.paste_prob = copy_paste_cfg.get("paste_prob", 0.5)
        self.max_ships = copy_paste_cfg.get("max_ships_per_image", 3)
        self.blur_boundary = copy_paste_cfg.get("blur_boundary", True)
        self.blur_kernel = copy_paste_cfg.get("blur_kernel_size", 3)
        allowed_classes = copy_paste_cfg.get(
            "allowed_target_classes", ["sea_surface", "oil_spill"]
        )
        self.allowed_target_classes = self._resolve_allowed_target_classes(allowed_classes)
        self.min_allowed_target_fraction = float(
            copy_paste_cfg.get("min_allowed_target_fraction", 0.85)
        )
        self.match_local_intensity = bool(
            copy_paste_cfg.get("match_local_intensity", False)
        )
        self.ignore_index = int(config.get("loss", {}).get("ce_ignore_index", -100))

        lib_path = library_path or resolve_ship_library_path(config)
        self.library_path = Path(lib_path)
        self.library_path.mkdir(parents=True, exist_ok=True)

        self._library: List[ShipCrop] = []
        self._library_built = False

    def _resolve_allowed_target_classes(self, values: List[str | int]) -> set[int]:
        name_to_idx = {
            "sea_surface": SEA_SURFACE_CLASS_IDX,
            "oil_spill": OIL_SPILL_CLASS_IDX,
            "look_alike": LOOK_ALIKE_CLASS_IDX,
            "ship": SHIP_CLASS_IDX,
            "land": LAND_CLASS_IDX,
        }
        resolved: set[int] = set()
        for value in values:
            if isinstance(value, str):
                if value not in name_to_idx:
                    raise ValueError(f"Unknown copy-paste target class: {value}")
                resolved.add(int(name_to_idx[value]))
            else:
                resolved.add(int(value))
        if not resolved:
            raise ValueError("copy_paste.allowed_target_classes must not be empty")
        return resolved

    def build_ship_library(self, dataset: Any) -> None:
        library_file = self.library_path / f"ship_library_v{self._LIBRARY_VERSION}.pkl"
        logger.info(f"Using ship library path: {self.library_path}")

        if library_file.exists():
            logger.info(f"Loading existing ship library from {library_file}")
            self._load_library(library_file)
            return

        logger.info("Building ship crop library from training data...")

        ship_crops = []
        total_ships = 0
        sizes = []

        pbar = tqdm(
            range(len(dataset)),
            total=len(dataset),
            desc="Ship library",
            leave=False,
        )
        for idx in pbar:
            meta = dataset.metadata[idx]
            if not meta["has_ship"]:
                continue

            img_path = dataset.image_paths[idx]
            raw_image = dataset.load_raw_image(idx)
            mask = dataset._load_mask(idx)

            ship_binary = (mask == SHIP_CLASS_IDX).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                ship_binary, connectivity=8
            )

            for label_id in range(1, num_labels):  
                component_mask = labels == label_id
                pixel_count = int(component_mask.sum())

                if pixel_count < 3:  
                    continue

                rows = np.where(component_mask.any(axis=1))[0]
                cols = np.where(component_mask.any(axis=0))[0]
                y1, y2 = rows.min(), rows.max() + 1
                x1, x2 = cols.min(), cols.max() + 1

                pad = 5
                H, W = mask.shape
                y1p = max(0, y1 - pad)
                y2p = min(H, y2 + pad)
                x1p = max(0, x1 - pad)
                x2p = min(W, x2 + pad)

                crop_img = raw_image[y1p:y2p, x1p:x2p].copy()
                crop_mask = component_mask[y1p:y2p, x1p:x2p].astype(np.int64)

                ship_crop = ShipCrop(
                    image_patch=crop_img,
                    mask_patch=crop_mask,
                    source_path=str(img_path),
                    bbox=(y1p, x1p, y2p, x2p),
                    pixel_count=pixel_count,
                )
                ship_crops.append(ship_crop)
                total_ships += 1
                sizes.append(pixel_count)
                pbar.set_postfix(ships=total_ships)

        pbar.close()

        self._library = ship_crops
        self._library_built = True

        if sizes:
            logger.info(f"Ship library built: {total_ships} instances")
            logger.info(
                f"  Size distribution: min={min(sizes)}, "
                f"median={np.median(sizes):.0f}, max={max(sizes)}"
            )
        else:
            logger.warning("No ship instances found in training data!")

        self._save_library(library_file)

    def _save_library(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._library, f)
        logger.info(f"Ship library saved to {path}")

    def _load_library(self, path: Path) -> None:
        if path.suffix.lower() != ".pkl":
            raise ValueError(f"Expected a .pkl ship library file, got: {path}")
        logger.warning(
            "Loading ship library via pickle. Only load trusted files to avoid "
            "arbitrary code execution risks."
        )
        with open(path, "rb") as f:
            self._library = pickle.load(f)
        invalid_patches = [
            idx
            for idx, crop in enumerate(self._library)
            if getattr(crop, "image_patch", None) is None
            or np.asarray(crop.image_patch).ndim != 2
        ]
        if invalid_patches:
            raise RuntimeError(
                "Ship library contains incompatible non-amplitude crops. "
                f"Rebuild the library at {path.parent} for raw-amplitude copy-paste."
            )
        self._library_built = True
        logger.info(f"Ship library loaded: {len(self._library)} instances")

    def apply(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._library_built or len(self._library) == 0:
            return image, mask

        if np.random.random() > self.paste_prob:
            return image, mask

        H, W = mask.shape
        if image.ndim != 2:
            raise ValueError(
                "Copy-paste expects raw SAR amplitude images with shape (H, W). "
                f"Got {image.shape}."
            )
        n_ships = np.random.randint(1, self.max_ships + 1)

        indices = np.random.choice(len(self._library), size=n_ships, replace=True)

        aug_image = image.copy().astype(np.float32)
        if aug_image.max() > 1.0:
            aug_image = aug_image / 255.0
        aug_mask = mask.copy()

        for idx in indices:
            crop = self._library[idx]
            ch, cw = crop.image_patch.shape[:2]
            patch_image = crop.image_patch

            if ch >= H or cw >= W:
                continue
            if patch_image.ndim != 2:
                continue

            paste_y, paste_x = self._find_paste_location(
                aug_mask,
                crop.mask_patch.astype(bool),
                ch,
                cw,
                H,
                W,
            )
            if paste_y is None:
                continue

            py1, px1 = paste_y, paste_x
            py2, px2 = paste_y + ch, paste_x + cw

            ship_pixels = crop.mask_patch.astype(bool)
            patch_to_paste = patch_image
            if self.match_local_intensity:
                patch_to_paste = self._match_local_patch_statistics(
                    patch_image,
                    aug_image[py1:py2, px1:px2],
                )

            if self.blur_boundary and self.blur_kernel > 0:
                aug_image = self._paste_with_blur(
                    aug_image,
                    patch_to_paste,
                    ship_pixels,
                    py1,
                    px1,
                    py2,
                    px2,
                    self.blur_kernel,
                )
            else:
                aug_image[py1:py2, px1:px2][ship_pixels] = patch_to_paste[ship_pixels]

            aug_mask[py1:py2, px1:px2][ship_pixels] = SHIP_CLASS_IDX

        if image.dtype == np.uint8:
            aug_image = (aug_image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            aug_image = aug_image.astype(np.float32)

        return aug_image, aug_mask

    def _find_paste_location(
        self,
        mask: np.ndarray,
        ship_mask: np.ndarray,
        ch: int,
        cw: int,
        H: int,
        W: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        allowed_target_mask = np.isin(mask, list(self.allowed_target_classes))
        for _ in range(50):  
            max_y = max(H - ch, 0)
            max_x = max(W - cw, 0)
            y = 0 if max_y == 0 else int(np.random.randint(0, max_y + 1))
            x = 0 if max_x == 0 else int(np.random.randint(0, max_x + 1))

            patch_mask = mask[y : y + ch, x : x + cw]
            target_pixels = patch_mask[ship_mask]
            if target_pixels.size == 0:
                continue
            if np.any(target_pixels == self.ignore_index):
                continue
            if np.any(target_pixels == LAND_CLASS_IDX):
                continue
            if np.any(target_pixels == SHIP_CLASS_IDX):
                continue
            if (
                LOOK_ALIKE_CLASS_IDX not in self.allowed_target_classes
                and np.any(target_pixels == LOOK_ALIKE_CLASS_IDX)
            ):
                continue
            allowed_fraction = float(
                allowed_target_mask[y : y + ch, x : x + cw][ship_mask].mean()
            )
            if allowed_fraction < self.min_allowed_target_fraction:
                continue
            return y, x

        return None, None

    def _paste_with_blur(
        self,
        image: np.ndarray,
        patch: np.ndarray,
        ship_mask: np.ndarray,
        y1: int,
        x1: int,
        y2: int,
        x2: int,
        kernel_size: int,
    ) -> np.ndarray:
        ship_mask_float = ship_mask.astype(np.float32)
        blend_weight = cv2.GaussianBlur(
            ship_mask_float,
            (kernel_size * 2 + 1, kernel_size * 2 + 1),
            sigmaX=kernel_size / 2,
        )
        if image.ndim == 3:
            blend_weight = blend_weight[:, :, np.newaxis]  

        blend_weight = blend_weight.clip(0.0, 1.0)

        region = image[y1:y2, x1:x2].astype(np.float32)
        patch_float = patch.astype(np.float32)
        if patch_float.max() > 1.0:
            patch_float = patch_float / 255.0

        blended = blend_weight * patch_float + (1 - blend_weight) * region
        image[y1:y2, x1:x2] = blended

        return image

    def _match_local_patch_statistics(
        self,
        patch: np.ndarray,
        target_region: np.ndarray,
    ) -> np.ndarray:
        patch_float = patch.astype(np.float32, copy=False)
        target_float = target_region.astype(np.float32, copy=False)
        if patch_float.max() > 1.0:
            patch_float = patch_float / 255.0
        if target_float.max() > 1.0:
            target_float = target_float / 255.0

        if patch_float.ndim == 2:
            patch_mean = float(patch_float.mean())
            patch_std = float(patch_float.std())
            target_mean = float(target_float.mean())
            target_std = float(target_float.std())
            matched = (patch_float - patch_mean) / max(patch_std, 1e-6)
            matched = matched * max(target_std, 1e-6) + target_mean
            return matched.clip(0.0, 1.0).astype(np.float32)

        patch_mean = patch_float.mean(axis=(0, 1), keepdims=True)
        patch_std = patch_float.std(axis=(0, 1), keepdims=True)
        target_mean = target_float.mean(axis=(0, 1), keepdims=True)
        target_std = target_float.std(axis=(0, 1), keepdims=True)
        matched = (patch_float - patch_mean) / np.maximum(patch_std, 1e-6)
        matched = matched * np.maximum(target_std, 1e-6) + target_mean
        return matched.clip(0.0, 1.0).astype(np.float32)

