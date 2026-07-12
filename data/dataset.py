import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from constants import (
    CLASS_COLORS_RGB,
    CLASS_COLORS_RGB_DOC_ALTERNATE,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    SHIP_CLASS_IDX,
)
from data.augmentation import (
    ClassAwareCropper,
    SARSegmentationAugmentation,
    get_config_normalize_stats,
)
from utils.config import resolve_model_num_channels, resolve_sar_feature_cache_dir
from utils.config import resolve_sar_channel_names

logger = logging.getLogger(__name__)

CLASS_COLORS = np.array(CLASS_COLORS_RGB, dtype=np.float32)

OIL_SPILL_IDX = CLASS_NAMES.index("oil_spill")  
LOOK_ALIKE_IDX = CLASS_NAMES.index("look_alike")  


def color_to_class(color: np.ndarray, class_colors: Optional[np.ndarray] = None) -> int:
    color_table = CLASS_COLORS if class_colors is None else class_colors
    pixel = color.astype(np.float32).reshape(1, 3)
    diff = pixel[:, np.newaxis, :] - color_table[np.newaxis, :, :]  
    dist = (diff**2).sum(axis=2)  
    return int(dist.argmin())


def rgb_mask_to_class_index(
    rgb_mask: np.ndarray,
    class_colors: Optional[np.ndarray] = None,
) -> np.ndarray:
    assert rgb_mask.ndim == 3 and rgb_mask.shape[2] == 3, (
        f"Expected (H, W, 3) RGB mask, got {rgb_mask.shape}"
    )

    H, W, _ = rgb_mask.shape
    color_table = CLASS_COLORS if class_colors is None else class_colors
    pixels = rgb_mask.reshape(-1, 3).astype(np.float32)

    diff = pixels[:, np.newaxis, :] - color_table[np.newaxis, :, :]  
    dist = (diff**2).sum(axis=2)  

    class_indices = dist.argmin(axis=1).astype(np.int64)  
    return class_indices.reshape(H, W)


def remap_ignore_labels(
    mask: np.ndarray,
    *,
    ignore_index: int,
    mask_ignore_values: tuple[int, ...],
) -> np.ndarray:
    remapped = np.asarray(mask, dtype=np.int64).copy()
    for raw_value in mask_ignore_values:
        remapped[remapped == int(raw_value)] = ignore_index
    valid = (remapped == ignore_index) | ((remapped >= 0) & (remapped < NUM_CLASSES))
    if not np.all(valid):
        invalid_values = np.unique(remapped[~valid]).tolist()
        raise ValueError(
            f"Mask contains unsupported class indices: {invalid_values}. "
            f"Expected 0..{NUM_CLASSES - 1} or ignore values {mask_ignore_values}."
        )
    return np.ascontiguousarray(remapped)


class OilSpillDataset(Dataset):

    @staticmethod
    def color_to_class(color: np.ndarray) -> int:
        return color_to_class(color)

    def __init__(
        self,
        root: str,
        split: str,
        config: dict,
        transform: Optional[Callable] = None,
        sar_encoder: Optional[Any] = None,
    ) -> None:
        assert split in ("train", "test"), (
            f"split must be 'train' or 'test', got '{split}'"
        )

        self.root = Path(root)
        self.split = split
        self.config = config
        self.transform = transform
        self.sar_encoder = sar_encoder
        self.copy_paste = None  
        self.class_aware_cropper = ClassAwareCropper(config)
        self.enable_train_augmentations = split == "train"
        self.train_intensity_transform = (
            SARSegmentationAugmentation(config).get_train_intensity_transform()
            if self.enable_train_augmentations
            else None
        )
        self.current_epoch = 1
        dataset_cfg = config.get("dataset", {})
        self.class_colors = self._resolve_class_colors(dataset_cfg)
        self.allow_missing_masks = bool(dataset_cfg.get("allow_missing_masks", False))
        self.ignore_index = int(config.get("loss", {}).get("ce_ignore_index", -100))
        raw_ignore_values = dataset_cfg.get("mask_ignore_values")
        if raw_ignore_values is None:
            raw_ignore_values = dataset_cfg.get("mask_ignore_value")
        if raw_ignore_values is None:
            raw_ignore_values = [255] if self.ignore_index < 0 else [self.ignore_index]
        elif not isinstance(raw_ignore_values, (list, tuple, set)):
            raw_ignore_values = [raw_ignore_values]
        self.mask_ignore_values = tuple(int(value) for value in raw_ignore_values)
        self.feature_cache_enabled = bool(
            config.get("sar_features", {}).get("cache", {}).get("enabled", False)
        )
        self.feature_cache_dir = (
            resolve_sar_feature_cache_dir(config)
            if self.feature_cache_enabled
            else None
        )
        train_aug_cfg = config.get("augmentation", {}).get("train", {})
        self.crop_size = int(train_aug_cfg.get("crop_size", 512))

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root}\n"
                "Please set dataset.root in your config file, for example "
                "configs/main-config.yaml, "
                "to the path of the Krestenitis 2019 oil spill dataset."
            )

        self.split_dir = self.root / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"
        self.labels_1d_dir = self.split_dir / "labels_1D"

        image_candidates = list(self.images_dir.glob("*.jpg"))
        image_candidates += list(self.images_dir.glob("*.png"))
        image_candidates += list(self.images_dir.glob("*.JPG"))
        self.image_paths = sorted(dict.fromkeys(image_candidates))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        self.label_paths: List[Optional[Path]] = []
        self.label_1d_paths: List[Optional[Path]] = []
        self.metadata: List[Dict[str, Any]] = []

        for img_path in self.image_paths:
            stem = img_path.stem

            lbl = self.labels_dir / f"{stem}.png"
            if not lbl.exists():
                lbl = None

            lbl_1d = self.labels_1d_dir / f"{stem}.png"
            if not lbl_1d.exists():
                lbl_1d = None

            self.label_paths.append(lbl)
            self.label_1d_paths.append(lbl_1d)

        if self.split == "train":
            self._log_label_color_diagnostics(sample_size=5)

        if not self.allow_missing_masks:
            missing_mask_images = [
                str(img_path)
                for img_path, lbl, lbl_1d in zip(
                    self.image_paths, self.label_paths, self.label_1d_paths
                )
                if lbl is None and lbl_1d is None
            ]
            if missing_mask_images:
                preview = "\n".join(missing_mask_images[:5])
                extra = ""
                if len(missing_mask_images) > 5:
                    extra = f"\n... and {len(missing_mask_images) - 5} more"
                raise FileNotFoundError(
                    f"Missing segmentation masks for {len(missing_mask_images)} "
                    f"{split} images. Example paths:\n{preview}{extra}\n"
                    "Set dataset.allow_missing_masks=true only for workflows that "
                    "do not require supervision."
                )

        logger.info(
            f"Scanning {split} set for class metadata ({len(self.image_paths)} images)..."
        )
        self._compute_metadata()

        logger.info(
            f"OilSpillDataset [{split}]: {len(self)} images. "
            f"With ships: {sum(m['has_ship'] for m in self.metadata)}, "
            f"With oil: {sum(m['has_oil_spill'] for m in self.metadata)}, "
            f"With look-alike: {sum(m['has_look_alike'] for m in self.metadata)}"
        )

    def _resolve_class_colors(self, dataset_cfg: dict) -> np.ndarray:
        base_colors = np.array(CLASS_COLORS_RGB, dtype=np.float32)
        override = dataset_cfg.get("label_color_override")
        if not override:
            return base_colors

        if not isinstance(override, dict):
            raise ValueError(
                "dataset.label_color_override must be null or a dict of class_name -> [R, G, B]."
            )

        resolved = base_colors.copy()
        for idx, class_name in enumerate(CLASS_NAMES):
            if class_name not in override:
                continue
            color = override[class_name]
            if not isinstance(color, (list, tuple)) or len(color) != 3:
                raise ValueError(
                    f"dataset.label_color_override['{class_name}'] must be [R, G, B]."
                )
            resolved[idx] = np.array(color, dtype=np.float32)

        logger.debug(
            "Using dataset.label_color_override for RGB->class mapping: %s",
            {
                class_name: resolved[idx].astype(int).tolist()
                for idx, class_name in enumerate(CLASS_NAMES)
            },
        )
        return resolved

    def _log_label_color_diagnostics(self, sample_size: int = 5) -> None:
        rgb_label_paths = [path for path in self.label_paths if path is not None]
        if not rgb_label_paths:
            logger.warning(
                "No RGB label files found in %s; skipping label color convention diagnostics.",
                self.labels_dir,
            )
            return

        seed = int(self.config.get("dataset", {}).get("split_seed", 42)) + 17
        rng = np.random.default_rng(seed)
        sample_n = min(sample_size, len(rgb_label_paths))
        sampled_idx = rng.choice(len(rgb_label_paths), size=sample_n, replace=False)

        detected_colors: set[tuple[int, int, int]] = set()
        for idx in sampled_idx:
            label_path = rgb_label_paths[int(idx)]
            raw = cv2.imread(str(label_path))
            if raw is None:
                logger.warning(
                    "Failed to read sampled RGB label for diagnostics: %s", label_path
                )
                continue
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            unique_colors = np.unique(rgb.reshape(-1, 3), axis=0)
            for color in unique_colors:
                detected_colors.add((int(color[0]), int(color[1]), int(color[2])))

        configured_colors = {
            (int(color[0]), int(color[1]), int(color[2]))
            for color in self.class_colors.astype(np.int64)
        }

        logger.debug(
            "Label color diagnostics (train): sampled %d RGB masks. Detected colors=%s",
            sample_n,
            sorted(detected_colors),
        )
        logger.debug(
            "Configured class colors=%s",
            {
                class_name: self.class_colors[idx].astype(int).tolist()
                for idx, class_name in enumerate(CLASS_NAMES)
            },
        )

        unexpected = sorted(detected_colors - configured_colors)
        missing = sorted(configured_colors - detected_colors)
        if unexpected or missing:
            logger.warning(
                "Label color convention mismatch detected. Unexpected colors=%s, Missing configured colors=%s. "
                "Known alternate docs values: ship=%s, land=%s. "
                "Set dataset.label_color_override to enforce the intended mapping.",
                unexpected,
                missing,
                CLASS_COLORS_RGB_DOC_ALTERNATE["ship"],
                CLASS_COLORS_RGB_DOC_ALTERNATE["land"],
            )

    def _compute_metadata(self) -> None:
        self.metadata = []
        metadata_iter = zip(self.label_1d_paths, self.label_paths)
        pbar = tqdm(
            metadata_iter,
            total=len(self.label_paths),
            desc=f"Metadata [{self.split}]",
            leave=False,
        )
        for lbl_1d, lbl_rgb in pbar:
            meta = {
                "has_ship": False,
                "has_oil_spill": False,
                "has_look_alike": False,
                "has_land": False,
                "pixel_counts": {name: 0 for name in CLASS_NAMES},
                "pixel_fractions": {name: 0.0 for name in CLASS_NAMES},
                "ship_component_count": 0,
                "ship_component_max_area": 0,
            }

            mask = None

            if lbl_1d is not None:
                raw = cv2.imread(str(lbl_1d), cv2.IMREAD_GRAYSCALE)
                if raw is not None:
                    mask = raw.astype(np.int64)
                else:
                    raise IOError(f"Failed to read 1D mask: {lbl_1d}")

            if mask is None and lbl_rgb is not None:
                raw_rgb = cv2.imread(str(lbl_rgb))
                if raw_rgb is not None:
                    raw_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)
                    mask = rgb_mask_to_class_index(raw_rgb, self.class_colors)
                else:
                    raise IOError(f"Failed to read RGB mask: {lbl_rgb}")

            if mask is not None:
                pixel_counts = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)
                total_pixels = max(int(pixel_counts.sum()), 1)
                ship_mask = (mask == SHIP_CLASS_IDX).astype(np.uint8)
                if ship_mask.any():
                    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                        ship_mask, connectivity=8
                    )
                    ship_component_count = max(num_labels - 1, 0)
                    ship_component_max_area = (
                        int(stats[1:, cv2.CC_STAT_AREA].max()) if num_labels > 1 else 0
                    )
                else:
                    ship_component_count = 0
                    ship_component_max_area = 0

                meta["has_ship"] = bool(np.any(mask == SHIP_CLASS_IDX))
                meta["has_oil_spill"] = bool(np.any(mask == OIL_SPILL_IDX))
                meta["has_look_alike"] = bool(np.any(mask == LOOK_ALIKE_IDX))
                meta["has_land"] = bool(np.any(mask == NUM_CLASSES - 1))
                meta["pixel_counts"] = {
                    name: int(pixel_counts[idx]) for idx, name in enumerate(CLASS_NAMES)
                }
                meta["pixel_fractions"] = {
                    name: float(pixel_counts[idx] / total_pixels)
                    for idx, name in enumerate(CLASS_NAMES)
                }
                meta["ship_component_count"] = ship_component_count
                meta["ship_component_max_area"] = ship_component_max_area

            self.metadata.append(meta)

        pbar.close()

    def __len__(self) -> int:
        return len(self.image_paths)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def set_crop_size(self, crop_size: int) -> None:
        self.crop_size = int(crop_size)

    def select_indices(self, indices: List[int]) -> None:
        selected = sorted(int(i) for i in indices)
        self.image_paths = [self.image_paths[i] for i in selected]
        self.label_paths = [self.label_paths[i] for i in selected]
        self.label_1d_paths = [self.label_1d_paths[i] for i in selected]
        self.metadata = [self.metadata[i] for i in selected]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        raw_image = self.load_raw_image(idx)
        expected_channels = resolve_model_num_channels(self.config)

        mask = self._load_mask(idx)

        if self.enable_train_augmentations and self.train_intensity_transform is not None:
            raw_image = np.asarray(
                self.train_intensity_transform(image=raw_image)["image"],
                dtype=np.float32,
            )

        working_image: np.ndarray = raw_image

        if self.enable_train_augmentations and self.copy_paste is not None:
            working_image, mask = self.copy_paste.apply(working_image, mask)

        crop_mode = "full-image"
        if self.enable_train_augmentations:
            working_image, mask, crop_mode = self.class_aware_cropper.crop(
                working_image,
                mask,
                self.crop_size,
            )

        if self.transform is not None:
            transformed = self.transform(
                image=working_image,
                mask=mask.astype(np.int32, copy=False),
            )
            working_image = np.asarray(transformed["image"], dtype=np.float32)
            mask = np.asarray(transformed["mask"], dtype=np.int64)

        image = self._encode_feature_image(working_image)
        image = self._normalize_feature_image(image)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.astype(np.int64))

        assert image.shape[0] == expected_channels, (
            f"Expected {expected_channels} channel image, got {image.shape}"
        )
        assert torch.isfinite(image).all(), "Non-finite SAR feature values detected"
        assert mask.dtype == torch.int64, f"Expected int64 mask, got {mask.dtype}"

        meta = self.metadata[idx]
        return {
            "image": image,
            "mask": mask,
            "crop_mode": crop_mode,
            "image_path": str(img_path),
            "has_ship": meta["has_ship"],
            "has_oil_spill": meta["has_oil_spill"],
            "has_look_alike": meta["has_look_alike"],
            "has_land": meta["has_land"],
        }

    def _feature_cache_path(self, image_path: Path) -> Path:
        if self.feature_cache_dir is None:
            raise RuntimeError("SAR feature cache is disabled for this dataset.")
        return self.feature_cache_dir / self.split / f"{image_path.stem}.npy"

    def _load_raw_amplitude_from_path(self, img_path: Path) -> np.ndarray:
        amplitude = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if amplitude is None:
            raise IOError(f"Failed to read image: {img_path}")
        return np.ascontiguousarray(amplitude.astype(np.float32) / 255.0)

    def _encode_feature_image(self, raw_image: np.ndarray) -> np.ndarray:
        amplitude = raw_image.astype(np.float32)
        if amplitude.ndim == 3:
            amplitude = amplitude.squeeze(-1)
        amplitude = np.clip(amplitude, 0.0, 1.0)

        expected_channels = resolve_model_num_channels(self.config)
        if self.sar_encoder is not None:
            image = self.sar_encoder.transform(amplitude)
        elif expected_channels == 1:
            image = amplitude[..., np.newaxis]
        else:
            image = np.repeat(amplitude[..., np.newaxis], expected_channels, axis=-1)

        if image.shape[-1] != expected_channels:
            raise RuntimeError(
                f"Expected {expected_channels} encoded channels, got {image.shape[-1]}. "
                "Check model.num_channels and sar_features channel ordering in the active config."
            )

        return np.ascontiguousarray(image.astype(np.float32))

    def _normalize_feature_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3:
            raise RuntimeError(f"Expected (H, W, C) feature image, got {image.shape}")

        channel_names = resolve_sar_channel_names(self.config)
        amplitude_indices = [
            idx for idx, name in enumerate(channel_names) if name == "amplitude"
        ]
        amp_p1 = 0.0
        amp_p99 = 1.0
        if self.sar_encoder is not None:
            p1 = getattr(self.sar_encoder, "amplitude_p1", None)
            p99 = getattr(self.sar_encoder, "amplitude_p99", None)
            if p1 is not None and p99 is not None and float(p99) > float(p1):
                amp_p1 = float(p1)
                amp_p99 = float(p99)

        # Clip each channel independently. Amplitude is further p1-p99 stretched;
        # texture maps are already ~[0,1] from SARFeatureEncoder.
        normalized = np.clip(image.astype(np.float32).copy(), 0.0, 1.0)
        for amplitude_idx in amplitude_indices:
            amp = np.clip(normalized[..., amplitude_idx], amp_p1, amp_p99)
            normalized[..., amplitude_idx] = (amp - amp_p1) / (amp_p99 - amp_p1 + 1e-8)

        mean, std = get_config_normalize_stats(self.config)
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
        std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
        if normalized.shape[-1] != mean_arr.shape[-1]:
            raise RuntimeError(
                f"Normalization stats length {mean_arr.shape[-1]} does not match image channels {normalized.shape[-1]}"
            )

        normalized = (normalized - mean_arr) / std_arr
        return np.ascontiguousarray(normalized.astype(np.float32))

    def _compute_feature_image_from_path(self, img_path: Path) -> np.ndarray:
        return self._encode_feature_image(self._load_raw_amplitude_from_path(img_path))

    def load_feature_image(self, idx: int) -> np.ndarray:
        return self._encode_feature_image(self.load_raw_image(idx))

    def load_raw_image(self, idx: int) -> np.ndarray:
        img_path = self.image_paths[idx]
        if not self.feature_cache_enabled:
            return self._load_raw_amplitude_from_path(img_path)

        cache_path = self._feature_cache_path(img_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing cached SAR amplitude for {img_path}. "
                f"Expected cache file: {cache_path}"
            )

        image = np.load(cache_path, allow_pickle=False)
        if image.ndim != 2:
            raise RuntimeError(
                f"Invalid cached SAR amplitude tensor at {cache_path}: expected (H, W), got {image.shape}"
            )
        return np.ascontiguousarray(image.astype(np.float32, copy=False))

    def precompute_sar_features(self, overwrite: bool = False) -> None:
        if not self.feature_cache_enabled:
            return

        assert self.feature_cache_dir is not None
        written = 0
        skipped = 0
        pbar = tqdm(
            self.image_paths,
            total=len(self.image_paths),
            desc=f"SAR raw cache [{self.split}]",
            leave=False,
        )
        for img_path in pbar:
            cache_path = self._feature_cache_path(img_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists() and not overwrite:
                skipped += 1
                pbar.set_postfix(written=written, reused=skipped)
                continue

            features = self._load_raw_amplitude_from_path(img_path)
            tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp.npy")
            np.save(tmp_path, features)
            tmp_path.replace(cache_path)
            written += 1
            pbar.set_postfix(written=written, reused=skipped)

        pbar.close()

        logger.info(
            "SAR raw cache ready [%s]: wrote=%d, reused=%d, dir=%s",
            self.split,
            written,
            skipped,
            self.feature_cache_dir / self.split,
        )

    def _load_mask(self, idx: int) -> np.ndarray:
        lbl_1d = self.label_1d_paths[idx]
        lbl_rgb = self.label_paths[idx]

        if lbl_1d is not None:
            mask = cv2.imread(str(lbl_1d), cv2.IMREAD_UNCHANGED)
            if mask is not None:
                if mask.ndim == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                return remap_ignore_labels(
                    mask,
                    ignore_index=self.ignore_index,
                    mask_ignore_values=self.mask_ignore_values,
                )

        if lbl_rgb is not None:
            rgb = cv2.imread(str(lbl_rgb))
            if rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                return rgb_mask_to_class_index(rgb, self.class_colors)

        raise FileNotFoundError(
            f"No segmentation mask found for sample {self.image_paths[idx]}. "
            "Provide labels_1D or RGB labels, or enable dataset.allow_missing_masks "
            "for unsupervised visualization-only workflows."
        )

    def get_class_pixel_counts(self) -> Dict[str, int]:
        counts = {name: 0 for name in CLASS_NAMES}
        for idx in range(len(self)):
            mask = self._load_mask(idx)
            for c, name in enumerate(CLASS_NAMES):
                counts[name] += int(np.sum(mask == c))
        return counts
