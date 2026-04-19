from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

from constants import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from utils.config import resolve_model_num_channels, resolve_sar_channel_names

logger = logging.getLogger(__name__)

RARE_CLASS_TO_INDEX = {
    "oil_spill": CLASS_NAMES.index("oil_spill"),
    "look_alike": CLASS_NAMES.index("look_alike"),
    "ship": CLASS_NAMES.index("ship"),
}


def get_input_normalize_stats(
    num_channels: int,
    amplitude_channel_index: Optional[int] = 0,
) -> tuple[list[float], list[float]]:
    if num_channels <= 0:
        raise ValueError(f"num_channels must be positive, got {num_channels}")
    mean = [0.0] * num_channels
    std = [1.0] * num_channels
    if amplitude_channel_index is not None and 0 <= amplitude_channel_index < num_channels:
        mean[amplitude_channel_index] = float(IMAGENET_MEAN[0])
        std[amplitude_channel_index] = float(IMAGENET_STD[0])
    return mean, std


def get_config_normalize_stats(config: dict) -> tuple[list[float], list[float]]:
    channel_names = resolve_sar_channel_names(config)
    amplitude_idx = channel_names.index("amplitude") if "amplitude" in channel_names else None
    return get_input_normalize_stats(
        resolve_model_num_channels(config),
        amplitude_channel_index=amplitude_idx,
    )


class SpeckleNoise(ImageOnlyTransform):
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.05, 0.15),
        p: float = 0.7,
    ) -> None:
        super().__init__(p=p)
        self.scale_range = scale_range

    def apply(self, image: np.ndarray, scale: float = 0.1, **kwargs) -> np.ndarray:
        original_dtype = image.dtype
        img_float = image.astype(np.float32)
        noise = np.random.rayleigh(scale=scale, size=img_float.shape[:2]).astype(np.float32)
        noise = noise / (scale * np.sqrt(np.pi / 2.0))
        if img_float.ndim == 3:
            noise = noise[:, :, np.newaxis]
        noisy = img_float * noise
        if original_dtype == np.uint8:
            return noisy.clip(0, 255).astype(np.uint8)
        return noisy.clip(0.0, 1.0).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return {"scale": scale}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("scale_range", "p")


class ClassAwareCropper:
    def __init__(self, config: dict) -> None:
        self.config = config
        train_cfg = config.get("augmentation", {}).get("train", {})
        self.crop_strategy = str(train_cfg.get("crop_strategy", "random")).lower()
        self.ignore_index = int(config.get("loss", {}).get("ce_ignore_index", -100))
        probs = train_cfg.get("class_aware_crop_probs", {})
        self.class_probs = {
            "ship": float(probs.get("ship", 0.25)),
            "oil_spill": float(probs.get("oil_spill", 0.25)),
            "look_alike": float(probs.get("look_alike", 0.25)),
            "random": float(probs.get("random", 0.25)),
        }

    @property
    def enabled(self) -> bool:
        return self.crop_strategy == "class_aware"

    def crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        crop_size: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        crop_size = int(crop_size)
        if crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got {crop_size}")

        image_padded, mask_padded = self._pad_if_needed(image, mask, crop_size)
        if not self.enabled:
            return self._random_crop(image_padded, mask_padded, crop_size, "random-context")

        crop_mode = self._sample_crop_mode()
        class_name = crop_mode.replace("-centric", "")
        class_idx = RARE_CLASS_TO_INDEX.get(class_name)
        if class_idx is not None:
            class_crop = self._class_crop(image_padded, mask_padded, crop_size, class_idx, crop_mode)
            if class_crop is not None:
                return class_crop

            fallback_name = self._sample_fallback_class(mask_padded, exclude=class_name)
            if fallback_name is not None:
                fallback_mode = f"{fallback_name}-fallback"
                fallback_crop = self._class_crop(
                    image_padded,
                    mask_padded,
                    crop_size,
                    RARE_CLASS_TO_INDEX[fallback_name],
                    fallback_mode,
                )
                if fallback_crop is not None:
                    return fallback_crop

        return self._random_crop(image_padded, mask_padded, crop_size, "random-context")

    def _sample_crop_mode(self) -> str:
        modes = ["ship-centric", "oil_spill-centric", "look_alike-centric", "random-context"]
        weights = np.array(
            [
                self.class_probs["ship"],
                self.class_probs["oil_spill"],
                self.class_probs["look_alike"],
                self.class_probs["random"],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(weights).all() or weights.sum() <= 0.0:
            return "random-context"
        weights = weights / weights.sum()
        return str(np.random.choice(modes, p=weights))

    def _sample_fallback_class(
        self,
        mask: np.ndarray,
        *,
        exclude: str,
    ) -> Optional[str]:
        present = [
            class_name
            for class_name, class_idx in RARE_CLASS_TO_INDEX.items()
            if class_name != exclude and np.any(mask == class_idx)
        ]
        if not present:
            return None
        return str(np.random.choice(present))

    def _class_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        crop_size: int,
        class_idx: int,
        crop_mode: str,
    ) -> Optional[tuple[np.ndarray, np.ndarray, str]]:
        positions = np.argwhere(mask == class_idx)
        if positions.size == 0:
            return None
        center_y, center_x = positions[np.random.randint(len(positions))]
        return self._crop_around_center(image, mask, crop_size, int(center_y), int(center_x), crop_mode)

    def _random_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        crop_size: int,
        crop_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        h, w = mask.shape
        max_y = max(h - crop_size, 0)
        max_x = max(w - crop_size, 0)
        top = 0 if max_y == 0 else int(np.random.randint(0, max_y + 1))
        left = 0 if max_x == 0 else int(np.random.randint(0, max_x + 1))
        return self._slice_crop(image, mask, top, left, crop_size, crop_mode)

    def _crop_around_center(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        crop_size: int,
        center_y: int,
        center_x: int,
        crop_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        h, w = mask.shape
        half = crop_size // 2
        top = int(np.clip(center_y - half, 0, max(h - crop_size, 0)))
        left = int(np.clip(center_x - half, 0, max(w - crop_size, 0)))
        return self._slice_crop(image, mask, top, left, crop_size, crop_mode)

    def _slice_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        top: int,
        left: int,
        crop_size: int,
        crop_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        bottom = top + crop_size
        right = left + crop_size
        if image.ndim == 2:
            image_crop = image[top:bottom, left:right]
        else:
            image_crop = image[top:bottom, left:right, ...]
        mask_crop = mask[top:bottom, left:right]
        return (
            np.ascontiguousarray(image_crop),
            np.ascontiguousarray(mask_crop),
            crop_mode,
        )

    def _pad_if_needed(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        crop_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape
        pad_h = max(crop_size - h, 0)
        pad_w = max(crop_size - w, 0)
        if pad_h == 0 and pad_w == 0:
            return image, mask

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if image.ndim == 2:
            image_padded = np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="reflect",
            )
        else:
            image_padded = np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="reflect",
            )
        mask_padded = np.pad(
            mask,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=self.ignore_index,
        )
        return np.ascontiguousarray(image_padded), np.ascontiguousarray(mask_padded)


class SARSegmentationAugmentation:
    def __init__(self, config: dict) -> None:
        self.config = config
        aug_cfg = config.get("augmentation", {})
        self.train_cfg = aug_cfg.get("train", {})
        self.test_cfg = aug_cfg.get("test", {})

    def get_train_transform(self, crop_size_override: Optional[int] = None) -> A.Compose:
        crop_size = int(crop_size_override or self.train_cfg.get("crop_size", 512))
        rotation_deg = float(self.train_cfg.get("rotation_degrees", 15))

        transforms_list = [
            A.HorizontalFlip(p=float(self.train_cfg.get("horizontal_flip_prob", 0.5))),
            A.VerticalFlip(p=float(self.train_cfg.get("vertical_flip_prob", 0.3))),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.9, 1.1),
                rotate=(-rotation_deg, rotation_deg),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT,
                p=0.3,
            ),
        ]
        return A.Compose(transforms_list)

    def get_train_intensity_transform(self) -> Optional[A.Compose]:
        speckle_cfg = self.train_cfg.get("speckle_noise", {})
        if not bool(speckle_cfg.get("enabled", True)):
            return None
        return A.Compose(
            [
                SpeckleNoise(
                    scale_range=tuple(speckle_cfg.get("scale_range", [0.05, 0.15])),
                    p=0.7,
                )
            ]
        )

    def get_test_transform(self) -> A.Compose:
        resize_size = int(self.test_cfg.get("resize", 640))
        return A.Compose(
            [
                A.Resize(
                    height=resize_size,
                    width=resize_size,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                )
            ]
        )

    def update_crop_size(self, new_size: int) -> A.Compose:
        logger.info("Updating training crop size to %dx%d", new_size, new_size)
        return self.get_train_transform(crop_size_override=new_size)


def get_progressive_crop_size(epoch: int, schedule: Optional[Dict[int, int]] = None) -> int:
    if schedule is None:
        schedule = {1: 512, 41: 576}
    for threshold in sorted(schedule.keys(), reverse=True):
        if epoch >= threshold:
            return int(schedule[threshold])
    return int(schedule[min(schedule.keys())])
