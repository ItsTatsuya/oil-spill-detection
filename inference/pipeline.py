from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.multiscale import MultiScaleInference
from inference.sliding_window import SlidingWindowInference
from inference.tta import TestTimeAugmentation
from inference.whole_image import WholeImageInference

logger = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(self, config: dict, profile: str = "full") -> None:
        self.config = config
        self.profile = profile.lower()

        inference_cfg = config.get("inference", {})
        tta_cfg = inference_cfg.get("tta", {})
        ms_cfg = inference_cfg.get("multiscale", {})
        sw_cfg = inference_cfg.get("sliding_window", {})
        whole_cfg = inference_cfg.get("whole_image", {})

        tta_enabled = bool(tta_cfg.get("enabled", False))
        ms_enabled = bool(ms_cfg.get("enabled", False))
        sw_enabled = bool(sw_cfg.get("enabled", True))
        # Prefer single-pass whole-image for train-time "fast" validation.
        whole_default = self.profile == "fast"
        whole_enabled = bool(whole_cfg.get("enabled", whole_default))

        precision = str(config.get("training", {}).get("precision", "bf16")).lower()
        if precision == "bf16" and torch.cuda.is_available():
            self.amp_dtype: Optional[torch.dtype] = torch.bfloat16
        elif precision == "fp16" and torch.cuda.is_available():
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None

        if self.profile == "fast":
            tta_enabled = False
            ms_enabled = False
            # Fast path: whole-image single forward (huge speedup vs sliding window).
            if whole_enabled:
                sw_enabled = False
            logger.info(
                "Inference profile 'fast': whole_image=%s sliding_window=%s "
                "tta=False multiscale=False.",
                whole_enabled,
                sw_enabled,
            )
        elif self.profile == "full":
            # Final scoring: prefer sliding window when configured; whole-image off
            # unless explicitly enabled without sliding window.
            if sw_enabled:
                whole_enabled = False

        train_crop = int(
            config.get("augmentation", {}).get("train", {}).get("crop_size", 640)
        )

        self.whole_image: Optional[WholeImageInference] = None
        if whole_enabled:
            align = int(whole_cfg.get("align", 32))
            self.whole_image = WholeImageInference(align=align)
            logger.info("Whole-image inference enabled (align=%d).", align)

        self.sliding_window: Optional[SlidingWindowInference] = None
        if sw_enabled:
            crop_size = int(sw_cfg.get("crop_size", train_crop))
            # Full profile can use denser stride; default half crop.
            default_stride = crop_size // 2
            if self.profile == "fast":
                # Non-overlapping tiles if whole-image is off.
                default_stride = crop_size
            stride = sw_cfg.get("stride", default_stride)
            stride = int(stride) if stride is not None else default_stride
            # Per-profile batch override for 8GB cards.
            if self.profile == "fast":
                batch_size = int(sw_cfg.get("fast_batch_size", sw_cfg.get("batch_size", 2)))
            else:
                batch_size = int(sw_cfg.get("batch_size", 2))
            self.sliding_window = SlidingWindowInference(
                crop_size=crop_size,
                stride=stride,
                batch_size=batch_size,
                amp_dtype=self.amp_dtype,
            )
            logger.info(
                "Sliding-window inference enabled: crop=%d stride=%d batch=%d amp=%s.",
                crop_size,
                stride,
                batch_size,
                self.amp_dtype,
            )

        self.tta: Optional[TestTimeAugmentation] = None
        if tta_enabled:
            self.tta = TestTimeAugmentation(
                augmentation_types=tta_cfg.get("augmentation_types"),
                aggregation=str(tta_cfg.get("aggregation", "mean_softmax")),
            )

        self.multiscale: Optional[MultiScaleInference] = None
        if ms_enabled:
            self.multiscale = MultiScaleInference(
                scales=ms_cfg.get("scales"),
                aggregation=str(ms_cfg.get("aggregation", "mean_softmax")),
                tta=self.tta,
            )

    def describe(self) -> str:
        parts: list[str] = []
        if self.whole_image is not None:
            parts.append(f"whole_image(align={self.whole_image.align})")
        elif self.sliding_window is not None:
            parts.append(
                f"sliding_window({self.sliding_window.crop_size}/"
                f"{self.sliding_window.stride})"
            )
        else:
            parts.append("single_scale")
        if self.multiscale is not None:
            parts.append("multiscale")
        if self.tta is not None:
            parts.append("tta")
        return " + ".join(parts)

    def estimate_forward_passes_per_image(self) -> int:
        scale_count = len(self.multiscale.scales) if self.multiscale is not None else 1
        tta_count = len(self.tta.augmentation_types) if self.tta is not None else 1
        if self.whole_image is not None:
            tile_count = 1
        elif self.sliding_window is not None:
            # 1250x650 / 640 with stride≈crop → ~4 tiles; with half stride ~6-8.
            tile_count = 4 if self.sliding_window.stride >= self.sliding_window.crop_size else 8
        else:
            tile_count = 1
        return int(scale_count * tta_count * tile_count)

    def describe_profile(self) -> dict:
        scales = list(self.multiscale.scales) if self.multiscale is not None else [1.0]
        return {
            "evaluation_profile": self.profile,
            "pipeline_description": self.describe(),
            "tta_enabled": self.tta is not None,
            "tta_augmentations": (
                len(self.tta.augmentation_types) if self.tta is not None else 0
            ),
            "multiscale_enabled": self.multiscale is not None,
            "multiscale_scales": scales,
            "num_scales": len(scales),
            "whole_image_enabled": self.whole_image is not None,
            "sliding_window_enabled": self.sliding_window is not None,
            "sliding_window_crop": (
                self.sliding_window.crop_size if self.sliding_window is not None else None
            ),
            "sliding_window_stride": (
                self.sliding_window.stride if self.sliding_window is not None else None
            ),
            "forward_passes_per_image": self.estimate_forward_passes_per_image(),
        }

    def _forward_probabilities(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.whole_image is not None:
            return self.whole_image.predict(
                model, image_batch, amp_dtype=self.amp_dtype
            )
        if self.sliding_window is not None:
            return self.sliding_window.predict(model, image_batch)

        device_type = image_batch.device.type
        use_amp = device_type == "cuda" and self.amp_dtype is not None
        with torch.amp.autocast(
            device_type, enabled=use_amp, dtype=self.amp_dtype or torch.float16
        ):
            output = model(image_batch)
            logits = output["seg_logits"]
            if logits.shape[-2:] != image_batch.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=image_batch.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        return F.softmax(logits.float(), dim=1)

    def _predict_with_tta(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
    ) -> torch.Tensor:
        assert self.tta is not None
        accumulated = None
        for aug_type in self.tta.augmentation_types:
            aug_image = self.tta.apply_augmentation(image_batch, aug_type)
            probs = self._forward_probabilities(model, aug_image)
            probs = self.tta.reverse_augmentation(probs, aug_type)
            if accumulated is None:
                accumulated = probs
            else:
                accumulated = accumulated + probs
        assert accumulated is not None
        return accumulated / float(len(self.tta.augmentation_types))

    @torch.inference_mode()
    def predict_probabilities(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if self.multiscale is not None:
            probs = self.multiscale.predict(model, image_batch)
        elif self.tta is not None:
            probs = self._predict_with_tta(model, image_batch)
        else:
            probs = self._forward_probabilities(model, image_batch)

        if target_size is not None and probs.shape[-2:] != target_size:
            probs = F.interpolate(
                probs,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return probs

    @torch.inference_mode()
    def predict_segmentation_map(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        probs = self.predict_probabilities(model, image_batch, target_size=target_size)
        pred = probs.argmax(dim=1).cpu().numpy().astype(np.int64)
        # Val/eval always use batch size 1; keep (H, W) for that path. Never use
        # squeeze(0) on a multi-image batch — that silently mis-shapes metrics.
        if pred.shape[0] == 1:
            return pred[0]
        return pred
