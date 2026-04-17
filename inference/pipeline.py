from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.multiscale import MultiScaleInference
from inference.tta import TestTimeAugmentation

logger = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(self, config: dict, profile: str = "full") -> None:
        self.config = config
        self.profile = profile.lower()

        inference_cfg = config.get("inference", {})
        tta_cfg = inference_cfg.get("tta", {})
        ms_cfg = inference_cfg.get("multiscale", {})

        tta_enabled = bool(tta_cfg.get("enabled", False))
        ms_enabled = bool(ms_cfg.get("enabled", False))
        configured_tta_enabled = tta_enabled
        configured_ms_enabled = ms_enabled

        if self.profile == "fast":
            tta_enabled = False
            ms_enabled = False
            if configured_tta_enabled or configured_ms_enabled:
                logger.warning(
                    "Inference profile 'fast' disables TTA/multiscale "
                    "(configured tta=%s, multiscale=%s -> effective tta=%s, multiscale=%s).",
                    configured_tta_enabled,
                    configured_ms_enabled,
                    tta_enabled,
                    ms_enabled,
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
        if self.multiscale is not None:
            return "multiscale" + (" + tta" if self.tta is not None else "")
        if self.tta is not None:
            return "single_scale + tta"
        return "single_scale resize-only"

    def estimate_forward_passes_per_image(self) -> int:
        scale_count = len(self.multiscale.scales) if self.multiscale is not None else 1
        tta_count = len(self.tta.augmentation_types) if self.tta is not None else 1
        return int(scale_count * tta_count)

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
            "forward_passes_per_image": self.estimate_forward_passes_per_image(),
        }

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
            probs = self.tta.predict(model, image_batch)
        else:
            output = model(image_batch)
            logits = output["seg_logits"]
            probs = F.softmax(logits, dim=1)

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
        return probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
