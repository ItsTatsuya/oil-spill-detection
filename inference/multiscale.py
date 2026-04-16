import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiScaleInference:

    def __init__(
        self,
        scales: Optional[List[float]] = None,
        aggregation: str = "mean_softmax",
        tta: Optional[object] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.scales = scales or [0.75, 1.0, 1.25, 1.5]
        self.aggregation = aggregation
        self.tta = tta
        self.device = device

        logger.info(
            f"MultiScaleInference: scales={self.scales}, "
            f"TTA={'enabled' if tta else 'disabled'}"
        )

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        image_tensor: torch.Tensor,
    ) -> torch.Tensor:
        assert image_tensor.dim() == 4, (
            f"Expected (B, C, H, W) input, got {image_tensor.shape}"
        )
        assert image_tensor.shape[0] == 1, (
            "Multi-scale inference is designed for batch size 1"
        )

        model.eval()
        device = self.device or next(model.parameters()).device
        image_tensor = image_tensor.to(device)

        _, C, H, W = image_tensor.shape
        original_size = (H, W)
        accumulated_probs = None
        valid_scale_count = 0

        for scale in self.scales:
            new_h = max(32, round(H * scale / 32) * 32)
            new_w = max(32, round(W * scale / 32) * 32)

            if (new_h, new_w) != (H, W):
                scaled_image = F.interpolate(
                    image_tensor,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                scaled_image = image_tensor

            if self.tta is not None:
                probs_scaled = self.tta.predict(model, scaled_image)
            else:
                output = model(scaled_image)
                logits = output["seg_logits"]
                probs_scaled = (
                    logits
                    if self._looks_like_probabilities(logits)
                    else F.softmax(logits, dim=1)
                )
                if probs_scaled.shape[-2:] != (new_h, new_w):
                    probs_scaled = F.interpolate(
                        probs_scaled,
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    )

            if probs_scaled.shape[-2:] != original_size:
                probs_at_original = F.interpolate(
                    probs_scaled,
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                probs_at_original = probs_scaled

            if torch.isnan(probs_at_original).any():
                logger.warning(
                    f"NaN detected at scale {scale:.2f}x — skipping this scale"
                )
                continue

            valid_scale_count += 1
            if accumulated_probs is None:
                accumulated_probs = probs_at_original
            else:
                accumulated_probs = accumulated_probs + probs_at_original

        if accumulated_probs is None:
            raise RuntimeError(
                "All inference scales produced NaN outputs. "
                "Check model weights and input data."
            )
        averaged_probs = accumulated_probs / valid_scale_count
        return averaged_probs

    def _looks_like_probabilities(self, tensor: torch.Tensor) -> bool:
        sample = tensor.detach().float()
        return bool(
            sample.min() >= -1e-3
            and sample.max() <= 1.0 + 1e-3
            and (sample.sum(dim=1) - 1.0).abs().max() < 1e-3
        )
