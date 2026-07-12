"""Sliding-window inference for full-resolution SAR images."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SlidingWindowInference:
    """Overlap-tile inference that preserves native image geometry.

    Training uses fixed square crops (e.g. 640x640) at native resolution. Final
    eval should tile at the training crop size and average overlapping probs.
    """

    def __init__(
        self,
        crop_size: int = 640,
        stride: Optional[int] = None,
        batch_size: int = 4,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.crop_size = int(crop_size)
        if self.crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got {crop_size}")
        self.stride = int(stride) if stride is not None else max(self.crop_size // 2, 1)
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.batch_size = max(int(batch_size), 1)
        self.amp_dtype = amp_dtype

    def _window_starts(self, length: int) -> list[int]:
        if length <= self.crop_size:
            return [0]
        starts = list(range(0, length - self.crop_size + 1, self.stride))
        last = length - self.crop_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _pad_if_needed(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = image.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        if pad_h == 0 and pad_w == 0:
            return image, (0, 0)
        padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        return padded, (pad_h, pad_w)

    def _forward_tiles(
        self,
        model: nn.Module,
        tiles: torch.Tensor,
    ) -> torch.Tensor:
        device_type = tiles.device.type
        use_amp = device_type == "cuda" and self.amp_dtype is not None
        with torch.amp.autocast(
            device_type, enabled=use_amp, dtype=self.amp_dtype or torch.float16
        ):
            outputs = model(tiles)
            logits = outputs["seg_logits"]
            if logits.shape[-2:] != (self.crop_size, self.crop_size):
                logits = F.interpolate(
                    logits,
                    size=(self.crop_size, self.crop_size),
                    mode="bilinear",
                    align_corners=False,
                )
        return F.softmax(logits.float(), dim=1)

    @torch.inference_mode()
    def predict(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
    ) -> torch.Tensor:
        if image_batch.dim() != 4:
            raise ValueError(
                f"Expected image batch (B, C, H, W), got shape {tuple(image_batch.shape)}"
            )
        if image_batch.shape[0] != 1:
            raise ValueError(
                "SlidingWindowInference currently requires batch size 1, "
                f"got {image_batch.shape[0]}"
            )

        orig_h, orig_w = int(image_batch.shape[-2]), int(image_batch.shape[-1])
        image, (pad_h, pad_w) = self._pad_if_needed(image_batch)
        _, _, h, w = image.shape

        y_starts = self._window_starts(h)
        x_starts = self._window_starts(w)
        windows: list[tuple[int, int]] = [(y, x) for y in y_starts for x in x_starts]

        # Infer class count from first tile batch (no wasted solo probe pass).
        first_chunk = windows[: self.batch_size]
        first_tiles = torch.cat(
            [
                image[:, :, y : y + self.crop_size, x : x + self.crop_size]
                for y, x in first_chunk
            ],
            dim=0,
        )
        first_probs = self._forward_tiles(model, first_tiles)
        num_classes = int(first_probs.shape[1])
        device = first_probs.device

        prob_sum = torch.zeros((1, num_classes, h, w), device=device, dtype=torch.float32)
        weight_sum = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)

        for i, (y, x) in enumerate(first_chunk):
            prob_sum[:, :, y : y + self.crop_size, x : x + self.crop_size] += first_probs[
                i : i + 1
            ]
            weight_sum[:, :, y : y + self.crop_size, x : x + self.crop_size] += 1.0

        for start in range(self.batch_size, len(windows), self.batch_size):
            chunk = windows[start : start + self.batch_size]
            tiles = torch.cat(
                [
                    image[:, :, y : y + self.crop_size, x : x + self.crop_size]
                    for y, x in chunk
                ],
                dim=0,
            )
            probs = self._forward_tiles(model, tiles)
            for i, (y, x) in enumerate(chunk):
                prob_sum[:, :, y : y + self.crop_size, x : x + self.crop_size] += probs[
                    i : i + 1
                ]
                weight_sum[:, :, y : y + self.crop_size, x : x + self.crop_size] += 1.0

        probs_full = prob_sum / weight_sum.clamp(min=1e-6)
        if pad_h > 0 or pad_w > 0:
            probs_full = probs_full[:, :, :orig_h, :orig_w]
        return probs_full
