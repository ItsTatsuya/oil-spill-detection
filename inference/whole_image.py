"""Single-pass full-image inference with pad-to-multiple geometry."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WholeImageInference:
    """One forward pass at native resolution (pad H/W to a network-friendly multiple).

    Much faster than sliding-window for train-time validation. Preserves aspect
    ratio (unlike square resize). Use sliding-window for final high-quality eval.
    """

    def __init__(self, align: int = 32) -> None:
        self.align = max(int(align), 1)

    def _pad_to_align(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = image.shape
        pad_h = (self.align - h % self.align) % self.align
        pad_w = (self.align - w % self.align) % self.align
        if pad_h == 0 and pad_w == 0:
            return image, (0, 0)
        padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        return padded, (pad_h, pad_w)

    @torch.inference_mode()
    def predict(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
        *,
        amp_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if image_batch.dim() != 4:
            raise ValueError(
                f"Expected (B, C, H, W), got shape {tuple(image_batch.shape)}"
            )
        orig_h, orig_w = int(image_batch.shape[-2]), int(image_batch.shape[-1])
        image, (pad_h, pad_w) = self._pad_to_align(image_batch)

        device_type = image.device.type
        use_amp = device_type == "cuda" and amp_dtype is not None
        with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype or torch.float16):
            outputs = model(image)
            logits = outputs["seg_logits"]
            if logits.shape[-2:] != image.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        probs = F.softmax(logits.float(), dim=1)
        if pad_h > 0 or pad_w > 0:
            probs = probs[:, :, :orig_h, :orig_w]
        return probs
