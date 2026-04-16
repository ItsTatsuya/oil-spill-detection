import logging
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from constants import CLASS_NAMES, DEFAULT_PIXEL_COUNTS, NUM_CLASSES

logger = logging.getLogger(__name__)


def compute_median_frequency_weights(
    pixel_counts: List[int],
    num_classes: int = NUM_CLASSES,
    ship_class_idx: int = 3,
    ship_weight_cap: float = 15.0,
) -> torch.Tensor:
    assert len(pixel_counts) == num_classes, (
        f"Expected {num_classes} pixel counts, got {len(pixel_counts)}"
    )

    total = sum(pixel_counts)
    freqs = [count / total for count in pixel_counts]
    median_freq = float(np.median(freqs))

    weights_list = []
    for c, freq in enumerate(freqs):
        w = median_freq / max(freq, 1e-9)
        if c == ship_class_idx:
            w = min(w, ship_weight_cap)
        weights_list.append(w)

    weights = torch.tensor(weights_list, dtype=torch.float32)

    if weights[0] >= 0.2:
        logger.warning(
            f"sea_surface weight unusually high: {weights[0]:.4f} (expected < 0.2 for typical SAR dataset)"
        )
    if not (4.0 < weights[1] < 8.0):
        logger.warning(
            f"oil_spill weight out of expected range: {weights[1]:.4f} (expected 4.0-8.0)"
        )
    if not (0.8 < weights[2] < 1.3):
        logger.warning(
            f"look_alike weight out of expected range: {weights[2]:.4f} (expected 0.8-1.3)"
        )
    raw_ship_weight = median_freq / max(freqs[ship_class_idx], 1e-9)
    if raw_ship_weight > ship_weight_cap:
        logger.info(
            f"ship weight capped at {ship_weight_cap} (raw={raw_ship_weight:.4f})"
        )
    else:
        logger.warning(
            f"ship weight {weights[ship_class_idx]:.4f} did not reach cap "
            f"({ship_weight_cap}) — ship may be under-represented in this split"
        )
    if not (0.9 < weights[4] < 1.4):
        logger.warning(
            f"land weight out of expected range: {weights[4]:.4f} (expected 0.9-1.4)"
        )

    logger.info("Focal loss class weights (median-frequency, ship capped):")
    for name, w in zip(CLASS_NAMES, weights):
        logger.info(f"  {name}: {w:.4f}")

    return weights


compute_inverse_frequency_weights = compute_median_frequency_weights


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Union[str, List[float], None] = "inverse_frequency",
        pixel_counts: Optional[List[int]] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha in ("inverse_frequency", "median_frequency"):
            counts = pixel_counts if pixel_counts is not None else DEFAULT_PIXEL_COUNTS
            alpha_tensor = compute_median_frequency_weights(counts)
        elif isinstance(alpha, (list, tuple)):
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
            assert len(alpha_tensor) == NUM_CLASSES, (
                f"alpha must have {NUM_CLASSES} elements, got {len(alpha_tensor)}"
            )
        elif alpha is None:
            alpha_tensor = torch.ones(NUM_CLASSES, dtype=torch.float32)
            logger.warning(
                "FocalLoss: alpha=None means uniform class weights. "
                "This will severely hurt rare class detection. "
                "Use alpha='inverse_frequency' for this dataset."
            )
        else:
            raise ValueError(
                f"alpha must be 'inverse_frequency', 'median_frequency', list, or None. Got: {alpha}"
            )

        self.register_buffer("base_alpha", alpha_tensor.clone())
        self.register_buffer("alpha", alpha_tensor.clone())

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        assert probs.dim() == 4, f"Expected probs shape (B, C, H, W), got {probs.shape}"
        assert targets.dim() == 3, (
            f"Expected targets shape (B, H, W), got {targets.shape}"
        )
        assert probs.shape[0] == targets.shape[0], (
            f"Batch size mismatch: probs {probs.shape[0]} vs targets {targets.shape[0]}"
        )

        B, C, H, W = probs.shape
        assert C == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {C}"

        probs = probs.clamp(min=1e-7, max=1.0 - 1e-7)

        valid_mask = targets != self.ignore_index

        safe_targets = targets.masked_fill(~valid_mask, 0)
        if valid_mask.any():
            assert (
                safe_targets[valid_mask].min() >= 0
                and safe_targets[valid_mask].max() < C
            ), (
                f"Targets must be in [0, {C - 1}] or equal ignore_index={self.ignore_index}"
            )

        log_probs = torch.log(probs)

        targets_expanded = safe_targets.unsqueeze(1)

        log_pt = log_probs.gather(dim=1, index=targets_expanded).squeeze(1)
        pt = probs.gather(dim=1, index=targets_expanded).squeeze(1)

        alpha_t = self.alpha.to(probs.device)[safe_targets]

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -alpha_t * focal_weight * log_pt

        loss = loss * valid_mask.float()

        if self.reduction == "mean":
            valid_count = valid_mask.float().sum().clamp(min=1.0)
            return loss.sum() / valid_count
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def update_gamma(self, new_gamma: float) -> None:
        old_gamma = self.gamma
        self.gamma = new_gamma
        logger.info(f"FocalLoss gamma updated: {old_gamma:.1f} -> {new_gamma:.1f}")

    def set_alpha(self, new_alpha: torch.Tensor) -> None:
        if new_alpha.shape != self.alpha.shape:
            raise ValueError(
                f"alpha shape mismatch: expected {tuple(self.alpha.shape)}, got {tuple(new_alpha.shape)}"
            )
        self.alpha.copy_(new_alpha.to(device=self.alpha.device, dtype=self.alpha.dtype))
