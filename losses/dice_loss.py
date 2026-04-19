import logging
from typing import List, Optional

import torch
import torch.nn as nn

from constants import NUM_CLASSES

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        smooth: float = 1.0,
        ignore_index: int = -100,
        class_weights: Optional[torch.Tensor] = None,
        aggregation: str = "per_image",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.aggregation = str(aggregation).lower()
        if self.aggregation not in {"batch", "per_image"}:
            raise ValueError(
                f"aggregation must be one of ['batch', 'per_image'], got {aggregation}"
            )
        if class_weights is not None:
            assert class_weights.shape == (num_classes,), (
                f"class_weights must have shape ({num_classes},), "
                f"got {class_weights.shape}"
            )
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer(
                "class_weights",
                torch.ones(num_classes, dtype=torch.float32),
            )

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        assert probs.dim() == 4, f"Expected probs shape (B, C, H, W), got {probs.shape}"
        assert targets.dim() == 3, (
            f"Expected targets shape (B, H, W), got {targets.shape}"
        )
        assert probs.shape[1] == self.num_classes, (
            f"Expected {self.num_classes} classes in probs, got {probs.shape[1]}"
        )

        valid_mask = (targets != self.ignore_index).float()
        class_weights = self.class_weights

        weighted_dice_loss = probs.new_tensor(0.0)

        for c in range(self.num_classes):
            pred_c = probs[:, c, :, :]
            target_c = (targets == c).float()

            pred_c = pred_c * valid_mask
            target_c = target_c * valid_mask

            if self.aggregation == "per_image":
                intersection = (pred_c * target_c).sum(dim=(1, 2))
                union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
                dice_c = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_loss_c = (1.0 - dice_c).mean() * class_weights[c]
            else:
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                dice_c = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_loss_c = (1.0 - dice_c) * class_weights[c]

            weighted_dice_loss = weighted_dice_loss + dice_loss_c

        weight_sum = torch.sum(class_weights).clamp(min=1.0)
        return weighted_dice_loss / weight_sum

    def compute_per_class_dice(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> List[float]:
        scores = []
        valid_mask = (targets != self.ignore_index).float()

        for c in range(self.num_classes):
            pred_c = probs[:, c, :, :] * valid_mask
            target_c = (targets == c).float() * valid_mask
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            dice_c = float(
                ((2.0 * intersection + self.smooth) / (union + self.smooth)).mean()
            )
            scores.append(dice_c)

        return scores
