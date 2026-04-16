import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConfusionPenaltyLoss(nn.Module):

    def __init__(
        self,
        penalize_pairs: Optional[List[List[int]]] = None,
        pair_weights: Optional[List[float]] = None,
        num_classes: int = 5,
        margin_start: float = 0.1,
        margin_end: float = 0.4,
        margin_ramp_start_epoch: Optional[int] = None,
        margin_ramp_end_epoch: Optional[int] = None,
        phase3_start_epoch: int = 151,
        phase3_end_epoch: int = 300,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.margin_start = margin_start
        self.margin_end = margin_end
        self.margin_ramp_start_epoch = int(
            phase3_start_epoch
            if margin_ramp_start_epoch is None
            else margin_ramp_start_epoch
        )
        self.margin_ramp_end_epoch = int(
            phase3_end_epoch if margin_ramp_end_epoch is None else margin_ramp_end_epoch
        )

        if penalize_pairs is None:
            self.penalize_pairs = [[1, 2], [0, 1], [0, 2]]
        else:
            self.penalize_pairs = penalize_pairs

        if pair_weights is None:
            if self.penalize_pairs == [[1, 2], [0, 1], [0, 2]]:
                self.pair_weights = [1.0, 0.35, 0.60]
            else:
                self.pair_weights = [1.0 for _ in self.penalize_pairs]
        else:
            if len(pair_weights) != len(self.penalize_pairs):
                raise ValueError(
                    "pair_weights length must match penalize_pairs length: "
                    f"{len(pair_weights)} != {len(self.penalize_pairs)}"
                )
            self.pair_weights = [float(w) for w in pair_weights]

        logger.info(
            f"ConfusionPenaltyLoss (margin-based): "
            f"penalizing pairs {self.penalize_pairs}, "
            f"pair_weights {self.pair_weights}, "
            f"margin ramps {margin_start}->{margin_end} in epochs "
            f"{self.margin_ramp_start_epoch}-{self.margin_ramp_end_epoch}"
        )

    def _get_effective_margin(self, epoch: int) -> float:
        if epoch < self.margin_ramp_start_epoch:
            return 0.0
        if self.margin_ramp_end_epoch <= self.margin_ramp_start_epoch:
            return float(self.margin_end)
        t = min(
            1.0,
            (epoch - self.margin_ramp_start_epoch)
            / max(1, self.margin_ramp_end_epoch - self.margin_ramp_start_epoch),
        )
        return self.margin_start + (self.margin_end - self.margin_start) * t

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        epoch: int = 1,
    ) -> torch.Tensor:
        assert probs.dim() == 4, f"Expected probs shape (B, C, H, W), got {probs.shape}"
        assert targets.dim() == 3, (
            f"Expected targets shape (B, H, W), got {targets.shape}"
        )

        effective_margin = self._get_effective_margin(epoch)

        weighted_penalty_sum = probs.new_tensor(0.0)
        total_pair_weight = 0.0

        for idx, (class_a, class_b) in enumerate(self.penalize_pairs):
            pair_weight = self.pair_weights[idx]
            assert 0 <= class_a < self.num_classes, (
                f"class_a={class_a} out of range [0, {self.num_classes})"
            )
            assert 0 <= class_b < self.num_classes, (
                f"class_b={class_b} out of range [0, {self.num_classes})"
            )
            assert class_a != class_b, f"Cannot penalize class {class_a} against itself"

            pred_a = probs[:, class_a, :, :]  
            pred_b = probs[:, class_b, :, :]  

            target_a = (targets == class_a).float()  
            target_b = (targets == class_b).float()  

            count_a = target_a.sum()
            count_b = target_b.sum()
            total_ab = (count_a + count_b).clamp(min=1.0)

            fp_a_as_b = (pred_b * target_a).sum()
            fp_b_as_a = (pred_a * target_b).sum()
            hard_penalty = (fp_a_as_b + fp_b_as_a) / total_ab

            margin_penalty = probs.new_tensor(0.0)
            if effective_margin > 0:
                margin_violation_a = F.relu(effective_margin - (pred_a - pred_b))
                if count_a > 0:
                    margin_loss_a = (
                        margin_violation_a * target_a
                    ).sum() / count_a.clamp(min=1.0)
                else:
                    margin_loss_a = probs.new_tensor(0.0)

                margin_violation_b = F.relu(effective_margin - (pred_b - pred_a))
                if count_b > 0:
                    margin_loss_b = (
                        margin_violation_b * target_b
                    ).sum() / count_b.clamp(min=1.0)
                else:
                    margin_loss_b = probs.new_tensor(0.0)

                margin_penalty = margin_loss_a + margin_loss_b

            weighted_penalty_sum = weighted_penalty_sum + pair_weight * (
                hard_penalty + margin_penalty
            )
            total_pair_weight += pair_weight

        if total_pair_weight <= 0.0:
            return probs.new_tensor(0.0)

        return weighted_penalty_sum / total_pair_weight
