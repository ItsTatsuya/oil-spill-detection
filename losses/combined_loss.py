from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import CLASS_NAMES
from losses.boundary_loss import BoundaryLoss
from losses.confusion_penalty_loss import ConfusionPenaltyLoss
from losses.dice_loss import DiceLoss
from losses.focal_loss import FocalLoss
from losses.lovasz import LovaszSoftmaxLoss


class CombinedLoss(nn.Module):
    def __init__(self, config: dict, pixel_counts: Optional[list] = None) -> None:
        super().__init__()
        loss_cfg = config.get("loss", {})
        if pixel_counts is None:
            dataset_counts = config.get("dataset", {}).get("pixel_counts")
            if isinstance(dataset_counts, dict):
                if all(name in dataset_counts for name in CLASS_NAMES):
                    pixel_counts = [int(dataset_counts[name]) for name in CLASS_NAMES]
            elif isinstance(dataset_counts, (list, tuple)) and len(dataset_counts) == len(
                CLASS_NAMES
            ):
                pixel_counts = [int(value) for value in dataset_counts]
        hybrid_cfg = loss_cfg.get("hybrid_loss_weights", {})
        self.ce_weight = float(hybrid_cfg.get("ce", 1.0))
        self.dice_weight = float(hybrid_cfg.get("dice", 1.0))
        self.jaccard_weight = float(hybrid_cfg.get("jaccard", 0.0))
        self.focal_weight = float(hybrid_cfg.get("focal", 0.0))
        self.ce_ignore_index = int(loss_cfg.get("ce_ignore_index", -100))
        self.ce_label_smoothing = float(loss_cfg.get("ce_label_smoothing", 0.0))
        self.ce_mode = str(loss_cfg.get("ce_mode", "standard")).lower()
        self.ohem_top_fraction = float(loss_cfg.get("ohem_top_fraction", 1.0))
        self.ohem_class_indices = [
            int(idx) for idx in loss_cfg.get("ohem_class_indices", [])
        ]
        self.ohem_start_epoch = int(loss_cfg.get("ohem_start_epoch", 0))
        self.dice_loss = DiceLoss(
            smooth=float(loss_cfg.get("dice", {}).get("smooth", 1.0))
        )
        self.focal_loss = FocalLoss(
            gamma=float(loss_cfg.get("focal", {}).get("gamma", 2.0)),
            alpha=loss_cfg.get("focal", {}).get("alpha", "inverse_frequency"),
            pixel_counts=pixel_counts,
        )
        self.boundary_weight = float(loss_cfg.get("boundary", {}).get("weight", 0.0))
        boundary_cfg = loss_cfg.get("boundary", {})
        self.boundary_ramp_start = int(
            boundary_cfg.get("ramp_start_epoch", 0)
        )
        self.boundary_ramp_end = int(
            boundary_cfg.get("ramp_end_epoch", self.boundary_ramp_start)
        )
        self.boundary_loss = BoundaryLoss(
            num_classes=int(config.get("model", {}).get("num_labels", 5)),
            theta0=int(boundary_cfg.get("theta0", 3)),
            theta=int(boundary_cfg.get("theta", 5)),
            downsample_factor=int(boundary_cfg.get("downsample_factor", 1)),
        )
        confusion_cfg = loss_cfg.get("confusion_penalty", {})
        self.confusion_weight = float(
            confusion_cfg.get("weight", 0.0)
        )
        self.confusion_ramp_start = int(
            confusion_cfg.get("ramp_start_epoch", 0)
        )
        self.confusion_ramp_end = int(
            confusion_cfg.get("ramp_end_epoch", self.confusion_ramp_start)
        )
        self.confusion_loss = ConfusionPenaltyLoss(
            penalize_pairs=confusion_cfg.get("penalize_pairs"),
            pair_weights=confusion_cfg.get("pair_weights"),
            num_classes=int(config.get("model", {}).get("num_labels", 5)),
            margin_start=float(confusion_cfg.get("margin_start", 0.1)),
            margin_end=float(confusion_cfg.get("margin_end", 0.4)),
            margin_ramp_start_epoch=int(
                confusion_cfg.get("margin_ramp_start_epoch", self.confusion_ramp_start)
            ),
            margin_ramp_end_epoch=int(
                confusion_cfg.get("margin_ramp_end_epoch", self.confusion_ramp_end)
            ),
        )
        self.lovasz_weight = float(loss_cfg.get("lovasz", {}).get("weight", 0.0))
        self.lovasz_ramp_start = int(
            loss_cfg.get("lovasz", {}).get("ramp_start_epoch", 0)
        )
        self.lovasz_ramp_end = int(
            loss_cfg.get("lovasz", {}).get("ramp_end_epoch", self.lovasz_ramp_start)
        )
        self.lovasz_loss = LovaszSoftmaxLoss(classes="present", per_image=False)
        auxiliary_cfg = loss_cfg.get("auxiliary", {})
        aux_weights_cfg = auxiliary_cfg.get("weights", {})
        self.auxiliary_enabled = bool(auxiliary_cfg.get("enabled", False))
        self.aux_s16_weight = float(aux_weights_cfg.get("s16", 0.0))
        self.aux_s8_weight = float(aux_weights_cfg.get("s8", 0.0))

    def _scheduled_weight(
        self,
        *,
        epoch: int,
        target_weight: float,
        ramp_start: int,
        ramp_end: int,
    ) -> float:
        if target_weight <= 0.0:
            return 0.0
        if ramp_start <= 0 and ramp_end <= 0:
            return float(target_weight)
        if epoch < ramp_start:
            return 0.0
        if epoch >= ramp_end:
            return float(target_weight)
        progress = float(epoch - ramp_start) / float(max(ramp_end - ramp_start, 1))
        return float(target_weight) * progress

    def _compute_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        epoch: int,
    ) -> torch.Tensor:
        if self.ce_mode != "class_restricted_ohem" or epoch < self.ohem_start_epoch:
            return self._compute_standard_ce_loss(logits, targets)

        per_pixel = F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ce_ignore_index,
            label_smoothing=self.ce_label_smoothing,
            reduction="none",
        )
        valid = targets != self.ce_ignore_index
        if self.ohem_class_indices:
            class_mask = torch.zeros_like(valid, dtype=torch.bool)
            for class_idx in self.ohem_class_indices:
                class_mask |= targets == int(class_idx)
            restricted_valid = valid & class_mask
            if restricted_valid.any():
                valid = restricted_valid
        losses = per_pixel[valid]
        if losses.numel() == 0:
            return self._compute_standard_ce_loss(logits, targets)
        keep = max(int(losses.numel() * self.ohem_top_fraction), 1)
        top_losses, _ = torch.topk(losses, k=min(keep, losses.numel()))
        return top_losses.mean()

    def _compute_standard_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ce_ignore_index,
            label_smoothing=self.ce_label_smoothing,
        )

    def _compute_auxiliary_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        probs = F.softmax(logits, dim=1)
        ce = self._compute_standard_ce_loss(logits, targets)
        dice = self.dice_loss(probs, targets)
        jaccard = (
            self._compute_jaccard_loss(probs, targets)
            if self.jaccard_weight > 0.0
            else logits.new_zeros(())
        )
        return self.ce_weight * ce + self.dice_weight * dice + self.jaccard_weight * jaccard

    def get_boundary_weight(self, epoch: int) -> float:
        return self._scheduled_weight(
            epoch=epoch,
            target_weight=self.boundary_weight,
            ramp_start=self.boundary_ramp_start,
            ramp_end=self.boundary_ramp_end,
        )

    def get_ship_aux_weight(self, epoch: int) -> float:
        return 0.0

    def get_confusion_weight(self, epoch: int) -> float:
        return self._scheduled_weight(
            epoch=epoch,
            target_weight=self.confusion_weight,
            ramp_start=self.confusion_ramp_start,
            ramp_end=self.confusion_ramp_end,
        )

    def get_contrastive_weight(self, epoch: int) -> float:
        return 0.0

    def get_effective_loss_weights(self, epoch: int) -> dict[str, float]:
        return {
            "focal": float(self.focal_weight),
            "dice": float(self.dice_weight),
            "boundary": self.get_boundary_weight(epoch),
            "confusion_penalty": self.get_confusion_weight(epoch),
            "contrastive": 0.0,
            "ship_aux": 0.0,
            "lovasz_main": self._scheduled_weight(
                epoch=epoch,
                target_weight=self.lovasz_weight,
                ramp_start=self.lovasz_ramp_start,
                ramp_end=self.lovasz_ramp_end,
            ),
        }

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: torch.Tensor,
        epoch: int = 1,
        contrastive_weight: Optional[float] = None,
    ) -> dict[str, torch.Tensor | float]:
        if "seg_logits" not in predictions:
            raise KeyError("CombinedLoss expects predictions['seg_logits'].")

        logits = predictions["seg_logits"]
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        probs = F.softmax(logits, dim=1)

        ce = self._compute_ce_loss(logits, targets, epoch=epoch)
        dice = self.dice_loss(probs, targets)
        focal = (
            self.focal_loss(probs, targets)
            if self.focal_weight > 0.0
            else logits.new_zeros(())
        )
        jaccard = (
            self._compute_jaccard_loss(probs, targets)
            if self.jaccard_weight > 0.0
            else logits.new_zeros(())
        )
        lovasz_weight = self._scheduled_weight(
            epoch=epoch,
            target_weight=self.lovasz_weight,
            ramp_start=self.lovasz_ramp_start,
            ramp_end=self.lovasz_ramp_end,
        )
        lovasz = (
            self.lovasz_loss(probs, targets)
            if lovasz_weight > 0.0
            else logits.new_zeros(())
        )
        boundary_weight = self.get_boundary_weight(epoch)
        boundary = (
            self.boundary_loss(probs, targets)
            if boundary_weight > 0.0
            else logits.new_zeros(())
        )
        confusion_weight = self.get_confusion_weight(epoch)
        confusion_penalty = (
            self.confusion_loss(probs, targets, epoch=epoch)
            if confusion_weight > 0.0
            else logits.new_zeros(())
        )

        total = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.focal_weight * focal
            + self.jaccard_weight * jaccard
            + lovasz_weight * lovasz
            + boundary_weight * boundary
            + confusion_weight * confusion_penalty
        )

        zero = logits.new_zeros(())
        aux_s16_loss = zero
        aux_s8_loss = zero
        if self.auxiliary_enabled:
            aux_s16_logits = predictions.get("aux_s16")
            if aux_s16_logits is not None and self.aux_s16_weight > 0.0:
                aux_s16_loss = self._compute_auxiliary_loss(aux_s16_logits, targets)
                total = total + self.aux_s16_weight * aux_s16_loss
            aux_s8_logits = predictions.get("aux_s8")
            if aux_s8_logits is not None and self.aux_s8_weight > 0.0:
                aux_s8_loss = self._compute_auxiliary_loss(aux_s8_logits, targets)
                total = total + self.aux_s8_weight * aux_s8_loss

        return {
            "total": total,
            "ce": ce.detach(),
            "focal": focal.detach(),
            "dice": dice.detach(),
            "jaccard": jaccard.detach(),
            "lovasz": lovasz.detach(),
            "boundary": boundary.detach(),
            "confusion_penalty": confusion_penalty.detach(),
            "contrastive": zero.detach(),
            "boundary_contrast": zero.detach(),
            "boundary_contrast_loss": zero.detach(),
            "ship_aux": zero.detach(),
            "aux_s8": aux_s8_loss.detach(),
            "aux_s16": aux_s16_loss.detach(),
            "ce_norm": ce.detach(),
            "focal_norm": focal.detach(),
            "dice_norm": dice.detach(),
            "jaccard_norm": jaccard.detach(),
            "lovasz_norm": lovasz.detach(),
            "boundary_norm": boundary.detach(),
            "confusion_norm": confusion_penalty.detach(),
            "boundary_weight_used": float(boundary_weight),
            "confusion_weight_used": float(confusion_weight),
            "hybrid_ce_weight_used": float(self.ce_weight),
            "hybrid_dice_weight_used": float(self.dice_weight),
            "hybrid_jaccard_weight_used": float(self.jaccard_weight),
            "hybrid_focal_weight_used": float(self.focal_weight),
            "hybrid_lovasz_weight_used": float(lovasz_weight),
        }

    def _compute_jaccard_loss(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        num_classes = probs.shape[1]
        one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        union = (probs + one_hot - probs * one_hot).sum(dim=(0, 2, 3)).clamp(min=1e-6)
        return (1.0 - (intersection / union)).mean()
