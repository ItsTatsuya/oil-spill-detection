from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SEA_CLASS = 0
OIL_CLASS = 1
LOOKALIKE_CLASS = 2


class OilLookalikeContrastiveLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 0.07,
        min_pixels: int = 4,
        max_pixels: int = 256,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def _sample_class_pixels(
        self,
        proj_feats: torch.Tensor,
        labels: torch.Tensor,
        class_idx: int,
    ) -> torch.Tensor | None:
        _, _, h, w = proj_feats.shape
        labels_ds = F.interpolate(
            labels.float().unsqueeze(1),
            size=(h, w),
            mode="nearest",
        ).squeeze(1).long()
        class_mask = labels_ds == class_idx
        if not class_mask.any():
            return None

        feats_hwd = proj_feats.permute(0, 2, 3, 1)
        selected = feats_hwd[class_mask]
        n = selected.shape[0]
        if n > self.max_pixels:
            idx = torch.randperm(n, device=selected.device)[: self.max_pixels]
            selected = selected[idx]
        return selected

    def forward(self, proj_feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        proj_norm = F.normalize(proj_feats, dim=1)

        oil_feats = self._sample_class_pixels(proj_norm, labels, OIL_CLASS)
        la_feats = self._sample_class_pixels(proj_norm, labels, LOOKALIKE_CLASS)

        if oil_feats is None or la_feats is None:
            return proj_feats.sum() * 0.0
        if oil_feats.shape[0] < self.min_pixels or la_feats.shape[0] < self.min_pixels:
            return proj_feats.sum() * 0.0

        oil_sim = (oil_feats @ oil_feats.T) / self.temperature
        cross_sim = (oil_feats @ la_feats.T) / self.temperature

        n_oil = oil_feats.shape[0]
        eye_mask = torch.eye(n_oil, dtype=torch.bool, device=oil_feats.device)

        losses = []
        for i in range(n_oil):
            pos = oil_sim[i][~eye_mask[i]]
            neg = cross_sim[i]
            if pos.numel() == 0:
                continue
            pos_exp = torch.exp(pos).sum()
            neg_exp = torch.exp(neg).sum()
            losses.append(-torch.log(pos_exp / (pos_exp + neg_exp + 1e-8)))

        if not losses:
            return proj_feats.sum() * 0.0
        return torch.stack(losses).mean()


@dataclass(frozen=True)
class ContrastiveBatch:
    feats: torch.Tensor
    labels: torch.Tensor


class ConfusionAwareContrastiveLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 0.07,
        min_pixels: int = 4,
        sea_max_pixels: int = 256,
        oil_max_pixels: int = 256,
        look_max_pixels: int = 256,
        boundary_negative_margin_px: int = 64,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.min_pixels = int(min_pixels)
        self.sea_max_pixels = int(sea_max_pixels)
        self.oil_max_pixels = int(oil_max_pixels)
        self.look_max_pixels = int(look_max_pixels)
        self.boundary_negative_margin_px = int(boundary_negative_margin_px)

    def forward(
        self,
        proj_feats: torch.Tensor,
        labels: torch.Tensor,
        seg_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if proj_feats is None:
            return labels.sum() * 0.0

        proj_norm = F.normalize(proj_feats, dim=1)
        labels_ds = self._resize_labels(labels, proj_feats.shape[-2:])
        probs_ds = self._resize_probs(seg_logits, proj_feats.shape[-2:]) if seg_logits is not None else None

        oil_feats = self._sample_class_pixels(
            proj_norm,
            labels_ds == OIL_CLASS,
            self.oil_max_pixels,
        )
        look_feats = self._sample_class_pixels(
            proj_norm,
            labels_ds == LOOKALIKE_CLASS,
            self.look_max_pixels,
        )
        sea_feats = self._sample_class_pixels(
            proj_norm,
            self._build_hard_sea_mask(labels_ds, probs_ds, labels.shape[-2:]),
            self.sea_max_pixels,
        )

        feature_groups: list[ContrastiveBatch] = []
        if oil_feats is not None and oil_feats.shape[0] >= self.min_pixels:
            feature_groups.append(
                ContrastiveBatch(
                    feats=oil_feats,
                    labels=torch.full(
                        (oil_feats.shape[0],),
                        OIL_CLASS,
                        device=oil_feats.device,
                        dtype=torch.long,
                    ),
                )
            )
        if look_feats is not None and look_feats.shape[0] >= self.min_pixels:
            feature_groups.append(
                ContrastiveBatch(
                    feats=look_feats,
                    labels=torch.full(
                        (look_feats.shape[0],),
                        LOOKALIKE_CLASS,
                        device=look_feats.device,
                        dtype=torch.long,
                    ),
                )
            )
        if sea_feats is not None and sea_feats.shape[0] >= self.min_pixels:
            feature_groups.append(
                ContrastiveBatch(
                    feats=sea_feats,
                    labels=torch.full(
                        (sea_feats.shape[0],),
                        SEA_CLASS,
                        device=sea_feats.device,
                        dtype=torch.long,
                    ),
                )
            )

        if len(feature_groups) < 2:
            return proj_feats.sum() * 0.0

        feats = torch.cat([group.feats for group in feature_groups], dim=0)
        class_labels = torch.cat([group.labels for group in feature_groups], dim=0)
        return self._supervised_contrastive(feats, class_labels)

    def _resize_labels(
        self,
        labels: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        return (
            F.interpolate(labels.float().unsqueeze(1), size=size, mode="nearest")
            .squeeze(1)
            .long()
        )

    def _resize_probs(
        self,
        seg_logits: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        probs = (
            seg_logits
            if self._looks_like_probabilities(seg_logits)
            else F.softmax(seg_logits.detach(), dim=1)
        )
        if probs.shape[-2:] != size:
            probs = F.interpolate(
                probs,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        return probs.detach()

    def _build_hard_sea_mask(
        self,
        labels_ds: torch.Tensor,
        probs_ds: Optional[torch.Tensor],
        original_hw: tuple[int, int],
    ) -> torch.Tensor:
        sea_mask = labels_ds == SEA_CLASS
        fg_mask = (labels_ds == OIL_CLASS) | (labels_ds == LOOKALIKE_CLASS)

        if not sea_mask.any():
            return sea_mask

        height, width = labels_ds.shape[-2:]
        margin_y = max(
            1,
            int(round(self.boundary_negative_margin_px * height / max(original_hw[0], 1))),
        )
        margin_x = max(
            1,
            int(round(self.boundary_negative_margin_px * width / max(original_hw[1], 1))),
        )
        margin = max(margin_y, margin_x)
        kernel = 2 * margin + 1

        near_boundary = (
            F.max_pool2d(
                fg_mask.float().unsqueeze(1),
                kernel_size=kernel,
                stride=1,
                padding=margin,
            )
            .squeeze(1)
            > 0
        )
        near_boundary = near_boundary & sea_mask

        confusing_top2 = sea_mask.new_zeros(sea_mask.shape, dtype=torch.bool)
        if probs_ds is not None:
            top2 = probs_ds.topk(k=min(2, probs_ds.shape[1]), dim=1).indices
            confusing_top2 = sea_mask & (
                (top2 == OIL_CLASS).any(dim=1) | (top2 == LOOKALIKE_CLASS).any(dim=1)
            )

        return near_boundary | confusing_top2

    def _sample_class_pixels(
        self,
        proj_feats: torch.Tensor,
        class_mask: torch.Tensor,
        max_pixels: int,
    ) -> torch.Tensor | None:
        if max_pixels <= 0 or not class_mask.any():
            return None

        feats_hwd = proj_feats.permute(0, 2, 3, 1)
        selected = feats_hwd[class_mask]
        count = selected.shape[0]
        if count > max_pixels:
            indices = torch.randperm(count, device=selected.device)[:max_pixels]
            selected = selected[indices]
        return selected

    def _supervised_contrastive(
        self,
        feats: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        if feats.shape[0] < 2:
            return feats.sum() * 0.0

        logits = torch.matmul(feats, feats.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        logits_mask = ~torch.eye(feats.shape[0], dtype=torch.bool, device=feats.device)
        positive_mask = (
            class_labels.unsqueeze(0) == class_labels.unsqueeze(1)
        ) & logits_mask
        positive_counts = positive_mask.sum(dim=1)
        if not (positive_counts > 0).any():
            return feats.sum() * 0.0

        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (
            positive_mask.float() * log_prob
        ).sum(dim=1) / positive_counts.clamp(min=1).float()

        valid = positive_counts > 0
        return -mean_log_prob_pos[valid].mean()

    def _looks_like_probabilities(self, tensor: torch.Tensor) -> bool:
        sample = tensor.detach().float()
        return bool(
            sample.min() >= -1e-3
            and sample.max() <= 1.0 + 1e-3
            and (sample.sum(dim=1) - 1.0).abs().max() < 1e-3
        )
