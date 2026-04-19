from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kornia import morphology as kmorph
except Exception:  # pragma: no cover - optional dependency
    kmorph = None


class BoundaryContrastLoss(nn.Module):
    def __init__(
        self,
        oil_class_idx: int = 1,
        look_alike_class_idx: int = 2,
        boundary_width: int = 3,
        max_samples: int = 256,
        margin: float = 0.3,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.oil_class_idx = int(oil_class_idx)
        self.look_alike_class_idx = int(look_alike_class_idx)
        self.boundary_width = int(boundary_width)
        self.max_samples = int(max_samples)
        self.margin = float(margin)
        self.loss_weight = float(loss_weight)

    def _downsample_labels(
        self,
        target: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        if target.shape[-2:] == size:
            return target.long()
        return (
            F.interpolate(
                target.unsqueeze(1).float(),
                size=size,
                mode="nearest",
            )
            .squeeze(1)
            .long()
        )

    def _dilate(self, mask: torch.Tensor) -> torch.Tensor:
        kernel_size = 2 * self.boundary_width + 1
        if kmorph is not None:
            kernel = torch.ones(
                (kernel_size, kernel_size),
                device=mask.device,
                dtype=mask.dtype,
            )
            return kmorph.dilation(mask, kernel)
        return F.max_pool2d(
            mask, kernel_size=kernel_size, stride=1, padding=self.boundary_width
        )

    def _erode(self, mask: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._dilate(1.0 - mask)

    def _build_boundary_band(self, oil_mask: torch.Tensor) -> torch.Tensor:
        dilated = self._dilate(oil_mask)
        eroded = self._erode(oil_mask)
        return (dilated - eroded).clamp(min=0.0, max=1.0)

    def forward(self, proj_feats: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if proj_feats.dim() != 4:
            raise ValueError(
                f"Expected proj_feats shape [B, D, H, W], got {tuple(proj_feats.shape)}"
            )
        if target.dim() != 3:
            raise ValueError(
                f"Expected target shape [B, H, W], got {tuple(target.shape)}"
            )

        batch_losses = []
        b, d, h, w = proj_feats.shape
        labels_ds = self._downsample_labels(target, (h, w))
        feats_hwd = proj_feats.permute(0, 2, 3, 1)

        for i in range(b):
            mask_i = labels_ds[i]
            oil_region = (
                (mask_i == self.oil_class_idx).float().unsqueeze(0).unsqueeze(0)
            )
            boundary_band = (
                self._build_boundary_band(oil_region).squeeze(0).squeeze(0) > 0.5
            )

            oil_pixels = (mask_i == self.oil_class_idx) & boundary_band
            look_pixels = (mask_i == self.look_alike_class_idx) & boundary_band

            oil_feat = feats_hwd[i][oil_pixels]
            look_feat = feats_hwd[i][look_pixels]
            if oil_feat.numel() == 0 or look_feat.numel() == 0:
                continue

            sample_count = min(self.max_samples, oil_feat.shape[0], look_feat.shape[0])
            oil_perm = torch.randperm(oil_feat.shape[0], device=oil_feat.device)[
                :sample_count
            ]
            look_perm = torch.randperm(look_feat.shape[0], device=look_feat.device)[
                :sample_count
            ]

            oil_feat = oil_feat[oil_perm]
            look_feat = look_feat[look_perm]

            oil_feat = F.normalize(oil_feat, p=2, dim=1)
            look_feat = F.normalize(look_feat, p=2, dim=1)

            cosine_pairs = torch.mm(oil_feat, look_feat.t())
            batch_losses.append(F.relu(cosine_pairs - self.margin).mean())

        if not batch_losses:
            return proj_feats.new_zeros(())

        return self.loss_weight * torch.stack(batch_losses).mean()
