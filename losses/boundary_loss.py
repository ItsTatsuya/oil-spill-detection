import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def extract_boundaries(
    mask: torch.Tensor,
    kernel_size: int,
    num_classes: int,
) -> torch.Tensor:
    B, H, W = mask.shape

    invalid_mask = (mask < 0) | (mask >= num_classes)

    safe_mask = mask.long().clamp(0, num_classes - 1)
    one_hot = F.one_hot(safe_mask, num_classes).float()
    one_hot = one_hot.permute(0, 3, 1, 2)

    one_hot = one_hot * (~invalid_mask).unsqueeze(1).float()

    padding = kernel_size // 2

    one_hot_flat = one_hot.reshape(B * num_classes, 1, H, W)

    eroded_flat = -F.max_pool2d(
        -one_hot_flat,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    eroded = eroded_flat.reshape(B, num_classes, H, W)

    boundary = (one_hot - eroded).clamp(0.0, 1.0)

    return boundary


class BoundaryLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        theta0: int = 3,
        theta: int = 5,
        downsample_factor: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.theta0 = theta0
        self.theta = theta
        self.downsample_factor = int(downsample_factor)

        assert theta0 % 2 == 1, f"theta0 must be odd, got {theta0}"
        assert theta % 2 == 1, f"theta must be odd, got {theta}"
        assert self.downsample_factor >= 1, (
            f"downsample_factor must be >= 1, got {self.downsample_factor}"
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

        if self.downsample_factor > 1:
            down_h = max(probs.shape[-2] // self.downsample_factor, 1)
            down_w = max(probs.shape[-1] // self.downsample_factor, 1)
            down_size = (down_h, down_w)
            probs = F.interpolate(
                probs.float(),
                size=down_size,
                mode="bilinear",
                align_corners=False,
            )
            targets = (
                F.interpolate(
                    targets.unsqueeze(1).float(),
                    size=down_size,
                    mode="nearest",
                )
                .squeeze(1)
                .long()
            )

        boundary_fine = extract_boundaries(targets, self.theta0, self.num_classes)
        boundary_coarse = extract_boundaries(targets, self.theta, self.num_classes)

        boundary = (boundary_fine + boundary_coarse).clamp(0.0, 1.0)
        boundary_mass = boundary.sum(dim=1, keepdim=True)
        boundary_pixels = boundary_mass > 0

        if not boundary_pixels.any():
            return probs.new_tensor(0.0)

        target_dist = boundary / boundary_mass.clamp(min=1.0)

        with torch.amp.autocast(probs.device.type, enabled=False):
            per_class_kl = F.kl_div(
                probs.float().clamp(min=1e-6).log(),
                target_dist.float(),
                reduction="none",
            )
            per_pixel_kl = per_class_kl.sum(dim=1, keepdim=True)

        boundary_loss = (
            per_pixel_kl * boundary_pixels.float()
        ).sum() / boundary_pixels.float().sum().clamp(min=1.0)

        return boundary_loss
