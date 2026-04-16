import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

VALID_AUG_TYPES = [
    "original",
    "horizontal_flip",
    "vertical_flip",
    "both_flips",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "hflip_rotate_90",
    "hflip_rotate_270",
    "rot90",
    "rot180",
    "rot270",
    "rot90_hflip",
]


class TestTimeAugmentation:

    def __init__(
        self,
        augmentation_types: Optional[List[str]] = None,
        aggregation: str = "mean_softmax",
        device: Optional[torch.device] = None,
    ) -> None:
        if augmentation_types is None:
            augmentation_types = VALID_AUG_TYPES

        for aug in augmentation_types:
            assert aug in VALID_AUG_TYPES, (
                f"Unknown augmentation type: {aug}. Valid: {VALID_AUG_TYPES}"
            )

        self.augmentation_types = augmentation_types
        self.aggregation = aggregation
        self.device = device

        logger.debug(
            f"TTA initialized with {len(augmentation_types)} augmentations: "
            f"{augmentation_types}"
        )

    def apply_augmentation(self, image: torch.Tensor, aug_type: str) -> torch.Tensor:
        if aug_type == "original":
            return image
        elif aug_type == "horizontal_flip":
            return torch.flip(image, dims=[-1])  
        elif aug_type == "vertical_flip":
            return torch.flip(image, dims=[-2])  
        elif aug_type == "both_flips":
            return torch.flip(image, dims=[-1, -2])
        elif aug_type in ("rot90", "rotate_90"):
            return torch.rot90(image, k=1, dims=[-2, -1])  
        elif aug_type in ("rot180", "rotate_180"):
            return torch.rot90(image, k=2, dims=[-2, -1])  
        elif aug_type in ("rot270", "rotate_270"):
            return torch.rot90(image, k=3, dims=[-2, -1])  
        elif aug_type == "rot90_hflip":
            rotated = torch.rot90(image, k=1, dims=[-2, -1])
            return torch.flip(rotated, dims=[-1])
        elif aug_type == "hflip_rotate_90":
            flipped = torch.flip(image, dims=[-1])
            return torch.rot90(flipped, k=1, dims=[-2, -1])
        elif aug_type == "hflip_rotate_270":
            flipped = torch.flip(image, dims=[-1])
            return torch.rot90(flipped, k=3, dims=[-2, -1])
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")

    def reverse_augmentation(
        self, prediction: torch.Tensor, aug_type: str
    ) -> torch.Tensor:
        if aug_type == "original":
            return prediction
        elif aug_type == "horizontal_flip":
            return torch.flip(prediction, dims=[-1])  
        elif aug_type == "vertical_flip":
            return torch.flip(prediction, dims=[-2])  
        elif aug_type == "both_flips":
            return torch.flip(prediction, dims=[-1, -2])  
        elif aug_type in ("rot90", "rotate_90"):
            return torch.rot90(prediction, k=3, dims=[-2, -1])
        elif aug_type in ("rot180", "rotate_180"):
            return torch.rot90(prediction, k=2, dims=[-2, -1])
        elif aug_type in ("rot270", "rotate_270"):
            return torch.rot90(prediction, k=1, dims=[-2, -1])
        elif aug_type == "rot90_hflip":
            flipped = torch.flip(prediction, dims=[-1])
            return torch.rot90(flipped, k=3, dims=[-2, -1])
        elif aug_type == "hflip_rotate_90":
            rotated_back = torch.rot90(prediction, k=3, dims=[-2, -1])
            return torch.flip(rotated_back, dims=[-1])
        elif aug_type == "hflip_rotate_270":
            rotated_back = torch.rot90(prediction, k=1, dims=[-2, -1])
            return torch.flip(rotated_back, dims=[-1])
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        image_tensor: torch.Tensor,
    ) -> torch.Tensor:
        assert image_tensor.dim() == 4, (
            f"Expected (B, C, H, W) input, got {image_tensor.shape}"
        )

        model.eval()
        device = self.device or next(model.parameters()).device
        image_tensor = image_tensor.to(device)

        accumulated_probs = None

        for aug_type in self.augmentation_types:
            aug_image = self.apply_augmentation(image_tensor, aug_type)

            output = model(aug_image)
            logits = output["seg_logits"]
            probs = (
                logits
                if self._looks_like_probabilities(logits)
                else F.softmax(logits, dim=1)
            )

            if probs.shape[-2:] != image_tensor.shape[-2:]:
                probs = F.interpolate(
                    probs,
                    size=image_tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            probs_reversed = self.reverse_augmentation(probs, aug_type)

            if accumulated_probs is None:
                accumulated_probs = probs_reversed
            else:
                accumulated_probs = accumulated_probs + probs_reversed

        averaged_probs = accumulated_probs / len(self.augmentation_types)
        return averaged_probs

    def _looks_like_probabilities(self, tensor: torch.Tensor) -> bool:
        sample = tensor.detach().float()
        return bool(
            sample.min() >= -1e-3
            and sample.max() <= 1.0 + 1e-3
            and (sample.sum(dim=1) - 1.0).abs().max() < 5e-2
        )
