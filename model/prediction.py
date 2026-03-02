"""
Shared multi-scale predictor with optional TTA for oil spill segmentation — PyTorch.

Input / output format: NCHW tensors throughout.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger('oil_spill')


class MultiScalePredictor:
    """Multi-scale prediction with optional Test-Time Augmentation (TTA).

    Parameters
    ----------
    model       : nn.Module  (OilSpillSegformerModel)
    scales      : list of float
    batch_size  : int
    use_tta     : bool
    tta_num_augmentations : int
    device      : torch.device  (auto-detected when None)
    """

    def __init__(
        self,
        model,
        scales=(0.75, 1.0, 1.25),
        batch_size=4,
        use_tta=False,
        tta_num_augmentations=8,
        device=None,
    ):
        self.model = model
        self.scales = list(scales)
        self.batch_size = batch_size
        self.use_tta = use_tta

        self.expected_height   = model.expected_height
        self.expected_width    = model.expected_width
        self.expected_channels = model.expected_channels

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        if self.use_tta:
            from data.test_time_augmentation import TestTimeAugmentation
            self.tta = TestTimeAugmentation(
                model,
                num_augmentations=tta_num_augmentations,
                use_flips=True,
                use_scales=False,
                use_rotations=True,
                include_original=True,
                device=device,
            )

    @torch.inference_mode()
    def _predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Run a single forward pass. batch: [B, C, H, W]."""
        return self.model(batch)

    @torch.inference_mode()
    def predict(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale inference.

        Parameters
        ----------
        image_batch : [B, C, H, W]  float32, values in [0, 1]

        Returns
        -------
        logits : [B, num_classes, H, W]  float32
        """
        image_batch = image_batch.to(self.device).float()
        B, C, H, W = image_batch.shape

        # Fix channel mismatch
        if C != self.expected_channels:
            if self.expected_channels == 1 and C > 1:
                image_batch = image_batch.mean(dim=1, keepdim=True)
            elif self.expected_channels == 3 and C == 1:
                image_batch = image_batch.expand(-1, 3, -1, -1)

        all_probs = []

        for scale in self.scales:
            try:
                # Scale → round to multiple of 8 for SegFormer
                sh = max(8, int(round(H * scale / 8)) * 8)
                sw = max(8, int(round(W * scale / 8)) * 8)

                scaled = F.interpolate(image_batch, size=(sh, sw),
                                       mode='bilinear', align_corners=False)
                model_input = F.interpolate(
                    scaled,
                    size=(self.expected_height, self.expected_width),
                    mode='bilinear', align_corners=False,
                )

                if self.use_tta:
                    logits = self.tta.predict(model_input)
                else:
                    logits = self._predict_batch(model_input)

                logits = logits.float()

                # Resize back to original resolution if needed
                if logits.shape[-2:] != (H, W):
                    logits = F.interpolate(logits, size=(H, W),
                                           mode='bilinear', align_corners=False)

                # tta.predict() returns log-probs; _predict_batch returns raw logits
                if self.use_tta:
                    all_probs.append(torch.exp(logits))
                else:
                    all_probs.append(F.softmax(logits, dim=1))

            except Exception as e:
                logger.warning("Scale %.2f failed: %s", scale, e)
                continue

        # Fallback: plain forward pass at expected resolution
        if not all_probs:
            try:
                inp = F.interpolate(
                    image_batch,
                    size=(self.expected_height, self.expected_width),
                    mode='bilinear', align_corners=False,
                )
                logits = self.model(inp).float()
                if logits.shape[-2:] != (H, W):
                    logits = F.interpolate(logits, size=(H, W),
                                           mode='bilinear', align_corners=False)
                # Return log-probs for consistency with normal path
                return F.log_softmax(logits, dim=1)
            except Exception:
                num_classes = getattr(self.model, 'num_classes', 5)
                return torch.full((B, num_classes, H, W), fill_value=float('-inf'),
                                  device=self.device)

        # Average probabilities → log-prob as logits
        fused = torch.stack(all_probs, dim=0).mean(dim=0)  # [B, C, H, W]
        fused = fused.clamp(1e-7, 1.0 - 1e-7)
        return fused.log()


