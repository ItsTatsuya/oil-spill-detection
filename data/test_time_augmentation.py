"""
Test-Time Augmentation (TTA) for oil spill segmentation — PyTorch implementation.

Input / output convention: NCHW tensors [B, C, H, W].
"""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger('oil_spill')


class TestTimeAugmentation:
    """Apply multiple augmentations at inference time and average predictions.

    Parameters
    ----------
    model                : nn.Module  (OilSpillSegformerModel)
    num_augmentations    : int
    use_flips            : bool
    use_scales           : bool
    use_rotations        : bool
    include_original     : bool
    device               : torch.device  (auto-detected when None)
    """

    def __init__(self, model, num_augmentations=8, use_flips=True, use_scales=True,
                 use_rotations=True, include_original=True, device=None):
        self.model              = model
        self.num_augmentations  = num_augmentations
        self.use_flips          = use_flips
        self.use_scales         = use_scales
        self.use_rotations      = use_rotations
        self.include_original   = include_original

        self.scales    = [0.75, 1.0, 1.25] if use_scales else [1.0]
        self.rotations = [0, math.pi/12, -math.pi/12, math.pi/6, -math.pi/6] \
                         if use_rotations else [0]

        self.expected_height   = model.expected_height
        self.expected_width    = model.expected_width
        self.expected_channels = model.expected_channels

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        print(f"TTA configured with {num_augmentations} augmentations")

    # -----------------------------------------------------------------------
    # Per-image augmentation (operates on [C, H, W] tensors)
    # -----------------------------------------------------------------------
    def _augment_image(self, image: torch.Tensor, aug_type: str):
        """Augment a single [C, H, W] image.

        Returns
        -------
        aug_image : [C, H, W]
        aug_info  : dict  (used to reverse the transform later)
        """
        from data.augmentation import rotate_image

        orig_C, orig_H, orig_W = image.shape

        def _resize(img, h, w):
            return F.interpolate(img.unsqueeze(0), size=(h, w),
                                 mode='bilinear', align_corners=False).squeeze(0)

        def _to_model_input(img):
            if img.shape[1] != self.expected_height or img.shape[2] != self.expected_width:
                img = _resize(img, self.expected_height, self.expected_width)
            return img

        if aug_type == 'original':
            img = _to_model_input(image)
            return img, {'type': 'original', 'orig_size': (orig_H, orig_W)}

        elif aug_type == 'h_flip':
            img = torch.flip(image, [2])   # flip W
            img = _to_model_input(img)
            return img, {'type': 'h_flip', 'orig_size': (orig_H, orig_W)}

        elif aug_type == 'v_flip':
            img = torch.flip(image, [1])   # flip H
            img = _to_model_input(img)
            return img, {'type': 'v_flip', 'orig_size': (orig_H, orig_W)}

        elif aug_type == 'h_flip_v_flip':
            img = torch.flip(image, [2])
            img = torch.flip(img, [1])
            img = _to_model_input(img)
            return img, {'type': 'h_flip_v_flip', 'orig_size': (orig_H, orig_W)}

        elif aug_type.startswith('rotate_'):
            angle = float(aug_type.split('_', 1)[1])
            img   = rotate_image(image, angle, method='bilinear')
            img   = _to_model_input(img)
            return img, {'type': 'rotate', 'angle': -angle, 'orig_size': (orig_H, orig_W)}

        elif aug_type.startswith('scale_'):
            scale = float(aug_type.split('_', 1)[1])
            if scale != 1.0:
                sh = max(1, int(round(orig_H * scale)))
                sw = max(1, int(round(orig_W * scale)))
                img = _resize(image, sh, sw)
            else:
                img = image
            img = _to_model_input(img)
            return img, {'type': 'scale', 'scale': scale, 'orig_size': (orig_H, orig_W)}

        else:
            img = _to_model_input(image)
            return img, {'type': 'original', 'orig_size': (orig_H, orig_W)}

    # -----------------------------------------------------------------------
    # Reverse augmentation on prediction map [num_classes, H, W]
    # -----------------------------------------------------------------------
    def _reverse_augmentation(self, pred: torch.Tensor, aug_info: dict) -> torch.Tensor:
        """Invert augmentation applied to a prediction map [C, H, W]."""
        from data.augmentation import rotate_image

        def _resize(p, size):
            return F.interpolate(p.unsqueeze(0), size=size,
                                 mode='bilinear', align_corners=False).squeeze(0)

        aug_type  = aug_info['type']
        orig_size = aug_info.get('orig_size')

        if aug_type == 'original':
            out = _resize(pred, orig_size) if orig_size else pred

        elif aug_type == 'h_flip':
            out = torch.flip(pred, [2])
            if orig_size:
                out = _resize(out, orig_size)

        elif aug_type == 'v_flip':
            out = torch.flip(pred, [1])
            if orig_size:
                out = _resize(out, orig_size)

        elif aug_type == 'h_flip_v_flip':
            out = torch.flip(pred, [2])
            out = torch.flip(out, [1])
            if orig_size:
                out = _resize(out, orig_size)

        elif aug_type == 'rotate':
            angle = aug_info['angle']
            out   = rotate_image(pred, angle, method='bilinear')
            if orig_size:
                out = _resize(out, orig_size)

        elif aug_type == 'scale':
            out = _resize(pred, orig_size) if orig_size else pred

        else:
            out = _resize(pred, orig_size) if orig_size else pred

        return out

    # -----------------------------------------------------------------------
    def _generate_augmentation_types(self):
        aug_types = []
        if self.include_original:
            aug_types.append('original')
        if self.use_flips:
            aug_types.extend(['h_flip', 'v_flip', 'h_flip_v_flip'])
        if self.use_rotations:
            for angle in self.rotations:
                if angle != 0:
                    aug_types.append(f'rotate_{angle}')
        if self.use_scales:
            for scale in self.scales:
                if scale != 1.0:
                    aug_types.append(f'scale_{scale}')

        if len(aug_types) > self.num_augmentations:
            if self.include_original:
                aug_types = ['original'] + [t for t in aug_types if t != 'original'][
                    :self.num_augmentations - 1
                ]
            else:
                aug_types = aug_types[:self.num_augmentations]

        return aug_types

    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply TTA and return averaged log-probability logits.

        Parameters
        ----------
        images : [B, C, H, W]  float32

        Returns
        -------
        logits : [B, num_classes, H, W]  float32  (log-probabilities)
        """
        images = images.to(self.device).float()
        B, C, H, W = images.shape

        # Fix channel mismatch
        if C != self.expected_channels:
            if self.expected_channels == 1 and C > 1:
                images = images.mean(dim=1, keepdim=True)
            elif self.expected_channels == 3 and C == 1:
                images = images.expand(-1, 3, -1, -1)

        aug_types  = self._generate_augmentation_types()
        all_probs  = []

        for aug_type in aug_types:
            try:
                aug_imgs  = []   # list of [C, Hm, Wm] tensors (model-input size)
                aug_infos = []

                for i in range(B):
                    aug_img, aug_info = self._augment_image(images[i], aug_type)
                    aug_imgs.append(aug_img)
                    aug_infos.append(aug_info)

                batch = torch.stack(aug_imgs).to(self.device)  # [B, C, Hm, Wm]
                batch = batch.clamp(0.0, 1.0)

                preds = self.model(batch).float()    # [B, num_classes, Hm, Wm]
                probs = F.softmax(preds, dim=1)      # [B, num_classes, Hm, Wm]

                # Reverse augmentation per image
                rev = []
                for i in range(B):
                    rev.append(self._reverse_augmentation(probs[i], aug_infos[i]))
                all_probs.append(torch.stack(rev))   # [B, num_classes, H, W]

            except Exception as e:
                logger.warning("TTA augmentation '%s' failed: %s", aug_type, e)
                continue

        # Fallback: plain forward pass
        if not all_probs:
            try:
                inp = F.interpolate(images,
                                    size=(self.expected_height, self.expected_width),
                                    mode='bilinear', align_corners=False)
                out = self.model(inp).float()
                if out.shape[-2:] != (H, W):
                    out = F.interpolate(out, size=(H, W),
                                        mode='bilinear', align_corners=False)
                return F.log_softmax(out, dim=1)   # ← convert logits to log-probs
            except Exception:
                num_classes = getattr(self.model, 'num_classes', 5)
                # Return uniform log-probs as a last resort
                fill = -math.log(num_classes)
                return torch.full((B, num_classes, H, W), fill_value=fill,
                                  device=self.device)

        # Average probabilities → return as log-probs
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)   # [B, C, H, W]
        avg_probs = avg_probs.clamp(1e-7, 1.0 - 1e-7)
        return avg_probs.log()
