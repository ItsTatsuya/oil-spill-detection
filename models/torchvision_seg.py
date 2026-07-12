"""Torchvision segmentation backbones (DeepLabV3) with unified oil-spill API."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

logger = logging.getLogger(__name__)

_TV_BUILDERS = {
    "deeplabv3_resnet50": deeplabv3_resnet50,
    "deeplabv3_resnet101": deeplabv3_resnet101,
    "deeplabv3_mobilenet_v3_large": deeplabv3_mobilenet_v3_large,
}

# Backbone feature channel count feeding DeepLabHead / FCNHead.
_BACKBONE_OUT_CHANNELS = {
    "deeplabv3_resnet50": (2048, 1024),
    "deeplabv3_resnet101": (2048, 1024),
    "deeplabv3_mobilenet_v3_large": (960, 40),
}


def _adapt_first_conv_to_channels(
    conv: nn.Conv2d,
    num_channels: int,
    *,
    channel_init: str = "rgb_mean_inflate",
) -> nn.Conv2d:
    """Return a new Conv2d with ``num_channels`` inputs, initialized from RGB weights."""
    if conv.weight.shape[1] == num_channels:
        return conv
    if conv.weight.shape[1] != 3:
        raise ValueError(
            f"Expected pretrained stem with 3 input channels, got {conv.weight.shape[1]}"
        )
    if num_channels < 1:
        raise ValueError(f"num_channels must be positive, got {num_channels}")

    new_conv = nn.Conv2d(
        num_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        w = conv.weight.detach().clone()
        if channel_init == "rgb_mean_inflate":
            if num_channels == 1:
                new_conv.weight.copy_(w.mean(dim=1, keepdim=True))
            elif num_channels == 3:
                new_conv.weight.copy_(w)
            else:
                mean_ch = w.mean(dim=1, keepdim=True)
                extras = mean_ch.repeat(1, num_channels - 3, 1, 1)
                new_conv.weight.copy_(torch.cat([w, extras], dim=1))
        elif channel_init == "repeat":
            # Average then repeat (grayscale-style).
            mean_ch = w.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_ch.repeat(1, num_channels, 1, 1))
        else:
            raise ValueError(f"Unsupported channel_init: {channel_init}")
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias.detach())
    return new_conv


def _replace_classifier_heads(
    model: nn.Module,
    *,
    architecture: str,
    num_labels: int,
    aux_loss: bool,
) -> None:
    main_ch, aux_ch = _BACKBONE_OUT_CHANNELS[architecture]
    model.classifier = DeepLabHead(main_ch, num_labels)
    if aux_loss and model.aux_classifier is not None:
        model.aux_classifier = FCNHead(aux_ch, num_labels)
    elif not aux_loss:
        model.aux_classifier = None


class TorchvisionSegmentationModel(nn.Module):
    """Wrap torchvision segmentation models to return ``seg_logits``."""

    def __init__(self, model: nn.Module, architecture: str) -> None:
        super().__init__()
        self.architecture = architecture
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.model(pixel_values)
        logits = outputs["out"]
        if logits.shape[-2:] != pixel_values.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        result: dict[str, torch.Tensor] = {"seg_logits": logits}
        aux = outputs.get("aux")
        if aux is not None:
            if aux.shape[-2:] != pixel_values.shape[-2:]:
                aux = F.interpolate(
                    aux,
                    size=pixel_values.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            result["aux_logits"] = aux
        return result


def build_torchvision_segmentation_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    architecture = str(model_cfg.get("architecture", "deeplabv3_resnet50")).lower()
    if architecture not in _TV_BUILDERS:
        raise ValueError(
            f"Unsupported torchvision architecture '{architecture}'. "
            f"Supported: {sorted(_TV_BUILDERS)}"
        )

    num_labels = int(model_cfg.get("num_labels", 5))
    num_channels = int(model_cfg.get("num_channels", 3))
    channel_init = str(model_cfg.get("channel_init", "rgb_mean_inflate"))
    pretrained = bool(model_cfg.get("pretrained", True))
    aux_loss = bool(model_cfg.get("aux_loss", True))

    builder = _TV_BUILDERS[architecture]
    weights: Optional[object] = "DEFAULT" if pretrained else None
    logger.info(
        "Building %s (pretrained=%s, num_labels=%d, num_channels=%d, aux_loss=%s).",
        architecture,
        pretrained,
        num_labels,
        num_channels,
        aux_loss,
    )
    try:
        tv_model = builder(weights=weights, aux_loss=aux_loss)
    except TypeError:
        # Older torchvision may not accept weights= string the same way.
        tv_model = builder(pretrained=pretrained, aux_loss=aux_loss)

    # Adapt stem for non-RGB channel counts.
    if architecture.startswith("deeplabv3_resnet"):
        tv_model.backbone.conv1 = _adapt_first_conv_to_channels(
            tv_model.backbone.conv1,
            num_channels,
            channel_init=channel_init,
        )
    elif architecture == "deeplabv3_mobilenet_v3_large":
        # IntermediateLayerGetter: backbone['0'] is Conv2dNormActivation; [0] is Conv2d.
        stem_seq = tv_model.backbone["0"]
        stem_seq[0] = _adapt_first_conv_to_channels(
            stem_seq[0],
            num_channels,
            channel_init=channel_init,
        )
    else:
        raise RuntimeError(f"No stem adapt path for {architecture}")

    _replace_classifier_heads(
        tv_model,
        architecture=architecture,
        num_labels=num_labels,
        aux_loss=aux_loss,
    )

    return TorchvisionSegmentationModel(tv_model, architecture=architecture)


def is_torchvision_architecture(architecture: str) -> bool:
    return str(architecture).lower() in _TV_BUILDERS
