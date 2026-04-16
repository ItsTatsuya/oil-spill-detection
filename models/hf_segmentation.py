from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    MobileViTConfig,
    MobileViTForSemanticSegmentation,
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SegformerModel,
)

from models.segformer_oilspill_hybrid import SegformerOilSpillHybridModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HFModelSpec:
    architecture: str
    config_cls: type
    model_cls: type
    first_conv_key: str
    head_prefixes: tuple[str, ...]
    backbone_prefixes: tuple[str, ...]


MODEL_SPECS: dict[str, HFModelSpec] = {
    "segformer": HFModelSpec(
        architecture="segformer",
        config_cls=SegformerConfig,
        model_cls=SegformerForSemanticSegmentation,
        first_conv_key="segformer.encoder.patch_embeddings.0.proj.weight",
        head_prefixes=("decode_head.classifier.",),
        backbone_prefixes=("segformer.",),
    ),
    "mobilevit": HFModelSpec(
        architecture="mobilevit",
        config_cls=MobileViTConfig,
        model_cls=MobileViTForSemanticSegmentation,
        first_conv_key="mobilevit.conv_stem.convolution.weight",
        head_prefixes=("segmentation_head.classifier.",),
        backbone_prefixes=("mobilevit.",),
    ),
}

CUSTOM_SEGFORMER_HEAD_PREFIXES: tuple[str, ...] = (
    "decoder.",
    "enhancers.",
    "aspp.",
    "fuse_s16.",
    "fuse_s8.",
    "fuse_s4.",
    "esem_s16.",
    "esem_s8.",
    "esem_s4.",
    "aux_head_s16.",
    "aux_head_s8.",
    "projections.",
    "fuse.",
    "dropout.",
    "classifier.",
    "decode_head.",
)


def inflate_rgb_weight_with_mean_channel(
    weight: torch.Tensor,
    num_channels: int,
) -> torch.Tensor:
    if weight.ndim != 4:
        raise ValueError(f"Expected 4D conv weight, got shape {tuple(weight.shape)}")
    if weight.shape[1] != 3:
        raise ValueError(
            f"Expected pretrained RGB weight with 3 input channels, got {weight.shape[1]}"
        )
    if num_channels < 3:
        raise ValueError(
            f"rgb_mean_inflate requires at least 3 channels, got {num_channels}"
        )
    if num_channels == 3:
        return weight.detach().clone()

    extra_channels = num_channels - 3
    mean_channel = weight.mean(dim=1, keepdim=True)
    extras = mean_channel.repeat(1, extra_channels, 1, 1)
    return torch.cat([weight.detach().clone(), extras], dim=1)


def adapt_pretrained_state_dict(
    pretrained_state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
    *,
    first_conv_key: str,
    num_channels: int,
    channel_init: str,
    head_prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    if channel_init != "rgb_mean_inflate":
        raise ValueError(f"Unsupported channel init policy: {channel_init}")

    adapted: dict[str, torch.Tensor] = {}
    for key, value in pretrained_state_dict.items():
        if key.startswith(head_prefixes):
            continue
        if key == first_conv_key:
            adapted[key] = inflate_rgb_weight_with_mean_channel(value, num_channels)
            continue
        target = model_state_dict.get(key)
        if target is None or target.shape != value.shape:
            continue
        adapted[key] = value
    return adapted


def _resolve_label_maps(num_labels: int) -> tuple[dict[int, str], dict[str, int]]:
    id2label = {idx: str(idx) for idx in range(num_labels)}
    label2id = {label: idx for idx, label in id2label.items()}
    return id2label, label2id


def _repo_folder_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def _map_segformer_pretrained_key_for_custom_model(key: str) -> str:
    if key.startswith("segformer."):
        return f"encoder.{key[len('segformer.') :]}"
    return key


def _resolve_pretrained_source(config: dict) -> tuple[str, bool]:
    model_cfg = config.get("model", {})
    pretrained_name = str(model_cfg["pretrained_name"])
    cache_root = Path(str(model_cfg.get("pretrained_cache_dir", "models/pretrained")))
    require_local = bool(model_cfg.get("require_local_pretrained", True))

    candidates: list[Path] = []
    explicit_local = model_cfg.get("pretrained_local_dir")
    if explicit_local:
        candidates.append(Path(str(explicit_local)))
    candidates.append(cache_root / _repo_folder_name(pretrained_name))
    candidates.append(cache_root / pretrained_name.split("/")[-1])

    for candidate in candidates:
        if (candidate / "config.json").exists():
            logger.info(
                "Loading pretrained '%s' from local path '%s'.",
                pretrained_name,
                candidate,
            )
            return str(candidate), True

    if require_local:
        searched = "\n- ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Local pretrained model not found. Searched:\n- "
            + searched
            + "\nDownload it first, for example:\n"
            + "python download_pretrained.py --config configs/segformer_sar.yaml"
        )

    logger.warning(
        "Local pretrained '%s' not found. Falling back to Hugging Face hub.",
        pretrained_name,
    )
    return pretrained_name, False


def _build_hf_config(
    spec: HFModelSpec,
    config: dict,
    pretrained_source: str,
    local_files_only: bool,
) -> object:
    model_cfg = config.get("model", {})
    num_labels = int(model_cfg.get("num_labels", 5))
    num_channels = int(model_cfg.get("num_channels", 4))

    hf_config = spec.config_cls.from_pretrained(
        pretrained_source,
        local_files_only=local_files_only,
    )
    hf_config.num_labels = num_labels
    hf_config.num_channels = num_channels
    hf_config.return_dict = True
    hf_config.output_hidden_states = False
    hf_config.semantic_loss_ignore_index = int(
        config.get("loss", {}).get("ce_ignore_index", -100)
    )
    id2label, label2id = _resolve_label_maps(num_labels)
    hf_config.id2label = id2label
    hf_config.label2id = label2id
    return hf_config


class HFSegmentationModel(nn.Module):
    def __init__(self, architecture: str, model: nn.Module) -> None:
        super().__init__()
        self.architecture = architecture
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits
        if logits.shape[-2:] != pixel_values.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return {"seg_logits": logits}


class CustomSegformerModel(nn.Module):
    def __init__(
        self,
        config: SegformerConfig,
        decoder_dim: int = 256,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = SegformerModel(config)

        hidden_sizes = list(config.hidden_sizes)
        self.projections = nn.ModuleList(
            [nn.Conv2d(ch, decoder_dim, kernel_size=1) for ch in hidden_sizes]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                decoder_dim * len(hidden_sizes),
                decoder_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=max(
                    i
                    for i in range(min(32, decoder_dim), 0, -1)
                    if decoder_dim % i == 0
                ),
                num_channels=decoder_dim,
            ),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(p=float(dropout_prob))
        self.classifier = nn.Conv2d(decoder_dim, int(config.num_labels), kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) == 0:
            raise RuntimeError("SegFormer encoder returned no hidden states.")

        target_size = hidden_states[0].shape[-2:]
        projected: list[torch.Tensor] = []
        for feature_map, projection in zip(hidden_states, self.projections):
            x = projection(feature_map)
            if x.shape[-2:] != target_size:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(x)

        fused = self.fuse(torch.cat(projected, dim=1))
        logits = self.classifier(self.dropout(fused))
        if logits.shape[-2:] != pixel_values.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return {"seg_logits": logits}


def _convert_segformer_pretrained_backbone_keys(
    pretrained_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in pretrained_state_dict.items():
        if not key.startswith("segformer.encoder."):
            continue
        converted[_map_segformer_pretrained_key_for_custom_model(key)] = value
    return converted


def _build_custom_segformer_model(config: dict) -> CustomSegformerModel:
    model_cfg = config.get("model", {})
    spec = MODEL_SPECS["segformer"]
    pretrained_source, local_files_only = _resolve_pretrained_source(config)
    hf_config = _build_hf_config(spec, config, pretrained_source, local_files_only)

    decoder_dim = int(model_cfg.get("decoder_channels", 256))
    decoder_dropout = float(model_cfg.get("decoder_dropout", 0.1))
    model = CustomSegformerModel(
        hf_config,
        decoder_dim=decoder_dim,
        dropout_prob=decoder_dropout,
    )

    num_channels = int(model_cfg.get("num_channels", 4))
    channel_init = str(model_cfg.get("channel_init", "rgb_mean_inflate"))

    pretrained_model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_source,
        local_files_only=local_files_only,
    )
    converted = _convert_segformer_pretrained_backbone_keys(
        pretrained_model.state_dict()
    )
    first_conv_key = _map_segformer_pretrained_key_for_custom_model(spec.first_conv_key)
    if first_conv_key not in model.state_dict():
        raise RuntimeError(
            "Resolved first conv key is missing in custom SegFormer state dict: "
            f"{first_conv_key}"
        )
    adapted = adapt_pretrained_state_dict(
        converted,
        model.state_dict(),
        first_conv_key=first_conv_key,
        num_channels=num_channels,
        channel_init=channel_init,
        head_prefixes=CUSTOM_SEGFORMER_HEAD_PREFIXES,
    )

    missing, unexpected = model.load_state_dict(adapted, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected pretrained keys for segformer: {unexpected}")
    if first_conv_key in missing:
        raise RuntimeError(
            "Failed to initialize the first SegFormer patch embedding layer from pretrained weights."
        )
    return model


def _build_oilspill_hybrid_segformer_model(config: dict) -> SegformerOilSpillHybridModel:
    model_cfg = config.get("model", {})
    spec = MODEL_SPECS["segformer"]
    pretrained_source, local_files_only = _resolve_pretrained_source(config)
    hf_config = _build_hf_config(spec, config, pretrained_source, local_files_only)

    decoder_cfg = model_cfg.get("hybrid_decoder", {})
    model = SegformerOilSpillHybridModel(hf_config, decoder_cfg)

    num_channels = int(model_cfg.get("num_channels", 4))
    channel_init = str(model_cfg.get("channel_init", "rgb_mean_inflate"))

    pretrained_model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_source,
        local_files_only=local_files_only,
    )
    converted = _convert_segformer_pretrained_backbone_keys(
        pretrained_model.state_dict()
    )
    first_conv_key = _map_segformer_pretrained_key_for_custom_model(spec.first_conv_key)
    if first_conv_key not in model.state_dict():
        raise RuntimeError(
            "Resolved first conv key is missing in hybrid SegFormer state dict: "
            f"{first_conv_key}"
        )
    adapted = adapt_pretrained_state_dict(
        converted,
        model.state_dict(),
        first_conv_key=first_conv_key,
        num_channels=num_channels,
        channel_init=channel_init,
        head_prefixes=CUSTOM_SEGFORMER_HEAD_PREFIXES,
    )

    missing, unexpected = model.load_state_dict(adapted, strict=False)
    if unexpected:
        raise RuntimeError(
            f"Unexpected pretrained keys for oilspill_hybrid segformer: {unexpected}"
        )
    if first_conv_key in missing:
        raise RuntimeError(
            "Failed to initialize the first SegFormer patch embedding layer from pretrained weights."
        )
    return model


def build_hf_segmentation_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    architecture = str(model_cfg.get("architecture", "segformer")).lower()
    if architecture not in MODEL_SPECS:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported: {sorted(MODEL_SPECS)}"
        )

    if architecture == "segformer":
        requested_decoder = str(model_cfg.get("segformer_decoder", "hf_mlp")).lower()
        if requested_decoder in {"custom", "custom_cnn"}:
            logger.info(
                "SegFormer decoder selection: requested='%s', resolved='custom_cnn'.",
                requested_decoder,
            )
            return _build_custom_segformer_model(config)
        if requested_decoder == "oilspill_hybrid":
            logger.info(
                "SegFormer decoder selection: requested='%s', resolved='oilspill_hybrid'.",
                requested_decoder,
            )
            return _build_oilspill_hybrid_segformer_model(config)

        if requested_decoder != "hf_mlp":
            logger.warning(
                "Unknown model.segformer_decoder='%s'. Falling back to hf_mlp decoder path.",
                requested_decoder,
            )
        logger.info(
            "SegFormer decoder selection: requested='%s', resolved='hf_mlp'.",
            requested_decoder,
        )

    spec = MODEL_SPECS[architecture]
    pretrained_source, local_files_only = _resolve_pretrained_source(config)
    hf_config = _build_hf_config(spec, config, pretrained_source, local_files_only)
    model = spec.model_cls(hf_config)

    num_channels = int(model_cfg.get("num_channels", 4))
    channel_init = str(model_cfg.get("channel_init", "rgb_mean_inflate"))

    pretrained_model = spec.model_cls.from_pretrained(
        pretrained_source,
        local_files_only=local_files_only,
    )
    adapted = adapt_pretrained_state_dict(
        pretrained_model.state_dict(),
        model.state_dict(),
        first_conv_key=spec.first_conv_key,
        num_channels=num_channels,
        channel_init=channel_init,
        head_prefixes=spec.head_prefixes,
    )
    missing, unexpected = model.load_state_dict(adapted, strict=False)
    unexpected = [
        name for name in unexpected if not name.startswith(spec.head_prefixes)
    ]
    if unexpected:
        raise RuntimeError(
            f"Unexpected pretrained keys for {architecture}: {unexpected}"
        )
    if any(name == spec.first_conv_key for name in missing):
        raise RuntimeError(
            f"Failed to initialize first convolution for {architecture}: {spec.first_conv_key}"
        )

    return HFSegmentationModel(architecture=architecture, model=model)
