from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_SAR_CHANNELS = [
    "amplitude",
    "local_variance",
    "gradient_magnitude",
    "glcm_contrast",
]
_SUPPORTED_SAR_CHANNELS = set(_DEFAULT_SAR_CHANNELS)
_DEFAULT_SCHEMA_VERSION = "vv_singleband_rawfirst_v2"
_KNOWN_TOP_LEVEL_KEYS = {
    "dataset",
    "model",
    "sar_features",
    "training",
    "optimizer",
    "scheduler",
    "augmentation",
    "loss",
    "curriculum",
    "inference",
    "evaluation",
}

_OBSOLETE_KEYS = {
    "training.full_validation_every_n_epochs": "training.full_validation_every_n_epochs was removed with fast/both validation support.",
    "training.phase3_validate_every_n_epochs": "training.phase3_validate_every_n_epochs was removed. Use training.validate_every_n_epochs.",
    "model.input_channels": "model.input_channels was replaced by model.num_channels.",
    "model.ocr": "model.ocr was removed.",
    "model.logit_fusion": "model.logit_fusion was removed.",
    "model.backbone": "DeepLab backbone configuration was removed.",
    "model.backbone_variant": "DeepLab backbone configuration was removed.",
    "model.pretrained": "Use model.pretrained_name instead of model.pretrained.",
    "model.pretrained_dir": "model.pretrained_dir was removed. Hugging Face pretrained_name is the supported entry point.",
    "model.aspp_channels": "DeepLab ASPP configuration was removed.",
    "model.atrous_rates": "DeepLab ASPP configuration was removed.",
    "model.aspp_dropout": "DeepLab ASPP configuration was removed.",
    "model.fusion_channels": "DeepLab decoder configuration was removed.",
    "model.decoder_dropout": "DeepLab decoder configuration was removed.",
    "model.use_ship_head": "Ship auxiliary heads were removed.",
    "model.ship_head_hidden": "Ship auxiliary heads were removed.",
    "model.ship_head_dropout": "Ship auxiliary heads were removed.",
    "model.cbam_type": "CBAM configuration was removed.",
    "model.cbam_groups": "CBAM configuration was removed.",
    "model.cbam_dropout": "CBAM configuration was removed.",
    "model.cbam_physics_gate": "CBAM configuration was removed.",
    "model.use_esem": "ESEM supervision was removed.",
    "model.aspp_use_deformable": "DeepLab ASPP configuration was removed.",
    "optimizer.backbone_lr": "optimizer.backbone_lr was removed. Use optimizer.lr.",
    "optimizer.head_lr": "optimizer.head_lr was removed. Use optimizer.lr.",
    "augmentation.train.targeted_crops": "Targeted crop scheduling was removed.",
    "loss.contrastive_multiscale": "Contrastive multiscale loss was removed from the supported training recipe.",
    "loss.tversky": "Tversky loss was removed from the supported training recipe.",
    "loss.boundary_contrast_weight": "Boundary contrast loss was removed from the supported training recipe.",
    "loss.boundary_contrast_phase": "Boundary contrast loss was removed from the supported training recipe.",
    "loss.boundary_contrast_start_epoch": "Boundary contrast loss was removed from the supported training recipe.",
    "loss.esem_supervision_weight": "ESEM supervision was removed from the supported training recipe.",
    "loss.esem_weight_decay_epoch": "ESEM supervision was removed from the supported training recipe.",
}


def _sanitize_path_fragment(value: str) -> str:
    sanitized = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value
    )
    return sanitized.strip("._") or "default"


def resolve_model_num_channels(config: dict[str, Any]) -> int:
    model_cfg = config.get("model", {})
    if "num_channels" in model_cfg:
        return int(model_cfg["num_channels"])
    if "input_channels" in model_cfg:
        return int(model_cfg["input_channels"])
    return 4


def resolve_sar_channel_names(config: dict[str, Any]) -> list[str]:
    sar_cfg = config.get("sar_features", {})
    explicit_channels = sar_cfg.get("channels")
    if isinstance(explicit_channels, (list, tuple)) and explicit_channels:
        return [str(name) for name in explicit_channels]

    channels: list[str] = []
    channel_idx = 1
    while True:
        key = f"channel_{channel_idx}"
        if key not in sar_cfg:
            break
        channels.append(str(sar_cfg[key]))
        channel_idx += 1

    if channels:
        return channels

    num_channels = resolve_model_num_channels(config)
    if num_channels <= len(_DEFAULT_SAR_CHANNELS):
        return _DEFAULT_SAR_CHANNELS[:num_channels]
    extra = [
        f"channel_{idx}"
        for idx in range(len(_DEFAULT_SAR_CHANNELS) + 1, num_channels + 1)
    ]
    return _DEFAULT_SAR_CHANNELS + extra


def resolve_channel_schema_version(config: dict[str, Any]) -> str:
    sar_cfg = config.get("sar_features", {})
    explicit = sar_cfg.get("channel_schema_version")
    if explicit:
        return _sanitize_path_fragment(str(explicit))

    channels = "-".join(
        _sanitize_path_fragment(name) for name in resolve_sar_channel_names(config)
    )
    return _sanitize_path_fragment(f"{_DEFAULT_SCHEMA_VERSION}-{channels}")


def resolve_train_split(config: dict[str, Any]) -> float:
    dataset_cfg = config.get("dataset", {})
    return float(dataset_cfg.get("train_split", 0.9))


def _resolve_dataset_signature(config: dict[str, Any]) -> str:
    dataset_cfg = config.get("dataset", {})
    dataset_root = Path(dataset_cfg.get("root", "./dataset")).expanduser().resolve()
    dataset_name = _sanitize_path_fragment(dataset_root.name or "dataset")
    train_split = resolve_train_split(config)
    split_seed = int(dataset_cfg.get("split_seed", 42))
    num_channels = resolve_model_num_channels(config)
    schema_version = resolve_channel_schema_version(config)
    return (
        f"{dataset_name}__split-{train_split:g}"
        f"__seed-{split_seed}__ch-{num_channels}"
        f"__schema-{schema_version}"
    )


def resolve_dataset_signature(config: dict[str, Any]) -> str:
    return _resolve_dataset_signature(config)


def resolve_sar_stats_path(config: dict[str, Any]) -> Path:
    dataset_cfg = config.get("dataset", {})
    explicit = dataset_cfg.get("stats_path")
    signature = resolve_dataset_signature(config)
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.suffix:
            return (
                explicit_path.parent
                / f"{explicit_path.stem}__{signature}{explicit_path.suffix}"
            )
        return explicit_path / f"{signature}.json"
    return Path("data") / "sar_stats" / f"{signature}.json"


def resolve_ship_library_path(config: dict[str, Any]) -> Path:
    copy_paste_cfg = (
        config.get("augmentation", {}).get("train", {}).get("copy_paste", {})
    )
    explicit = copy_paste_cfg.get("ship_library_path")
    signature = resolve_dataset_signature(config)
    if explicit:
        return Path(explicit) / signature
    return Path("data") / "ship_crops" / signature


def resolve_sar_feature_cache_dir(config: dict[str, Any]) -> Path:
    cache_cfg = config.get("sar_features", {}).get("cache", {})
    explicit = cache_cfg.get("dir")
    signature = resolve_dataset_signature(config)
    if explicit:
        return Path(explicit) / signature
    return Path("data") / "sar_cache" / signature


def _iter_config_paths(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if not isinstance(value, dict):
        return items
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        items.append((path, child))
        items.extend(_iter_config_paths(child, path))
    return items


def _reject_obsolete_keys(config: dict[str, Any]) -> None:
    present_paths = {path for path, _ in _iter_config_paths(config)}
    violations = [
        f"{path}: {_OBSOLETE_KEYS[path]}"
        for path in sorted(_OBSOLETE_KEYS)
        if path in present_paths or (path in config and "." not in path)
    ]
    if violations:
        raise ValueError(
            "Config contains removed keys from the legacy DeepLab stack:\n- "
            + "\n- ".join(violations)
        )


def validate_config(config: dict[str, Any]) -> list[str]:
    warnings_list: list[str] = []
    unknown_top = sorted(set(config.keys()) - _KNOWN_TOP_LEVEL_KEYS)
    for key in unknown_top:
        warnings_list.append(f"Unknown top-level config section '{key}'.")
    if (
        str(config.get("model", {}).get("channel_init", "rgb_mean_inflate"))
        != "rgb_mean_inflate"
    ):
        raise ValueError("model.channel_init must be 'rgb_mean_inflate'.")
    channels = resolve_sar_channel_names(config)
    invalid_channels = [
        name for name in channels if name not in _SUPPORTED_SAR_CHANNELS
    ]
    if invalid_channels:
        raise ValueError(
            "sar_features channels must be chosen from "
            f"{sorted(_SUPPORTED_SAR_CHANNELS)}, got invalid entries: {invalid_channels}"
        )
    if len(set(channels)) != len(channels):
        raise ValueError(
            f"sar_features channels must be unique and ordered, got {channels}"
        )
    num_channels = resolve_model_num_channels(config)
    if channels and len(channels) != num_channels:
        raise ValueError(
            "model.num_channels must match the active SAR channel schema length: "
            f"num_channels={num_channels}, channels={channels}"
        )

    train_profile = (
        str(config.get("training", {}).get("validation_profile", "fast"))
        .strip()
        .lower()
    )
    eval_profile = (
        str(config.get("evaluation", {}).get("profile", "full")).strip().lower()
    )
    inference_cfg = config.get("inference", {})
    tta_enabled = bool(inference_cfg.get("tta", {}).get("enabled", False))
    multiscale_enabled = bool(inference_cfg.get("multiscale", {}).get("enabled", False))

    if train_profile == "fast" and (tta_enabled or multiscale_enabled):
        warnings_list.append(
            "training.validation_profile='fast' disables inference.tta.enabled and "
            "inference.multiscale.enabled during training validation."
        )

    return warnings_list


def load_config(config_path: str, validate: bool = True) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Run from the project root with a valid config path, for example: "
            "python train.py --config configs/segformer_sar.yaml"
        )

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    _reject_obsolete_keys(config)

    if validate:
        for message in validate_config(config):
            warnings.warn(message, stacklevel=2)

    resolve_train_split(config)
    return config
