from __future__ import annotations

from models.hf_segmentation import MODEL_SPECS, build_hf_segmentation_model
from models.torchvision_seg import (
    build_torchvision_segmentation_model,
    is_torchvision_architecture,
)


def get_model_architecture(config: dict) -> str:
    architecture = str(config.get("model", {}).get("architecture", "segformer")).lower()
    if is_torchvision_architecture(architecture):
        return architecture
    if architecture in MODEL_SPECS:
        return architecture
    supported = sorted(
        set(MODEL_SPECS)
        | {
            "deeplabv3_resnet50",
            "deeplabv3_resnet101",
            "deeplabv3_mobilenet_v3_large",
        }
    )
    raise ValueError(
        f"Unsupported model architecture '{architecture}'. Supported: {supported}"
    )


def build_model(config: dict):
    architecture = str(config.get("model", {}).get("architecture", "segformer")).lower()
    if is_torchvision_architecture(architecture):
        return build_torchvision_segmentation_model(config)
    return build_hf_segmentation_model(config)


def get_model_run_name(config: dict) -> str:
    model_cfg = config.get("model", {})
    architecture = get_model_architecture(config)
    if is_torchvision_architecture(architecture):
        tag = "pretrained" if model_cfg.get("pretrained", True) else "scratch"
        return f"{architecture}_{tag}"
    pretrained_name = str(model_cfg.get("pretrained_name", architecture)).split("/")[-1]
    return f"{architecture}_{pretrained_name}"
