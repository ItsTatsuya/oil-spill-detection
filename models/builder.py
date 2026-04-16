from __future__ import annotations

from models.hf_segmentation import MODEL_SPECS, build_hf_segmentation_model


def get_model_architecture(config: dict) -> str:
    architecture = str(config.get("model", {}).get("architecture", "segformer")).lower()
    if architecture not in MODEL_SPECS:
        raise ValueError(
            f"Unsupported model architecture '{architecture}'. Supported: {sorted(MODEL_SPECS)}"
        )
    return architecture


def build_model(config: dict):
    return build_hf_segmentation_model(config)


def get_model_run_name(config: dict) -> str:
    model_cfg = config.get("model", {})
    architecture = get_model_architecture(config)
    pretrained_name = str(model_cfg.get("pretrained_name", architecture)).split("/")[-1]
    return f"{architecture}_{pretrained_name}"
