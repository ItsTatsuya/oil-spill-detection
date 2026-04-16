from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download local Hugging Face pretrained checkpoints used by SegFormer training"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Hugging Face model id (for example nvidia/segformer-b0-finetuned-ade-512-512)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path; uses model.pretrained_name when --model_id is omitted",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/pretrained",
        help="Directory to store downloaded model folder",
    )
    return parser.parse_args()


def load_model_id_from_config(config_path: Path) -> str:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = (cfg or {}).get("model", {})
    model_id = model_cfg.get("pretrained_name")
    if not model_id:
        raise ValueError("Could not find model.pretrained_name in config file")
    return str(model_id)


def _repo_folder_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def download_model(model_id: str, output_root: Path) -> Path:
    target_dir = output_root / _repo_folder_name(model_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] {model_id}")
    print(f"[target]   {target_dir}")

    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "*.json",
        ],
    )

    has_weights = (target_dir / "model.safetensors").exists() or (
        target_dir / "pytorch_model.bin"
    ).exists()
    if not (target_dir / "config.json").exists() or not has_weights:
        raise RuntimeError(
            f"Download incomplete for {model_id}. Expected config.json and weights in {target_dir}."
        )

    print(f"[ok]       {model_id} downloaded")
    return target_dir


def resolve_model_id(args: argparse.Namespace) -> str:
    if args.model_id:
        return str(args.model_id)
    if args.config:
        return load_model_id_from_config(Path(args.config))

    raise ValueError("Provide one of: --model_id or --config")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_id = resolve_model_id(args)
    print(f"Output directory: {output_root}")

    download_model(model_id, output_root)

    print("Done.")


if __name__ == "__main__":
    main()
