import argparse
import copy
import logging
import os
from pathlib import Path

import torch

from data.augmentation import SARSegmentationAugmentation
from data.dataset import OilSpillDataset
from data.sar_features import SARFeatureEncoder
from inference.evaluator import Evaluator
from models import build_model
from training.split import build_train_val_indices
from utils.config import load_config, resolve_sar_stats_path, resolve_train_split

logger = logging.getLogger(__name__)


def _resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def _load_checkpoint_state_dict(
    checkpoint: dict,
    *,
    use_ema_weights: bool,
) -> tuple[dict, str]:
    if use_ema_weights and "ema_state_dict" in checkpoint:
        return (
            checkpoint["ema_state_dict"]["shadow_state_dict"],
            "ema_state_dict.shadow_state_dict",
        )
    return checkpoint["model_state_dict"], "model_state_dict"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained SAR segmentation model"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint (.pth file)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_output",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("val", "test"),
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--use_ema_weights",
        action="store_true",
        help="Load checkpoint ema_state_dict.shadow_state_dict when present",
    )
    return parser.parse_args()


def _build_val_dataset(config: dict, transform, sar_encoder) -> OilSpillDataset:
    full_train = OilSpillDataset(
        root=config["dataset"]["root"],
        split="train",
        config=config,
        transform=transform,
        sar_encoder=sar_encoder,
    )
    _, val_indices = build_train_val_indices(
        full_train.metadata,
        train_split=resolve_train_split(config),
        seed=int(config.get("dataset", {}).get("split_seed", 42)),
    )
    val_dataset = copy.copy(full_train)
    val_dataset.enable_train_augmentations = False
    val_dataset.select_indices(val_indices)
    return val_dataset


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
    )

    config = load_config(args.config)
    config.setdefault("evaluation", {})["prediction_dir"] = args.output_dir

    device = _resolve_device()
    model = build_model(config)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict, state_dict_source = _load_checkpoint_state_dict(
        checkpoint,
        use_ema_weights=args.use_ema_weights,
    )

    # Remove '_orig_mod.' prefix added by torch.compile
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    logger.info(
        "Loaded checkpoint weights from %s onto %s.",
        state_dict_source,
        device,
    )

    stats_path = resolve_sar_stats_path(config)
    sar_encoder = SARFeatureEncoder(config, stats_path=str(stats_path))
    sar_encoder.load_stats()

    aug = SARSegmentationAugmentation(config)
    transform = aug.get_test_transform()
    if args.split == "test":
        dataset = OilSpillDataset(
            root=config["dataset"]["root"],
            split="test",
            config=config,
            transform=transform,
            sar_encoder=sar_encoder,
        )
    else:
        dataset = _build_val_dataset(config, transform, sar_encoder)

    if getattr(dataset, "feature_cache_enabled", False):
        dataset.precompute_sar_features()

    evaluator = Evaluator(config)
    evaluator.evaluate(model, dataset, config)


if __name__ == "__main__":
    main()
