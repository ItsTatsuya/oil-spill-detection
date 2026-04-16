import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import CLASS_NAMES
from utils.config import load_config, resolve_sar_stats_path

logger = logging.getLogger(__name__)


def _load_checkpoint_state_dict(
    checkpoint: dict,
    use_ema_weights: bool,
) -> tuple[dict, str]:
    if use_ema_weights and "ema_state_dict" in checkpoint:
        return checkpoint["ema_state_dict"]["shadow_state_dict"], "ema_state_dict.shadow_state_dict"
    return checkpoint["model_state_dict"], "model_state_dict"


def _evaluate_profile(model, dataset, config: dict, profile: str) -> dict:
    from inference.pipeline import InferencePipeline
    from training.metrics import SegmentationMetrics

    device = next(model.parameters()).device
    pipeline = InferencePipeline(config, profile=profile)
    tracker = SegmentationMetrics(
        num_classes=config.get("dataset", {}).get("num_classes", 5),
        class_names=CLASS_NAMES,
    )

    logger.info(
        "Running %s profile on %d samples (%s)",
        profile,
        len(dataset),
        pipeline.describe(),
    )

    for idx in tqdm(range(len(dataset)), desc=f"Evaluating [{profile}]"):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0)
        pred_map = pipeline.predict_segmentation_map(
            model,
            image,
            target_size=tuple(sample["mask"].shape),
        )
        pred_batch = torch.from_numpy(pred_map).long().unsqueeze(0)
        tracker.update(pred_batch, mask)

    metrics = tracker.compute()
    return {
        "mean_iou": float(metrics["mean_iou"]),
        "class_iou": {k: float(v) for k, v in metrics["class_iou"].items()},
        "per_class_precision": {
            k: float(v) for k, v in metrics.get("per_class_precision", {}).items()
        },
        "per_class_accuracy": {
            k: float(v) for k, v in metrics.get("per_class_accuracy", {}).items()
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on val/test split with fast and full profiles."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (.pth)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to write profile metrics JSON",
    )
    parser.add_argument(
        "--use_ema_weights",
        action="store_true",
        help="Use checkpoint ema_state_dict.shadow_state_dict when available",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("val", "test"),
        help="Dataset split to evaluate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )

    from data.augmentation import SARSegmentationAugmentation
    from data.dataset import OilSpillDataset
    from data.sar_features import SARFeatureEncoder
    from models import build_model
    from training.split import build_train_val_indices
    from utils.config import resolve_train_split

    config = load_config(args.config)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config)

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict, state_source = _load_checkpoint_state_dict(
        checkpoint, use_ema_weights=args.use_ema_weights
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

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
        dataset = copy.copy(full_train)
        dataset.enable_train_augmentations = False
        dataset.select_indices(val_indices)

    if getattr(dataset, "feature_cache_enabled", False):
        dataset.precompute_sar_features()

    fast_metrics = _evaluate_profile(model, dataset, config, profile="fast")
    full_metrics = _evaluate_profile(model, dataset, config, profile="full")

    result = {
        "config": args.config,
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "weights_source": state_source,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_miou": float(checkpoint.get("val_miou", 0.0)),
        "fast": fast_metrics,
        "full": full_metrics,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    logger.info(
        "Saved profile metrics to %s | fast mIoU=%.2f%% | full mIoU=%.2f%%",
        out_path,
        fast_metrics["mean_iou"] * 100.0,
        full_metrics["mean_iou"] * 100.0,
    )


if __name__ == "__main__":
    main()
