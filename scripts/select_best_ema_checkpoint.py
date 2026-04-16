import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import CLASS_NAMES
from utils.config import load_config, resolve_sar_stats_path, resolve_train_split

logger = logging.getLogger(__name__)


def _evaluate_profile(model, dataset, config: dict, profile: str) -> dict:
    from inference.pipeline import InferencePipeline
    from training.metrics import SegmentationMetrics

    device = next(model.parameters()).device
    pipeline = InferencePipeline(config, profile=profile)
    tracker = SegmentationMetrics(
        num_classes=config.get("dataset", {}).get("num_classes", 5),
        class_names=CLASS_NAMES,
    )

    for idx in range(len(dataset)):
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


def _build_val_dataset(config: dict):
    from data.augmentation import SARSegmentationAugmentation
    from data.dataset import OilSpillDataset
    from data.sar_features import SARFeatureEncoder
    from training.split import build_train_val_indices

    stats_path = resolve_sar_stats_path(config)
    sar_encoder = SARFeatureEncoder(config, stats_path=str(stats_path))
    sar_encoder.load_stats()

    aug = SARSegmentationAugmentation(config)
    transform = aug.get_test_transform()

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

    if getattr(val_dataset, "feature_cache_enabled", False):
        val_dataset.precompute_sar_features()
    return val_dataset


def _build_test_dataset(config: dict):
    from data.augmentation import SARSegmentationAugmentation
    from data.dataset import OilSpillDataset
    from data.sar_features import SARFeatureEncoder

    stats_path = resolve_sar_stats_path(config)
    sar_encoder = SARFeatureEncoder(config, stats_path=str(stats_path))
    sar_encoder.load_stats()
    transform = SARSegmentationAugmentation(config).get_test_transform()
    test_dataset = OilSpillDataset(
        root=config["dataset"]["root"],
        split="test",
        config=config,
        transform=transform,
        sar_encoder=sar_encoder,
    )
    if getattr(test_dataset, "feature_cache_enabled", False):
        test_dataset.precompute_sar_features()
    return test_dataset


def _candidate_paths(checkpoint_dir: Path, top_k: int) -> list[Path]:
    topk_paths = sorted(checkpoint_dir.glob("topk_ema_epoch_*.pth"))
    if topk_paths:
        scored = []
        for path in topk_paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            scored.append((float(checkpoint.get("val_miou", 0.0)), path))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in scored[:top_k]]

    for name in ("best_ema.pth", "best.pth", "latest.pth"):
        path = checkpoint_dir / name
        if path.exists():
            return [path]
    raise FileNotFoundError(f"No candidate checkpoints found in {checkpoint_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select the best EMA checkpoint by full-profile validation evaluation."
    )
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Checkpoint directory containing topk_ema_epoch_*.pth or best_ema.pth",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Path to write selection results JSON",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Maximum number of EMA checkpoint candidates to evaluate on val",
    )
    parser.add_argument(
        "--run_final_test_eval",
        action="store_true",
        help="Also evaluate the selected checkpoint on the test split. Disabled by default to avoid data leakage during selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

    from models import build_model

    config = load_config(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    candidates = _candidate_paths(checkpoint_dir, top_k=max(int(args.top_k), 1))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device).eval()
    val_dataset = _build_val_dataset(config)

    candidate_results = []
    for path in candidates:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        val_full = _evaluate_profile(model, val_dataset, config, profile="full")
        candidate_results.append(
            {
                "checkpoint": str(path),
                "epoch": int(checkpoint.get("epoch", 0)),
                "fast_val_miou": float(checkpoint.get("val_miou", 0.0)),
                "val_full": val_full,
            }
        )
        logger.info(
            "Candidate %s | fast val %.2f%% | full val %.2f%%",
            path.name,
            float(checkpoint.get("val_miou", 0.0)) * 100.0,
            val_full["mean_iou"] * 100.0,
        )

    best = max(candidate_results, key=lambda item: item["val_full"]["mean_iou"])
    best_path = Path(best["checkpoint"])
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    selected_val_fast = _evaluate_profile(model, val_dataset, config, profile="fast")
    selected_val_full = _evaluate_profile(model, val_dataset, config, profile="full")
    result = {
        "config": args.config,
        "checkpoint_dir": str(checkpoint_dir),
        "candidates": candidate_results,
        "selected_checkpoint": str(best_path),
        "selected_epoch": int(best_checkpoint.get("epoch", 0)),
        "selected_fast_val_miou": float(best_checkpoint.get("val_miou", 0.0)),
        "selected_val_fast": selected_val_fast,
        "selected_val_full": selected_val_full,
    }

    if args.run_final_test_eval:
        test_dataset = _build_test_dataset(config)
        selected_test_fast = _evaluate_profile(model, test_dataset, config, profile="fast")
        selected_test_full = _evaluate_profile(model, test_dataset, config, profile="full")
        result["selected_test_fast"] = selected_test_fast
        result["selected_test_full"] = selected_test_full

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.run_final_test_eval:
        logger.info(
            "Selected %s | full val %.2f%% | full test %.2f%%",
            best_path.name,
            selected_val_full["mean_iou"] * 100.0,
            result["selected_test_full"]["mean_iou"] * 100.0,
        )
    else:
        logger.info(
            "Selected %s | full val %.2f%% | test evaluation skipped",
            best_path.name,
            selected_val_full["mean_iou"] * 100.0,
        )


if __name__ == "__main__":
    main()
