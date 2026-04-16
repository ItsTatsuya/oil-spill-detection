import argparse
import copy
import logging
import os

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import random
import warnings
from pathlib import Path

import numpy as np
import torch

torch.set_float32_matmul_precision("high")
from constants import CLASS_NAMES
from data.augmentation import SARSegmentationAugmentation
from data.copy_paste import CopyPasteAugmentation
from data.dataset import OilSpillDataset
from data.sar_features import SARFeatureEncoder
from inference.evaluator import Evaluator
from models import build_model
from training.split import build_train_val_indices
from training.trainer import Trainer
from utils.config import load_config, resolve_sar_stats_path, resolve_train_split
from utils.distributed import cleanup_distributed, setup_distributed
from utils.logger import Logger

warnings.filterwarnings("ignore", message="Argument.*are not valid for transform")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAR semantic segmentation model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/segformer_sar.yaml",
        help="Path to configuration YAML file",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    resume_group.add_argument(
        "--no-resume",
        "-no-resume",
        "--no-auto-resume",
        dest="no_resume",
        action="store_true",
        help="Start from scratch and skip loading the latest checkpoint.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run a short debug training loop"
    )
    parser.add_argument(
        "--run-final-eval",
        action="store_true",
        help="Run an explicit final evaluation after training. Disabled by default to avoid test leakage.",
    )
    parser.add_argument(
        "--final-eval-split",
        type=str,
        default="test",
        choices=("val", "test"),
        help="Dataset split for the explicit final evaluation step.",
    )
    return parser.parse_args()


def auto_detect_gpu_config(config: dict) -> dict:
    train_cfg = config.setdefault("training", {})
    world_size = int(os.environ.get("WORLD_SIZE", max(torch.cuda.device_count(), 1)))
    batch_per_gpu = int(train_cfg.get("batch_size_per_gpu", 1))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    train_cfg["num_gpus"] = world_size
    train_cfg["effective_batch_size"] = batch_per_gpu * max(world_size, 1) * grad_accum
    return config


def compute_dataset_pixel_counts(metadata: list[dict]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for sample_meta in metadata:
        sample_counts = sample_meta.get("pixel_counts", {})
        if not isinstance(sample_counts, dict):
            continue
        for name in CLASS_NAMES:
            counts[name] += int(sample_counts.get(name, 0))
    return counts


def main():
    args = parse_args()
    config = auto_detect_gpu_config(load_config(args.config))

    seed = int(config.get("dataset", {}).get("split_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.debug:
        config["training"]["num_epochs"] = 4
        config["training"]["validate_every_n_epochs"] = 2
        config["training"]["save_every_n_epochs"] = 2
        config["training"]["batch_size_per_gpu"] = 2
        config["training"]["num_workers"] = 0
        config["training"]["val_num_workers"] = 0

    aug = SARSegmentationAugmentation(config)
    train_transform = aug.get_train_transform()
    test_transform = aug.get_test_transform()
    exp_logger = Logger(config=config)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        exp_logger.info(
            "Disabled reduced-precision BF16 GEMM reductions for stability "
            "(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False)."
        )

    stats_path = resolve_sar_stats_path(config)
    sar_encoder = SARFeatureEncoder(config, stats_path=str(stats_path))
    dataset_root = config["dataset"]["root"]

    full_train_dataset = OilSpillDataset(
        root=dataset_root,
        split="train",
        config=config,
        transform=train_transform,
        sar_encoder=sar_encoder,
    )
    val_dataset = copy.copy(full_train_dataset)
    val_dataset.transform = test_transform
    val_dataset.enable_train_augmentations = False
    val_dataset.copy_paste = None

    train_indices, val_indices = build_train_val_indices(
        full_train_dataset.metadata,
        train_split=resolve_train_split(config),
        seed=seed,
    )
    train_dataset = full_train_dataset
    train_dataset.select_indices(train_indices)
    val_dataset.select_indices(val_indices)

    if args.debug:
        train_dataset.select_indices(list(range(min(32, len(train_dataset)))))
        val_dataset.select_indices(list(range(min(8, len(val_dataset)))))

    split_pixel_counts = compute_dataset_pixel_counts(train_dataset.metadata)
    config.setdefault("dataset", {})["pixel_counts"] = split_pixel_counts
    exp_logger.info(
        f"Resolved dataset.pixel_counts from active train split ({len(train_dataset)} images): "
        f"{split_pixel_counts}"
    )

    if not Path(stats_path).exists():
        exp_logger.info("Fitting SAR feature encoder on training images...")
        image_paths = getattr(train_dataset, "image_paths", [])
        if not image_paths:
            raise RuntimeError("No training images available to fit SAR statistics.")
        sar_encoder.fit(image_paths)
    else:
        sar_encoder.load_stats()

    train_dataset.sar_encoder = sar_encoder
    val_dataset.sar_encoder = sar_encoder

    if train_dataset.feature_cache_enabled:
        for dataset in (train_dataset, val_dataset):
            dataset.precompute_sar_features()

    copy_paste_cfg = (
        config.get("augmentation", {}).get("train", {}).get("copy_paste", {})
    )
    if copy_paste_cfg.get("enabled", True) and not args.debug:
        exp_logger.info("Building ship copy-paste library...")
        copy_paste = CopyPasteAugmentation(config)
        copy_paste.build_ship_library(train_dataset)
        train_dataset.copy_paste = copy_paste
        val_dataset.copy_paste = None

    exp_logger.info("Building segmentation model...")
    model = build_model(config)

    # --- ADA LOVELACE OPTIMIZATION ---
    exp_logger.info("Compiling model for Ada Lovelace (Triton)...")
    from torch._inductor import config as inductor_config

    inductor_config.max_autotune = False
    inductor_config.max_autotune_gemm = False
    # We compile the model BEFORE wrapping it in DDP
    model = torch.compile(model)
    # ---------------------------------

    model, is_distributed, local_rank = setup_distributed(model)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        is_distributed=is_distributed,
        local_rank=local_rank,
        exp_logger=exp_logger,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif args.no_resume:
        exp_logger.info("Resume disabled; starting from scratch.")
    else:
        trainer.load_checkpoint()

    trainer.train()

    if local_rank == 0 and args.run_final_eval:
        final_eval_config = copy.deepcopy(config)
        final_eval_config.setdefault("evaluation", {})["profile"] = "full"
        checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        best_ckpt = checkpoint_dir / "best_ema.pth"
        if not best_ckpt.exists():
            best_ckpt = checkpoint_dir / "best.pth"
        if best_ckpt.exists():
            ckpt = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
            unwrapped = model.module if hasattr(model, "module") else model
            unwrapped.load_state_dict(ckpt["model_state_dict"])
        if args.final_eval_split == "test":
            eval_dataset = OilSpillDataset(
                root=dataset_root,
                split="test",
                config=config,
                transform=test_transform,
                sar_encoder=sar_encoder,
            )
            if args.debug:
                eval_dataset.select_indices(list(range(min(8, len(eval_dataset)))))
        else:
            eval_dataset = val_dataset
        if getattr(eval_dataset, "feature_cache_enabled", False):
            eval_dataset.precompute_sar_features()
        evaluator = Evaluator(final_eval_config)
        evaluator.evaluate(
            model.module if hasattr(model, "module") else model,
            eval_dataset,
            final_eval_config,
        )

    cleanup_distributed()
    exp_logger.finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()
