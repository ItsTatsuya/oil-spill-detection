from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from constants import CLASS_NAMES
from inference.pipeline import InferencePipeline
from utils.visualization import PredictionVisualizer

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config: dict) -> None:
        self.config = config
        eval_cfg = config.get("evaluation", {})
        self.save_predictions = bool(eval_cfg.get("save_predictions", True))
        self.prediction_dir = Path(eval_cfg.get("prediction_dir", "./predictions"))
        self.visualize_n_samples = int(eval_cfg.get("visualize_n_samples", 20))
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = PredictionVisualizer(
            config, output_dir=str(self.prediction_dir)
        )

    def evaluate(
        self,
        model: nn.Module,
        dataset: Any,
        config: Optional[dict] = None,
    ) -> Dict[str, Any]:
        from training.metrics import SegmentationMetrics

        cfg = config or self.config
        eval_profile = str(cfg.get("evaluation", {}).get("profile", "full"))
        pipeline = InferencePipeline(cfg, profile=eval_profile)
        metrics_tracker = SegmentationMetrics(
            num_classes=cfg.get("model", {}).get("num_labels", 5),
            class_names=CLASS_NAMES,
        )

        device = next(model.parameters()).device
        num_samples = int(len(dataset))
        profiling = pipeline.describe_profile()
        using_cuda = device.type == "cuda"

        model.eval()
        logger.info(
            "Starting evaluation on %d samples (%s)", num_samples, pipeline.describe()
        )

        if using_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start_time = time.perf_counter()

        for idx in tqdm(range(num_samples), desc="Evaluating"):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"]
            pred_map = pipeline.predict_segmentation_map(
                model,
                image,
                target_size=tuple(mask.shape),
            )
            metrics_tracker.update(
                torch.from_numpy(pred_map.astype(np.int64)).unsqueeze(0),
                mask.unsqueeze(0),
            )
            if self.save_predictions:
                self._save_prediction_artifacts(idx, sample["image"], mask, pred_map)

        if using_cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        peak_gpu_memory_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if using_cuda
            else None
        )

        metrics = metrics_tracker.compute()
        profiling.update(
            {
                "num_samples": num_samples,
                "estimated_total_forward_passes": (
                    profiling["forward_passes_per_image"] * num_samples
                ),
                "runtime_seconds": float(elapsed),
                "seconds_per_image": float(elapsed / num_samples)
                if num_samples
                else None,
                "images_per_second": float(num_samples / elapsed) if elapsed > 0 else None,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
            }
        )
        metrics["profiling"] = profiling
        self._save_results(metrics, self.prediction_dir / "evaluation_results.json")
        self._save_confusion_matrix(metrics)
        self._log_summary(metrics)
        return metrics

    def _save_prediction_artifacts(
        self,
        idx: int,
        image: torch.Tensor,
        mask: torch.Tensor,
        pred_map: np.ndarray,
    ) -> None:
        pred_path = self.prediction_dir / f"pred_{idx:04d}.npy"
        np.save(str(pred_path), pred_map.astype(np.uint8))
        if idx >= self.visualize_n_samples:
            return
        mask_np = mask.cpu().numpy()
        per_class_iou = {}
        for c, cname in enumerate(CLASS_NAMES):
            tp = int(np.sum((pred_map == c) & (mask_np == c)))
            fp = int(np.sum((pred_map == c) & (mask_np != c)))
            fn = int(np.sum((pred_map != c) & (mask_np == c)))
            union = tp + fp + fn
            per_class_iou[cname] = tp / (union + 1e-6) if union > 0 else float("nan")
        valid_ious = [v for v in per_class_iou.values() if not np.isnan(v)]
        sample_miou = float(np.mean(valid_ious)) if valid_ious else 0.0
        vis_dir = self.prediction_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / f"vis_{idx:04d}.png"
        self.visualizer.visualize_prediction(
            image=image,
            ground_truth=mask,
            prediction=pred_map,
            save_path=str(vis_path),
            sample_iou=sample_miou,
            per_class_iou=per_class_iou,
            title=f"Sample #{idx:04d}",
        )

    def _save_results(self, metrics: Dict[str, Any], path: Path) -> None:
        payload = {
            "mean_iou": float(metrics["mean_iou"]),
            "class_iou": {k: float(v) for k, v in metrics["class_iou"].items()},
            "per_class_accuracy": {
                k: float(v) for k, v in metrics.get("per_class_accuracy", {}).items()
            },
            "per_class_precision": {
                k: float(v) for k, v in metrics.get("per_class_precision", {}).items()
            },
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "profiling": metrics.get("profiling", {}),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_confusion_matrix(self, metrics: Dict[str, Any]) -> None:
        vis_dir = self.prediction_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer.plot_confusion_matrix(
            metrics["confusion_matrix"],
            str(vis_dir / "confusion_matrix.png"),
            normalize=True,
        )

    def _log_summary(self, metrics: Dict[str, Any]) -> None:
        logger.info(
            "Evaluation complete. Mean IoU: %.2f%%", metrics["mean_iou"] * 100.0
        )
        profiling = metrics.get("profiling", {})
        runtime_seconds = profiling.get("runtime_seconds")
        images_per_second = profiling.get("images_per_second")
        peak_gpu_memory_mb = profiling.get("peak_gpu_memory_mb")
        if runtime_seconds is not None and images_per_second is not None:
            logger.info(
                "Runtime: %.2fs total | %.3f img/s | %d forward passes/image",
                runtime_seconds,
                images_per_second,
                int(profiling.get("forward_passes_per_image", 1)),
            )
        if peak_gpu_memory_mb is not None:
            logger.info("Peak GPU memory: %.1f MB", peak_gpu_memory_mb)
        for name in CLASS_NAMES:
            logger.info("%s IoU: %.2f%%", name, metrics["class_iou"][name] * 100.0)
