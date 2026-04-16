import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

from constants import CLASS_NAMES, NUM_CLASSES


class SegmentationMetrics:

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES[:num_classes]

        self._confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if predictions.dim() == 4:
            pred_classes = predictions.argmax(dim=1)  
        elif predictions.dim() == 3:
            pred_classes = predictions
        else:
            raise ValueError(f"Unexpected predictions shape: {predictions.shape}")

        if pred_classes.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {pred_classes.shape} vs targets {targets.shape}"
            )

        pred_flat = pred_classes.reshape(-1).to(dtype=torch.int64)
        tgt_flat = targets.reshape(-1).to(dtype=torch.int64)

        valid = (tgt_flat >= 0) & (tgt_flat < self.num_classes)
        if not torch.any(valid):
            return

        pred_valid = pred_flat[valid]
        tgt_valid = tgt_flat[valid]
        indices = tgt_valid * self.num_classes + pred_valid
        cm_flat = torch.bincount(indices, minlength=self.num_classes**2)

        self._confusion_matrix += (
            cm_flat.view(self.num_classes, self.num_classes).cpu().numpy()
        )

    def compute_from_confusion_matrix(self, confusion_matrix: np.ndarray) -> Dict:
        if confusion_matrix.shape != (self.num_classes, self.num_classes):
            raise ValueError(
                "Expected confusion matrix with shape "
                f"{(self.num_classes, self.num_classes)}, got {confusion_matrix.shape}"
            )

        cm = confusion_matrix.astype(np.float64, copy=False)

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp  
        fn = cm.sum(axis=1) - tp  

        iou_per_class = tp / (tp + fp + fn + 1e-6)
        row_sums = cm.sum(axis=1)  
        present = row_sums > 0
        mean_iou = float(iou_per_class[present].mean()) if present.any() else 0.0

        precision_per_class = tp / (tp + fp + 1e-6)
        acc_per_class = tp / (row_sums + 1e-6)

        class_iou = {
            name: float(iou_per_class[c]) for c, name in enumerate(self.class_names)
        }
        per_class_precision = {
            name: float(precision_per_class[c]) for c, name in enumerate(self.class_names)
        }
        per_class_accuracy = {
            name: float(acc_per_class[c]) for c, name in enumerate(self.class_names)
        }

        return {
            "mean_iou": mean_iou,
            "class_iou": class_iou,
            "confusion_matrix": confusion_matrix.copy(),
            "per_class_precision": per_class_precision,
            "per_class_accuracy": per_class_accuracy,
        }

    def compute(self) -> Dict:
        return self.compute_from_confusion_matrix(self._confusion_matrix)

    def reset(self) -> None:
        self._confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def print_table(
        self,
        metrics: Optional[Dict] = None,
        baseline_iou: Optional[Dict[str, float]] = None,
    ) -> None:
        if metrics is None:
            metrics = self.compute()

        baseline = baseline_iou or {}

        print("\n" + "=" * 65)
        print(f"{'Class':<18} {'IoU (%)':>10} {'Baseline (%)':>14} {'Delta':>8}")
        print("-" * 65)

        for name in self.class_names:
            iou = metrics["class_iou"][name] * 100.0
            base = baseline.get(name, None)
            if base is not None:
                delta = iou - base
                delta_str = f"{delta:+.2f}"
                print(f"{name:<18} {iou:>10.2f} {base:>14.2f} {delta_str:>8}")
            else:
                print(f"{name:<18} {iou:>10.2f} {'N/A':>14} {'N/A':>8}")

        miou = metrics["mean_iou"] * 100.0
        base_mean = baseline.get("mean", None)
        print("-" * 65)
        if base_mean is not None:
            delta = miou - base_mean
            print(f"{'Mean IoU':<18} {miou:>10.2f} {base_mean:>14.2f} {delta:>+8.2f}")
        else:
            print(f"{'Mean IoU':<18} {miou:>10.2f}")
        print("=" * 65 + "\n")
