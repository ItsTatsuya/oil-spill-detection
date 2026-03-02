"""
Shared evaluation metrics for oil spill segmentation — PyTorch implementation.

- IoUMetric: thin wrapper around torchmetrics.JaccardIndex (drop-in for training loop)
- compute_iou: numpy-based confusion matrix; accepts torch tensors or numpy arrays
- All other functions are already numpy-based and unchanged.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics


CLASS_NAMES = ['Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land']


# ---------------------------------------------------------------------------
# torchmetrics-backed IoU metric (replaces Keras metric)
# ---------------------------------------------------------------------------
class IoUMetric:
    """Mean Intersection-over-Union backed by torchmetrics.JaccardIndex.

    Usage (mirrors the old Keras metric API):
        metric = IoUMetric(num_classes=5)
        metric.update_state(y_true, y_pred)   # y_pred: [B,C,H,W] logits/probs
        mean_iou = metric.result()             # scalar float tensor
        metric.reset_state()
    """

    def __init__(self, num_classes: int = 5, name: str = 'iou_metric', device=None):
        self.num_classes = num_classes
        self.name = name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._metric = torchmetrics.JaccardIndex(
            task='multiclass',
            num_classes=num_classes,
            average='none',
            ignore_index=None,
        ).to(self.device)
        self.class_names = CLASS_NAMES[:num_classes]

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Parameters
        ----------
        y_true : Tensor [B, H, W] long  —or—  [B, 1, H, W] uint8
        y_pred : Tensor [B, C, H, W] logits or probabilities
        """
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true.astype(np.int64))
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred.astype(np.float32))

        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)

        # Squeeze channel dim if [B,1,H,W]
        if y_true.dim() == 4:
            y_true = y_true.squeeze(1)
        y_true = y_true.long()

        # Resize pred to match labels if needed
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_pred = F.interpolate(y_pred.float(), size=y_true.shape[-2:],
                                   mode='bilinear', align_corners=False)

        # Convert logits → class indices
        y_pred_cls = y_pred.argmax(dim=1)  # [B, H, W]
        self._metric.update(y_pred_cls, y_true)

    def result(self):
        """Return mean IoU (scalar float tensor)."""
        class_ious = self._metric.compute()          # [num_classes]
        return class_ious.mean()

    def reset_state(self):
        self._metric.reset()

    def get_class_iou(self):
        """Return a dict {class_name: iou_value}."""
        class_ious = self._metric.compute().cpu().tolist()
        return {self.class_names[i]: class_ious[i] for i in range(self.num_classes)}


# ---------------------------------------------------------------------------
# Standalone metric functions (used in evaluate.py)
# ---------------------------------------------------------------------------
def compute_iou(y_true, y_pred, num_classes: int = 5):
    """
    Pixel-level IoU from torch tensors or numpy arrays.

    Parameters
    ----------
    y_true : Tensor/ndarray [B, H, W] int  —or—  [B, 1, H, W]
    y_pred : Tensor/ndarray [B, C, H, W] logits or probabilities

    Returns
    -------
    mean_iou        : float
    class_ious      : ndarray (num_classes,)
    confusion_matrix: ndarray (num_classes, num_classes)  rows=true, cols=pred
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    if y_true.ndim == 4:
        y_true = y_true.squeeze(1)          # [B, H, W]
    y_true = y_true.astype(np.int64)

    if y_pred.ndim == 4:
        y_pred = np.argmax(y_pred, axis=1)  # [B, H, W]
    y_pred = y_pred.astype(np.int64)

    # Build confusion matrix via bincount (fast, vectorised)
    nc2 = num_classes * num_classes
    cm = np.zeros(nc2, dtype=np.float64)
    for i in range(y_true.shape[0]):
        yt = np.clip(y_true[i].ravel(), 0, num_classes - 1)
        yp = np.clip(y_pred[i].ravel(), 0, num_classes - 1)
        cm += np.bincount(num_classes * yt + yp, minlength=nc2)
    cm = cm.reshape(num_classes, num_classes)

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp   # col_sum - tp
    fn = cm.sum(axis=1) - tp   # row_sum - tp
    denom = tp + fp + fn
    class_ious = np.where(denom > 0, tp / denom, 0.0)

    present = cm.sum(axis=1) > 0
    mean_iou = float(class_ious[present].mean()) if present.any() else 0.0

    return mean_iou, class_ious, cm


def compute_precision_recall_f1(confusion_matrix: np.ndarray):
    """Per-class Precision, Recall, F1 from a confusion matrix (rows=true, cols=pred)."""
    cm = confusion_matrix.astype(np.float64)
    tp = np.diag(cm)
    row_sums = cm.sum(axis=1)  # actual positives (TP + FN)
    col_sums = cm.sum(axis=0)  # predicted positives (TP + FP)

    precision = np.where(col_sums > 0, tp / col_sums, 0.0)
    recall = np.where(row_sums > 0, tp / row_sums, 0.0)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return precision, recall, f1


def compute_pixel_accuracy(confusion_matrix: np.ndarray) -> float:
    """Overall pixel accuracy."""
    return float(np.diag(confusion_matrix).sum() / (confusion_matrix.sum() + 1e-12))


def compute_frequency_weighted_iou(confusion_matrix: np.ndarray) -> float:
    """Frequency-weighted IoU (FWIoU)."""
    cm = confusion_matrix.astype(np.float64)
    tp = np.diag(cm)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    denom = row_sums + col_sums - tp
    iou = np.where(denom > 0, tp / denom, 0.0)
    freq = row_sums / (row_sums.sum() + 1e-12)
    return float(np.sum(freq * iou))


def compute_cohens_kappa(confusion_matrix: np.ndarray) -> float:
    """Cohen's Kappa coefficient."""
    cm = confusion_matrix.astype(np.float64)
    total = cm.sum()
    if total == 0:
        return 0.0
    p_o = np.diag(cm).sum() / total  # observed agreement
    row_sums = cm.sum(axis=1) / total
    col_sums = cm.sum(axis=0) / total
    p_e = np.sum(row_sums * col_sums)  # expected agreement
    if p_e >= 1.0:
        return 1.0
    return float((p_o - p_e) / (1.0 - p_e))


def compute_bootstrap_ci(
    per_image_labels,
    per_image_preds,
    num_classes: int = 5,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
):
    """
    Bootstrap confidence interval for mean IoU (vectorised with np.bincount).

    Parameters
    ----------
    per_image_labels : list of np.ndarray (H, W) int
    per_image_preds  : list of np.ndarray (H, W) int (argmax already applied)

    Returns
    -------
    ci_low, ci_high : float (0-1 scale)
    """
    rng = np.random.default_rng(seed)
    n = len(per_image_labels)
    nc2 = num_classes * num_classes
    bootstrap_mious = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        cm = np.zeros(nc2, dtype=np.float64)
        for i in idx:
            yt = per_image_labels[i].ravel().astype(np.int64)
            yp = per_image_preds[i].ravel().astype(np.int64)
            cm += np.bincount(num_classes * yt + yp, minlength=nc2)
        cm = cm.reshape(num_classes, num_classes)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / denom, 0.0)
        present = cm.sum(axis=1) > 0
        bootstrap_mious[b] = iou[present].mean() if present.any() else 0.0

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(bootstrap_mious, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_mious, 100 * (1 - alpha / 2)))
    return ci_low, ci_high
