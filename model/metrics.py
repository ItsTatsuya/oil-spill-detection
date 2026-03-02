"""
Shared evaluation metrics for oil spill segmentation.

Consolidates IoUMetric (Keras metric), compute_iou, precision/recall/F1,
Frequency-Weighted IoU, Boundary IoU, Pixel Accuracy, and Cohen's Kappa.
"""

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Keras metric (used in model.compile)
# ---------------------------------------------------------------------------
class IoUMetric(tf.keras.metrics.Metric):
    """Mean Intersection-over-Union as a Keras metric (for model.compile)."""

    def __init__(self, num_classes: int = 5, name: str = 'iou_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.mean_iou = tf.keras.metrics.MeanIoU(num_classes=num_classes)
        self.class_names = ['Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land']

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)
        compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

        def resize_fn():
            resized = tf.image.resize(
                y_pred, [y_true_shape[1], y_true_shape[2]], method='bilinear'
            )
            return tf.cast(resized, compute_dtype)

        def identity_fn():
            return tf.cast(y_pred, compute_dtype)

        y_pred_matched = tf.cond(
            tf.logical_or(
                tf.not_equal(y_true_shape[1], y_pred_shape[1]),
                tf.not_equal(y_true_shape[2], y_pred_shape[2]),
            ),
            resize_fn,
            identity_fn,
        )

        y_true_int = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        y_pred_int = tf.cast(tf.argmax(y_pred_matched, axis=-1), tf.int32)
        self.mean_iou.update_state(y_true_int, y_pred_int)

    def result(self):
        return self.mean_iou.result()

    def reset_state(self):
        self.mean_iou.reset_state()

    def get_class_iou(self):
        """Return a dict {class_name: iou_value}."""
        cm = self.mean_iou.total_cm
        row_sum = tf.cast(tf.reduce_sum(cm, axis=0), tf.float32)
        col_sum = tf.cast(tf.reduce_sum(cm, axis=1), tf.float32)
        tp = tf.cast(tf.linalg.tensor_diag_part(cm), tf.float32)
        iou = tf.math.divide_no_nan(tp, row_sum + col_sum - tp)
        return {self.class_names[i]: float(iou[i].numpy()) for i in range(self.num_classes)}


# ---------------------------------------------------------------------------
# Standalone metric functions (used in evaluate.py)
# ---------------------------------------------------------------------------
def compute_iou(y_true, y_pred, num_classes: int = 5):
    """
    Pixel-level IoU per class from TF tensors.

    Returns
    -------
    mean_iou : float
    class_ious : ndarray (num_classes,)
    confusion_matrix : ndarray (num_classes, num_classes)  rows=true, cols=pred
    """
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

    confusion_matrix = tf.zeros((num_classes, num_classes), dtype=tf.float32)
    for i in range(tf.shape(y_true)[0]):
        cm_i = tf.math.confusion_matrix(
            tf.reshape(y_true[i], [-1]),
            tf.reshape(y_pred[i], [-1]),
            num_classes=num_classes,
            dtype=tf.float32,
        )
        confusion_matrix += cm_i

    row_sum = tf.reduce_sum(confusion_matrix, axis=0)
    col_sum = tf.reduce_sum(confusion_matrix, axis=1)
    tp = tf.linalg.tensor_diag_part(confusion_matrix)
    class_ious = tf.math.divide_no_nan(tp, row_sum + col_sum - tp)

    present = tf.greater(col_sum, 0)
    mean_iou = tf.reduce_mean(tf.boolean_mask(class_ious, present))

    return float(mean_iou.numpy()), class_ious.numpy(), confusion_matrix.numpy()


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
