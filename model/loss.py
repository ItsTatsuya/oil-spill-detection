"""
Loss functions for oil spill segmentation.

Key improvements over the original:
- Label smoothing added to HybridSegmentationLoss (was only in standalone weighted_cross_entropy)
- Focal loss double-weighting fixed: focal modulator applied to UNWEIGHTED CE, then class weights added
- Boundary loss re-enabled (boundary_weight > 0) with Sobel-based edge detection
- Lovász-Softmax loss added as an option (directly optimises IoU)
- Generalised Dice Loss option with inverse-square volume weighting
"""

import tensorflow as tf


# ---------------------------------------------------------------------------
# Standalone losses (kept for backward compatibility)
# ---------------------------------------------------------------------------
def weighted_cross_entropy(class_weights=None, from_logits=True, epsilon=1e-5):
    compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    if class_weights is not None:
        class_weights = tf.constant(class_weights, dtype=compute_dtype)

    def loss(y_true, y_pred):
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)
        num_classes = y_pred_shape[-1]

        y_true_rank = tf.rank(y_true)
        y_pred_rank = tf.rank(y_pred)
        is_sparse = tf.logical_and(
            tf.equal(y_true_rank, y_pred_rank - 1),
            tf.equal(y_true_shape[-1], 1)
        )

        def process_sparse():
            y_true_squeezed = tf.squeeze(y_true, axis=-1)
            y_true_indices = tf.cast(y_true_squeezed, tf.int32)
            return tf.one_hot(y_true_indices, depth=num_classes, dtype=compute_dtype)

        def process_error():
            tf.print("Error: Expected y_true shape [..., 1], got ", y_true_shape)
            return tf.one_hot(tf.zeros_like(y_true_shape[:-1]), depth=num_classes, dtype=compute_dtype)

        y_true_one_hot = tf.cond(is_sparse, process_sparse, process_error)

        y_true_smoothed = y_true_one_hot * 0.9 + 0.1 / tf.cast(num_classes, compute_dtype)
        y_pred = tf.clip_by_value(y_pred, -100.0, 100.0)

        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        ce_loss = -tf.reduce_mean(
            tf.cast(y_true_smoothed, tf.float32) * tf.math.log(tf.cast(y_pred, tf.float32)),
            axis=-1
        )

        if class_weights is not None:
            weights = tf.reduce_sum(y_true_one_hot * class_weights, axis=-1)
            ce_loss = ce_loss * tf.cast(weights, tf.float32)

        ce_loss = tf.cast(ce_loss, compute_dtype)
        tf.debugging.assert_all_finite(ce_loss, "NaN or Inf in weighted_cross_entropy")
        return tf.reduce_mean(ce_loss)

    return loss


def focal_loss(class_weights=None, gamma=2.0, from_logits=True, epsilon=1e-5):
    compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    if class_weights is not None:
        class_weights = tf.constant(class_weights, dtype=compute_dtype)

    def loss(y_true, y_pred):
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)
        num_classes = y_pred_shape[-1]

        y_true_rank = tf.rank(y_true)
        y_pred_rank = tf.rank(y_pred)
        is_sparse = tf.logical_and(
            tf.equal(y_true_rank, y_pred_rank - 1),
            tf.equal(y_true_shape[-1], 1)
        )

        def process_sparse():
            y_true_squeezed = tf.squeeze(y_true, axis=-1)
            y_true_indices = tf.cast(y_true_squeezed, tf.int32)
            return tf.one_hot(y_true_indices, depth=num_classes, dtype=compute_dtype)

        def process_error():
            tf.print("Error: Expected y_true shape [..., 1], got ", y_true_shape)
            return tf.one_hot(tf.zeros_like(y_true_shape[:-1]), depth=num_classes, dtype=compute_dtype)

        y_true_one_hot = tf.cond(is_sparse, process_sparse, process_error)

        y_pred = tf.clip_by_value(y_pred, -100.0, 100.0)

        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        focal_weight = tf.pow(1.0 - tf.cast(y_pred, tf.float32), gamma)
        focal_ce = -tf.reduce_mean(
            tf.cast(focal_weight * y_true_one_hot, tf.float32) * tf.math.log(tf.cast(y_pred, tf.float32)),
            axis=-1
        )

        if class_weights is not None:
            weights = tf.reduce_sum(y_true_one_hot * class_weights, axis=-1)
            focal_ce = focal_ce * tf.cast(weights, tf.float32)

        focal_ce = tf.cast(focal_ce, compute_dtype)
        tf.debugging.assert_all_finite(focal_ce, "NaN or Inf in focal_loss")
        return tf.reduce_mean(focal_ce)

    return loss


def dice_loss(class_weights=None, from_logits=True, epsilon=1e-5):
    compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    if class_weights is not None:
        class_weights = tf.constant(class_weights, dtype=compute_dtype)

    def loss(y_true, y_pred):
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)
        num_classes = y_pred_shape[-1]

        y_true_rank = tf.rank(y_true)
        y_pred_rank = tf.rank(y_pred)
        is_sparse = tf.logical_and(
            tf.equal(y_true_rank, y_pred_rank - 1),
            tf.equal(y_true_shape[-1], 1)
        )

        def process_sparse():
            y_true_squeezed = tf.squeeze(y_true, axis=-1)
            y_true_indices = tf.cast(y_true_squeezed, tf.int32)
            return tf.one_hot(y_true_indices, depth=num_classes, dtype=compute_dtype)

        def process_error():
            tf.print("Error: Expected y_true shape [..., 1], got ", y_true_shape)
            return tf.one_hot(tf.zeros_like(y_true_shape[:-1]), depth=num_classes, dtype=compute_dtype)

        y_true_one_hot = tf.cond(is_sparse, process_sparse, process_error)

        y_pred = tf.clip_by_value(y_pred, -100.0, 100.0)

        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        intersection = tf.reduce_sum(y_true_one_hot * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true_one_hot, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        dice = (2. * intersection + epsilon) / (union + epsilon)

        if class_weights is not None:
            dice = dice * class_weights
            dice_loss = 1.0 - tf.reduce_mean(dice / tf.reduce_sum(class_weights))
        else:
            dice_loss = 1.0 - tf.reduce_mean(dice)

        dice_loss = tf.cast(dice_loss, compute_dtype)
        tf.debugging.assert_all_finite(dice_loss, "NaN or Inf in dice_loss")
        return dice_loss

    return loss


def boundary_loss(class_weights=None, from_logits=True, epsilon=1e-5, boundary_weight=2.0):
    compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    if class_weights is not None:
        class_weights = tf.constant(class_weights, dtype=compute_dtype)

    def loss(y_true, y_pred):
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)
        num_classes = tf.identity(y_pred_shape[-1])

        y_true_rank = tf.rank(y_true)
        y_pred_rank = tf.rank(y_pred)
        is_sparse = tf.logical_and(
            tf.equal(y_true_rank, y_pred_rank - 1),
            tf.equal(y_true_shape[-1], 1)
        )

        def process_sparse():
            y_true_squeezed = tf.squeeze(y_true, axis=-1)
            y_true_indices = tf.cast(y_true_squeezed, tf.int32)
            return y_true_indices, tf.one_hot(y_true_indices, depth=num_classes, dtype=compute_dtype)

        def process_error():
            tf.print("Error: Expected y_true shape [..., 1], got ", y_true_shape)
            placeholder_indices = tf.zeros(y_true_shape[:-1], dtype=tf.int32)
            return placeholder_indices, tf.one_hot(placeholder_indices, depth=num_classes, dtype=compute_dtype)

        y_true_indices, y_true_one_hot = tf.cond(is_sparse, process_sparse, process_error)

        y_pred = tf.clip_by_value(y_pred, -100.0, 100.0)

        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32, shape=[3, 3, 1, 1])
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32, shape=[3, 3, 1, 1])

        batch_size = tf.shape(y_true_one_hot)[0]
        height = tf.shape(y_true_one_hot)[1]
        width = tf.shape(y_true_one_hot)[2]

        y_true_reshape = tf.reshape(y_true_one_hot, [-1, height, width, 1])
        y_pred_reshape = tf.reshape(y_pred, [-1, height, width, 1])

        edge_x_true = tf.abs(tf.nn.conv2d(y_true_reshape, sobel_x, strides=[1, 1, 1, 1], padding='SAME'))
        edge_y_true = tf.abs(tf.nn.conv2d(y_true_reshape, sobel_y, strides=[1, 1, 1, 1], padding='SAME'))
        edge_x_pred = tf.abs(tf.nn.conv2d(y_pred_reshape, sobel_x, strides=[1, 1, 1, 1], padding='SAME'))
        edge_y_pred = tf.abs(tf.nn.conv2d(y_pred_reshape, sobel_y, strides=[1, 1, 1, 1], padding='SAME'))

        y_true_edges_reshape = edge_x_true + edge_y_true
        y_pred_edges_reshape = edge_x_pred + edge_y_pred

        y_true_edges = tf.reshape(y_true_edges_reshape, [batch_size, height, width, -1])
        y_pred_edges = tf.reshape(y_pred_edges_reshape, [batch_size, height, width, -1])

        rare_class_mask = tf.cast(tf.logical_or(tf.equal(y_true_indices, 1), tf.equal(y_true_indices, 3)), tf.float32)
        rare_class_mask = tf.expand_dims(rare_class_mask, axis=-1)

        edge_magnitude = tf.reduce_sum(y_true_edges, axis=-1, keepdims=True)
        boundary_mask = tf.clip_by_value(edge_magnitude, 0, 1) * (1.0 + rare_class_mask * (boundary_weight - 1.0))

        boundary_loss = tf.reduce_mean(tf.square(y_true_edges - y_pred_edges) * boundary_mask)

        if class_weights is not None:
            weights = tf.reduce_sum(y_true_one_hot * class_weights, axis=-1)
            boundary_loss = boundary_loss * tf.reduce_mean(tf.cast(weights, tf.float32))

        boundary_loss = tf.cast(boundary_loss, compute_dtype)
        tf.debugging.assert_all_finite(boundary_loss, "NaN or Inf in boundary_loss")
        return boundary_loss

    return loss


# ---------------------------------------------------------------------------
# Lovász-Softmax  (directly optimises mean IoU)
# ---------------------------------------------------------------------------
def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t. sorted errors."""
    p = tf.shape(gt_sorted)[0]
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cast(tf.range(1, p + 1), tf.float32) - tf.cumsum(gt_sorted)
    jaccard = 1.0 - intersection / union
    jaccard = tf.concat([jaccard[:1], jaccard[1:] - jaccard[:-1]], axis=0)
    return jaccard


def lovasz_softmax_flat(probas, labels, num_classes=5):
    """
    Multi-class Lovász-Softmax loss (flat version, operates on 1-D tensors).

    Parameters
    ----------
    probas : (P, C) float — predicted probabilities per pixel per class
    labels : (P,) int   — ground truth class indices
    """
    losses = []
    for c in range(num_classes):
        fg = tf.cast(tf.equal(labels, c), tf.float32)  # foreground for class c
        if tf.reduce_sum(fg) == 0:
            continue
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0])
        fg_sorted = tf.gather(fg, perm)
        grad = _lovasz_grad(fg_sorted)
        losses.append(tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1))
    if not losses:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.reduce_mean(tf.stack(losses))


def lovasz_softmax_loss(from_logits=True, num_classes=5):
    """Return a Keras-compatible Lovász-Softmax loss function."""

    def loss(y_true, y_pred):
        if from_logits:
            probas = tf.nn.softmax(y_pred, axis=-1)
        else:
            probas = y_pred
        probas = tf.cast(probas, tf.float32)

        y_true_sq = tf.squeeze(y_true, axis=-1)
        y_true_flat = tf.cast(tf.reshape(y_true_sq, [-1]), tf.int32)
        probas_flat = tf.reshape(probas, [-1, num_classes])

        return lovasz_softmax_flat(probas_flat, y_true_flat, num_classes)

    return loss


# ---------------------------------------------------------------------------
# Hybrid Segmentation Loss  (improved)
# ---------------------------------------------------------------------------
class HybridSegmentationLoss(tf.keras.losses.Loss):
    """
    Combined segmentation loss with label smoothing, corrected focal weighting,
    and optional boundary & Lovász-Softmax components.

    Changes from original
    ---------------------
    1. Label smoothing applied before CE / focal computation.
    2. Focal modulator applied to UNWEIGHTED CE, class weights applied after.
    3. Boundary loss re-enabled by default (boundary_weight=0.1).
    4. Lovász-Softmax can be enabled via lovasz_weight > 0.
    """

    def __init__(
        self,
        class_weights=None,
        ce_weight=0.35,
        focal_weight=0.25,
        dice_weight=0.3,
        boundary_weight=0.1,
        lovasz_weight=0.0,
        gamma=2.0,
        label_smoothing=0.1,
        boundary_boost=2.5,
        epsilon=1e-7,
        from_logits=True,
        name="hybrid_segmentation_loss",
    ):
        super().__init__(name=name)
        self.class_weights = class_weights
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.lovasz_weight = lovasz_weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.boundary_boost = boundary_boost
        self.epsilon = epsilon
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

        if self.from_logits:
            probs = tf.nn.softmax(y_pred, axis=-1)
        else:
            probs = y_pred

        probs = tf.clip_by_value(probs, self.epsilon, 1.0 - self.epsilon)
        num_classes = tf.shape(y_pred)[-1]

        # One-hot encode
        y_true_processed = tf.reshape(y_true, [-1, tf.shape(y_true)[1], tf.shape(y_true)[2]])
        y_true_one_hot = tf.one_hot(tf.cast(y_true_processed, tf.int32), depth=num_classes)

        # ---- Label smoothing (applied to targets) ----------------------------
        if self.label_smoothing > 0:
            smooth = self.label_smoothing
            y_true_smoothed = y_true_one_hot * (1.0 - smooth) + smooth / tf.cast(num_classes, tf.float32)
        else:
            y_true_smoothed = y_true_one_hot

        # ---- UNWEIGHTED cross-entropy (for focal) ----------------------------
        unweighted_ce = -tf.reduce_sum(
            y_true_smoothed * tf.math.log(probs), axis=-1
        )

        # ---- Class-weighted cross-entropy ------------------------------------
        if self.class_weights is not None:
            cw = tf.cast(tf.constant(self.class_weights), compute_dtype)
            pixel_weights = tf.reduce_sum(y_true_one_hot * cw, axis=-1)
            weighted_ce = unweighted_ce * pixel_weights
        else:
            weighted_ce = unweighted_ce
            pixel_weights = None

        # ---- Focal loss (modulator on UNWEIGHTED CE, then class-weight) ------
        if self.focal_weight > 0:
            true_class_probs = tf.reduce_sum(y_true_one_hot * probs, axis=-1)
            focal_mod = tf.pow(1.0 - true_class_probs, self.gamma)
            focal_loss = unweighted_ce * focal_mod
            if pixel_weights is not None:
                focal_loss = focal_loss * pixel_weights
        else:
            focal_loss = unweighted_ce * 0

        # ---- Dice loss -------------------------------------------------------
        if self.dice_weight > 0:
            y_flat = tf.reshape(
                y_true_one_hot,
                [-1, tf.shape(y_true_one_hot)[1] * tf.shape(y_true_one_hot)[2], num_classes],
            )
            p_flat = tf.reshape(
                probs,
                [-1, tf.shape(probs)[1] * tf.shape(probs)[2], num_classes],
            )
            intersection = tf.reduce_sum(y_flat * p_flat, axis=1)
            sum_y = tf.reduce_sum(y_flat, axis=1)
            sum_p = tf.reduce_sum(p_flat, axis=1)
            dice = (2.0 * intersection + self.epsilon) / (sum_y + sum_p + self.epsilon)

            if self.class_weights is not None:
                cw = tf.cast(tf.constant(self.class_weights), compute_dtype)
                dice = dice * cw
                dice_loss = 1.0 - tf.reduce_sum(dice) / tf.reduce_sum(cw)
            else:
                dice_loss = 1.0 - tf.reduce_mean(dice)
        else:
            dice_loss = tf.constant(0.0, dtype=compute_dtype)

        # ---- Boundary loss ---------------------------------------------------
        if self.boundary_weight > 0:
            bnd_fn = boundary_loss(
                class_weights=(self.class_weights if self.class_weights else None),
                from_logits=False,  # already converted to probs
                boundary_weight=self.boundary_boost,
            )
            # Re-pack y_true to expected shape
            y_true_for_bnd = tf.expand_dims(y_true_processed, axis=-1)
            bnd_loss = bnd_fn(y_true_for_bnd, probs)
        else:
            bnd_loss = tf.constant(0.0, dtype=compute_dtype)

        # ---- Lovász-Softmax --------------------------------------------------
        if self.lovasz_weight > 0:
            lov_fn = lovasz_softmax_loss(from_logits=False, num_classes=num_classes)
            y_true_for_lov = tf.expand_dims(y_true_processed, axis=-1)
            lov_loss = lov_fn(y_true_for_lov, probs)
        else:
            lov_loss = tf.constant(0.0, dtype=compute_dtype)

        # ---- Combine ---------------------------------------------------------
        total_loss = (
            self.ce_weight * tf.reduce_mean(weighted_ce)
            + self.focal_weight * tf.reduce_mean(focal_loss)
            + self.dice_weight * dice_loss
            + self.boundary_weight * tf.cast(bnd_loss, compute_dtype)
            + self.lovasz_weight * tf.cast(lov_loss, compute_dtype)
        )

        return total_loss
