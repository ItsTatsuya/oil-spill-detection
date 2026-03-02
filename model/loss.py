import tensorflow as tf

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

class HybridSegmentationLoss(tf.keras.losses.Loss):
    def __init__(self,
                 class_weights=None,
                 ce_weight=0.4,
                 focal_weight=0.3,
                 dice_weight=0.3,
                 boundary_weight=0.0,
                 gamma=3.0,
                 boundary_boost=2.5,
                 epsilon=1e-7,
                 from_logits=True,
                 name="hybrid_segmentation_loss"):

        super().__init__(name=name)
        self.class_weights = class_weights
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.gamma = gamma
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

        y_true_processed = tf.reshape(y_true, [-1, tf.shape(y_true)[1], tf.shape(y_true)[2]])
        y_true_one_hot = tf.one_hot(tf.cast(y_true_processed, tf.int32), depth=num_classes)

        ce_loss = -tf.reduce_sum(y_true_one_hot * tf.math.log(probs), axis=-1)

        if self.class_weights is not None:
            class_weights_tensor = tf.cast(tf.constant(self.class_weights), compute_dtype)
            weights = tf.reduce_sum(y_true_one_hot * class_weights_tensor, axis=-1)
            ce_loss = ce_loss * weights

        if self.focal_weight > 0:
            true_class_probs = tf.reduce_sum(y_true_one_hot * probs, axis=-1)
            focal_weight = tf.pow(1.0 - true_class_probs, self.gamma)
            focal_loss = ce_loss * focal_weight
        else:
            focal_loss = ce_loss * 0

        if self.dice_weight > 0:
            y_true_flat = tf.reshape(y_true_one_hot, [-1, tf.shape(y_true_one_hot)[1] * tf.shape(y_true_one_hot)[2], num_classes])
            probs_flat = tf.reshape(probs, [-1, tf.shape(probs)[1] * tf.shape(probs)[2], num_classes])

            intersection = tf.reduce_sum(y_true_flat * probs_flat, axis=1)
            sum_y_true = tf.reduce_sum(y_true_flat, axis=1)
            sum_y_pred = tf.reduce_sum(probs_flat, axis=1)

            dice = (2.0 * intersection + self.epsilon) / (sum_y_true + sum_y_pred + self.epsilon)

            if self.class_weights is not None:
                class_weights_tensor = tf.cast(tf.constant(self.class_weights), compute_dtype)
                dice = dice * class_weights_tensor
                dice_loss = 1.0 - tf.reduce_sum(dice) / tf.reduce_sum(class_weights_tensor)
            else:
                dice_loss = 1.0 - tf.reduce_mean(dice)
        else:
            dice_loss = tf.constant(0.0, dtype=compute_dtype)

        total_loss = (
            self.ce_weight * tf.reduce_mean(ce_loss) +
            self.focal_weight * tf.reduce_mean(focal_loss) +
            self.dice_weight * dice_loss
        )

        return total_loss
