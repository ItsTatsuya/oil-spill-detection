def silent_tf_import():
    import os
    import sys
    orig_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(orig_stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, orig_stderr_fd)
    os.close(devnull_fd)

    import tensorflow as tf

    os.dup2(saved_stderr_fd, orig_stderr_fd)
    os.close(saved_stderr_fd)

    return tf

tf = silent_tf_import()

class HybridSegmentationLoss(tf.keras.losses.Loss):
    def __init__(self,
                 class_weights=None,
                 ce_weight=0.5,
                 focal_weight=0.3,
                 dice_weight=0.2,
                 epsilon=1e-7,
                 from_logits=True,
                 name="hybrid_loss"):
        super().__init__(name=name)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32) if class_weights is not None else None
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.epsilon = epsilon
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        # One-hot encode y_true if needed
        if y_true.shape[-1] != y_pred.shape[-1]:
            y_true = tf.squeeze(y_true, axis=-1)
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Shape check
        tf.debugging.assert_equal(tf.shape(y_true), tf.shape(y_pred), message="Shapes of y_true and y_pred must match.")

        num_classes = tf.shape(y_pred)[-1]

        if self.class_weights is not None:
            tf.debugging.assert_equal(tf.shape(self.class_weights)[0], num_classes,
                                      message="class_weights length must match number of classes")

        # ---------- 1. Weighted Cross-Entropy ----------
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        if self.class_weights is not None:
            weights = tf.reduce_sum(y_true * self.class_weights, axis=-1)
            ce_loss *= weights
        ce_loss = tf.reduce_mean(ce_loss)

        # ---------- 2. Focal Loss (MSE-style variant) ----------
        focal_loss = tf.square(y_true - y_pred)
        if self.class_weights is not None:
            focal_loss *= self.class_weights
        focal_loss = tf.reduce_sum(focal_loss, axis=-1)  # sum across channels
        focal_loss = tf.reduce_mean(focal_loss)          # average over batch

        # ---------- 3. Weighted Dice Loss ----------
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        y_true_sum = tf.reduce_sum(y_true, axis=[1, 2])
        y_pred_sum = tf.reduce_sum(y_pred, axis=[1, 2])
        dice_score = (2. * intersection + self.epsilon) / (y_true_sum + y_pred_sum + self.epsilon)
        dice_loss = 1. - dice_score

        if self.class_weights is not None:
            dice_loss *= self.class_weights
            dice_loss = tf.reduce_sum(dice_loss, axis=-1) / tf.reduce_sum(self.class_weights)
        else:
            dice_loss = tf.reduce_mean(dice_loss, axis=-1)

        dice_loss = tf.reduce_mean(dice_loss)

        # ---------- Final Combined Loss ----------
        total_loss = (self.ce_weight * ce_loss +
                      self.focal_weight * focal_loss +
                      self.dice_weight * dice_loss)

        return total_loss
