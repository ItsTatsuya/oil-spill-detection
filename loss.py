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
                 ce_weight=0.2,         # Reduced from 0.5 to 0.2
                 focal_weight=0.5,      # Increased from 0.3 to 0.5
                 dice_weight=0.3,       # Increased from 0.2 to 0.3
                 ship_boost_factor=2.0, # New parameter to boost Ship class loss
                 epsilon=1e-7,
                 from_logits=True,
                 name="hybrid_loss"):
        super().__init__(name=name)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32) if class_weights is not None else None
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ship_boost_factor = ship_boost_factor  # Multiplier for Ship class
        self.epsilon = epsilon
        self.from_logits = from_logits
        self.ship_class_index = 3  # Ship class index

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

            # Create modified weights with ship class boosted
            modified_weights = tf.identity(self.class_weights)
            # Apply ship boost factor to ship class weight
            ship_mask = tf.one_hot(self.ship_class_index, depth=num_classes, dtype=tf.float32)
            ship_weight_boost = (self.ship_boost_factor - 1.0) * self.class_weights[self.ship_class_index] * ship_mask
            modified_weights = modified_weights + ship_weight_boost
        else:
            # If no class weights provided, create weights with only ship boosted
            modified_weights = tf.ones(num_classes, dtype=tf.float32)
            modified_weights = modified_weights + (self.ship_boost_factor - 1.0) * tf.one_hot(
                self.ship_class_index, depth=num_classes, dtype=tf.float32
            )

        # ---------- 1. Weighted Cross-Entropy with Hard Mining for Ship Class ----------
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Apply modified weights that include ship boost
        ce_weights = tf.reduce_sum(y_true * modified_weights, axis=-1)
        ce_loss *= ce_weights

        # Hard negative mining - focus more on difficult examples
        # Sort CE losses in descending order and keep top 70% hardest examples
        k = tf.cast(tf.math.ceil(0.7 * tf.cast(tf.size(ce_loss), tf.float32)), tf.int32)
        ce_loss_flat = tf.reshape(ce_loss, [-1])
        ce_loss_sorted = tf.sort(ce_loss_flat, direction='DESCENDING')
        hard_mining_threshold = ce_loss_sorted[k-1]
        hard_ce_mask = tf.cast(ce_loss >= hard_mining_threshold, tf.float32)

        ce_loss = tf.reduce_sum(ce_loss * hard_ce_mask) / (tf.reduce_sum(hard_ce_mask) + self.epsilon)

        # ---------- 2. Focal Loss with More Focus on Ship Class ----------
        # Modified focal loss that puts more emphasis on hard-to-classify examples
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weights = tf.pow(1.0 - pt, 2.0)  # Standard γ=2 for focal loss

        # Apply ship-specific boost to focal weights
        ship_presence = y_true[..., self.ship_class_index:self.ship_class_index+1]
        ship_boost_mask = ship_presence * (self.ship_boost_factor - 1.0)
        focal_weights = focal_weights * (1.0 + ship_boost_mask)

        # Calculate focal loss with modified weights
        focal_loss = tf.abs(y_true - y_pred) * focal_weights

        if modified_weights is not None:
            focal_loss *= modified_weights

        focal_loss = tf.reduce_sum(focal_loss, axis=-1)
        focal_loss = tf.reduce_mean(focal_loss)

        # ---------- 3. Weighted Dice Loss with Ship Class Emphasis ----------
        # Extract ship-specific components for emphasis
        y_true_ship = y_true[..., self.ship_class_index:self.ship_class_index+1]
        y_pred_ship = y_pred[..., self.ship_class_index:self.ship_class_index+1]

        # Calculate Dice scores for all classes
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        y_true_sum = tf.reduce_sum(y_true, axis=[1, 2])
        y_pred_sum = tf.reduce_sum(y_pred, axis=[1, 2])
        dice_score = (2. * intersection + self.epsilon) / (y_true_sum + y_pred_sum + self.epsilon)
        dice_loss = 1. - dice_score

        # Apply modified weights to dice loss
        if modified_weights is not None:
            dice_loss *= modified_weights
            dice_loss = tf.reduce_sum(dice_loss, axis=-1) / tf.reduce_sum(modified_weights)
        else:
            dice_loss = tf.reduce_mean(dice_loss, axis=-1)

        dice_loss = tf.reduce_mean(dice_loss)

        # Add extra ship-specific Dice loss component
        ship_intersection = tf.reduce_sum(y_true_ship * y_pred_ship, axis=[1, 2, 3])
        ship_true_sum = tf.reduce_sum(y_true_ship, axis=[1, 2, 3])
        ship_pred_sum = tf.reduce_sum(y_pred_ship, axis=[1, 2, 3])
        ship_dice_score = (2. * ship_intersection + self.epsilon) / (ship_true_sum + ship_pred_sum + self.epsilon)
        ship_dice_loss = tf.reduce_mean(1. - ship_dice_score)

        # Combine regular dice loss with extra ship-specific dice loss
        total_dice_loss = dice_loss + (self.ship_boost_factor - 1.0) * ship_dice_loss

        # ---------- Final Combined Loss ----------
        total_loss = (self.ce_weight * ce_loss +
                      self.focal_weight * focal_loss +
                      self.dice_weight * total_dice_loss)

        return total_loss
