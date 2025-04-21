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

def weighted_cross_entropy(class_weights=None, from_logits=True, epsilon=1e-7):
    """
    Weighted cross-entropy loss function for semantic segmentation.

    Args:
        class_weights: List of weights for each class
        from_logits: Whether y_pred is expected to be logits
        epsilon: Small constant to avoid numerical instability

    Returns:
        Weighted cross-entropy loss function
    """
    if class_weights is not None:
        class_weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # One-hot encode y_true if needed
        if y_true.shape[-1] != y_pred.shape[-1]:
            y_true = tf.squeeze(y_true, axis=-1)
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Add epsilon for numerical stability
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calculate cross-entropy
        ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        # Apply class weights if provided
        if class_weights is not None:
            weights = tf.reduce_sum(y_true * class_weights, axis=-1)
            ce_loss *= weights

        return tf.reduce_mean(ce_loss)

    return loss

def focal_loss(class_weights=None, gamma=2.0, from_logits=True, epsilon=1e-7):
    """
    Focal loss for addressing class imbalance with gamma focusing parameter.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        class_weights: List of alpha weights for each class
        gamma: Focusing parameter that reduces the loss contribution from easy examples
               Higher gamma values focus more on hard, misclassified examples
        from_logits: Whether y_pred is expected to be logits
        epsilon: Small constant to avoid numerical instability

    Returns:
        Focal loss function
    """
    if class_weights is not None:
        class_weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # One-hot encode y_true if needed
        if y_true.shape[-1] != y_pred.shape[-1]:
            y_true = tf.squeeze(y_true, axis=-1)
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Add epsilon for numerical stability
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calculate focal loss with gamma focusing parameter
        # (1 - p_t)^gamma * log(p_t) where p_t is the probability of the true class
        # For true class: p_t = y_pred
        # For false class: p_t = 1 - y_pred
        focal_weight = tf.pow(1.0 - y_pred, gamma)
        focal_ce = -tf.reduce_sum(focal_weight * y_true * tf.math.log(y_pred), axis=-1)

        # Apply class weights if provided
        if class_weights is not None:
            weights = tf.reduce_sum(y_true * class_weights, axis=-1)
            focal_ce *= weights

        return tf.reduce_mean(focal_ce)

    return loss

def dice_loss(class_weights=None, from_logits=True, epsilon=1e-7):
    """
    Dice loss for semantic segmentation.

    Args:
        class_weights: List of weights for each class
        from_logits: Whether y_pred is expected to be logits
        epsilon: Small constant to avoid numerical instability

    Returns:
        Dice loss function
    """
    if class_weights is not None:
        class_weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # One-hot encode y_true if needed
        if y_true.shape[-1] != y_pred.shape[-1]:
            y_true = tf.squeeze(y_true, axis=-1)
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Calculate Dice coefficient for each class
        # Flatten spatial dimensions
        y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])

        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)

        # Calculate Dice coefficient
        dice = (2. * intersection + epsilon) / (union + epsilon)

        # Apply class weights if provided
        if class_weights is not None:
            dice = dice * class_weights
            dice_loss = 1.0 - tf.reduce_mean(dice / tf.reduce_sum(class_weights))
        else:
            dice_loss = 1.0 - tf.reduce_mean(dice)

        return dice_loss

    return loss

# Maintain hybrid_loss for backward compatibility
hybrid_loss = weighted_cross_entropy()

# Enhanced HybridSegmentationLoss class with gamma parameter for focal loss
class HybridSegmentationLoss(tf.keras.losses.Loss):
    def __init__(self,
                 class_weights=None,
                 ce_weight=0.4,  # Default weight for cross entropy
                 focal_weight=0.3,  # Default weight for focal loss
                 dice_weight=0.3,  # Default weight for dice loss
                 gamma=2.0,  # Gamma parameter for focal loss
                 epsilon=1e-7,
                 from_logits=True,
                 name="hybrid_loss"):
        """
        Hybrid loss function combining weighted cross-entropy, focal loss, and dice loss.

        Args:
            class_weights: List of weights for each class
            ce_weight: Weight for cross-entropy component
            focal_weight: Weight for focal loss component
            dice_weight: Weight for dice loss component
            gamma: Focusing parameter for focal loss
            epsilon: Small constant for numerical stability
            from_logits: Whether predictions are logits
            name: Name of the loss function
        """
        super().__init__(name=name)
        self.class_weights = class_weights
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.epsilon = epsilon
        self.from_logits = from_logits

        # Initialize component loss functions
        self.weighted_ce = weighted_cross_entropy(class_weights, from_logits, epsilon)
        self.focal = focal_loss(class_weights, gamma, from_logits, epsilon)
        self.dice = dice_loss(class_weights, from_logits, epsilon)

        # Validate weights
        assert ce_weight + focal_weight + dice_weight > 0, "At least one loss weight must be positive"

        # Print configuration
        tf.print("Hybrid Loss Config - CE:", ce_weight, "Focal:", focal_weight,
                 "Dice:", dice_weight, "Gamma:", gamma)

    def call(self, y_true, y_pred):
        # Combine the three loss components
        loss_value = 0.0

        if self.ce_weight > 0:
            ce_loss = self.weighted_ce(y_true, y_pred)
            loss_value += self.ce_weight * ce_loss

        if self.focal_weight > 0:
            fl_loss = self.focal(y_true, y_pred)
            loss_value += self.focal_weight * fl_loss

        if self.dice_weight > 0:
            dc_loss = self.dice(y_true, y_pred)
            loss_value += self.dice_weight * dc_loss

        return loss_value
