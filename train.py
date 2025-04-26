import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil # Import shutil for file copying
from tqdm import tqdm
from datetime import datetime
import time
import sys  # For colored terminal output

# ANSI color codes for terminal highlighting
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colored_print(text, color=TermColors.BLUE, bold=False):
    """Print colored text to terminal"""
    prefix = TermColors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{TermColors.ENDC}")

def silent_tf_import():
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
from tensorflow.keras import mixed_precision # type: ignore

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')
policy = mixed_precision.global_policy()
print(f"Mixed precision policy: {policy.name}")

from data_loader import load_dataset
from augmentation import apply_augmentation

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("No GPU found, using CPU")

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, steps_per_epoch, validation_steps=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.step_times = []
        self.epoch_start_time = None

    def on_train_begin(self, logs=None):
        print("\n" + "="*80)
        print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total epochs: {self.total_epochs}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        if self.validation_steps:
            print(f"Validation steps: {self.validation_steps}")
        print("="*80 + "\n")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.step_times = []
        print(f"\nEpoch {epoch+1}/{self.total_epochs} - Starting at {datetime.now().strftime('%H:%M:%S')}")
        self.last_step_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        self.last_step_time = current_time
        if batch % max(1, self.steps_per_epoch // 20) == 0 or batch == self.steps_per_epoch - 1:
            loss = logs.get('loss', 0.0)
            iou = logs.get('iou_metric', 0.0)
            avg_step_time = np.mean(self.step_times[-50:]) if self.step_times else 0
            eta_seconds = avg_step_time * (self.steps_per_epoch - batch) if batch > 0 else 0
            eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}" if batch > 0 else "??:??"
            print(f"Step {batch+1}/{self.steps_per_epoch} - Loss: {loss:.4f} - IoU: {iou:.4f} - "
                  f"{avg_step_time*1000:.1f} ms/step - ETA: {eta_str}")

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        hours, remainder = divmod(epoch_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        val_loss = logs.get('val_loss', 0.0)
        val_iou = logs.get('val_iou_metric', 0.0)
        print(f"\nEpoch {epoch+1}/{self.total_epochs} completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        print(f"Loss: {logs.get('loss', 0.0):.4f} - IoU: {logs.get('iou_metric', 0.0):.4f} - "
              f"Val Loss: {val_loss:.4f} - Val IoU: {val_iou:.4f}")
        if hasattr(self.model.optimizer, 'lr'):
            lr = float(self.model.optimizer.lr.numpy()) if hasattr(self.model.optimizer.lr, 'numpy') else float(self.model.optimizer.lr)
            print(f"Learning rate: {lr:.2e}")
        print("="*80)

class MultiScalePredictor:
    def __init__(self, model, scales=[0.75, 1.0, 1.25], batch_size=2):
        self.model = model
        self.scales = scales
        self.expected_height = model.input_shape[1]
        self.expected_width = model.input_shape[2]
        self.batch_size = batch_size
        self.use_mixed_precision = True

    @tf.function
    def _predict_batch(self, batch):
        return self.model(batch, training=False)

    def predict(self, image_batch):
        batch_size = tf.shape(image_batch)[0]
        height = tf.shape(image_batch)[1]
        width = tf.shape(image_batch)[2]
        all_predictions = []
        for scale in self.scales:
            print(f"Processing scale {scale:.2f}...")
            scaled_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
            scaled_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
            scaled_height = tf.cast(tf.math.ceil(scaled_height / 8) * 8, tf.int32)
            scaled_width = tf.cast(tf.math.ceil(scaled_width / 8) * 8, tf.int32)
            scaled_batch = tf.image.resize(image_batch, size=(scaled_height, scaled_width), method='bilinear')
            model_input = tf.image.resize(scaled_batch, size=(self.expected_height, self.expected_width), method='bilinear')
            if self.use_mixed_precision:
                model_input = tf.cast(model_input, tf.float16)
            logits = self._predict_batch(model_input)
            logits = tf.cast(logits, tf.float32)
            resized_preds = tf.image.resize(logits, size=(height, width), method='bilinear')
            probs = tf.nn.softmax(resized_preds, axis=-1)
            all_predictions.append(probs)
        final_prediction = tf.reduce_mean(all_predictions, axis=0)
        epsilon = 1e-7
        final_prediction = tf.clip_by_value(final_prediction, epsilon, 1 - epsilon)
        final_logits = tf.math.log(final_prediction / (1 - final_prediction + epsilon))
        return final_logits

class IoUMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes=5, name='iou_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.mean_iou = tf.keras.metrics.MeanIoU(num_classes=num_classes)
        self.class_names = ['Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land']

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)
        compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

        def resize_fn():
            resized = tf.image.resize(y_pred, [y_true_shape[1], y_true_shape[2]], method='bilinear')
            return tf.cast(resized, compute_dtype)

        def identity_fn():
            return tf.cast(y_pred, compute_dtype)

        y_pred = tf.cond(
            tf.logical_or(
                tf.not_equal(y_true_shape[1], y_pred_shape[1]),
                tf.not_equal(y_true_shape[2], y_pred_shape[2])
            ),
            resize_fn,
            identity_fn
        )
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, tf.int32)
        self.mean_iou.update_state(y_true, y_pred)

    def result(self):
        return self.mean_iou.result()

    def reset_state(self):
        self.mean_iou.reset_state()

    def get_class_iou(self):
        confusion_matrix = self.mean_iou.total_cm
        sum_over_row = tf.cast(tf.reduce_sum(confusion_matrix, axis=0), tf.float32)
        sum_over_col = tf.cast(tf.reduce_sum(confusion_matrix, axis=1), tf.float32)
        true_positives = tf.cast(tf.linalg.tensor_diag_part(confusion_matrix), tf.float32)
        denominator = sum_over_row + sum_over_col - true_positives
        iou = tf.math.divide_no_nan(true_positives, denominator)
        return {self.class_names[i]: iou[i].numpy() for i in range(self.num_classes)}

class ClassWiseIoUCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to track class-wise IoU metrics during training.
    This ensures the confusion matrix is properly accumulated for accurate class-wise IoU calculation.
    """
    def __init__(self, validation_data, steps, num_classes=5, class_names=None):
        super().__init__()
        self.validation_data = validation_data
        self.steps = steps
        self.num_classes = num_classes
        self.class_names = class_names or ['Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land']
        # Create a dedicated IoU metric for this callback
        self.iou_metric = IoUMetric(num_classes=num_classes)

    def on_epoch_end(self, epoch, logs=None):
        # Reset the metric state at the start of each evaluation
        self.iou_metric.reset_state()

        # Process the validation dataset to fill the confusion matrix
        for i, (images, labels) in enumerate(self.validation_data):
            if i >= self.steps:
                break

            # Get predictions from the model
            predictions = self.model(images, training=False)

            # Update the IoU metric
            self.iou_metric.update_state(labels, predictions)

        # Calculate the overall IoU
        mean_iou = self.iou_metric.result().numpy()

        # Calculate class-wise IoUs
        class_ious = self.iou_metric.get_class_iou()

        # Print class-wise IoUs
        print("\nClass-wise IoU for epoch {}:".format(epoch + 1))
        for class_name, iou_val in class_ious.items():
            print(f"  {class_name}: {iou_val:.4f}")

        # Add to logs if needed (optional, for tensorboard)
        if logs is not None:
            for i, class_name in enumerate(self.class_names):
                logs[f'val_iou_{class_name.replace(" ", "_").lower()}'] = class_ious[class_name]

        # Save class-wise IoU to file
        if not os.path.exists('logs/class_ious'):
            os.makedirs('logs/class_ious')

        # Save to CSV
        with open(f'logs/class_ious/epoch_{epoch+1}_class_ious.csv', 'w') as f:
            f.write("Class,IoU\n")
            for class_name, iou_val in class_ious.items():
                f.write(f"{class_name},{iou_val}\n")
            f.write(f"Mean,{mean_iou}\n")

def create_cosine_decay_with_warmup(epochs, train_steps, batch_size, initial_lr=3e-5, warmup_epochs=10):
    total_steps = epochs * train_steps
    warmup_steps = warmup_epochs * train_steps
    warmup_lr = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0,
        decay_steps=warmup_steps,
        end_learning_rate=initial_lr
    )
    if total_steps <= warmup_steps:
        print(f"Warning: Total steps ({total_steps}) <= warmup steps ({warmup_steps})")
        print(f"Using linear decay from {initial_lr} to 1e-6")
        linear_lr = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps,
            end_learning_rate=1e-6
        )
        class CustomLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __call__(self, step):
                return linear_lr(step)
            def get_config(self):
                return {"initial_lr": initial_lr, "total_steps": total_steps}
        return CustomLRSchedule()
    cosine_lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=0.01
    )
    def lr_schedule(step):
        step = tf.cast(step, tf.float32)
        return tf.cond(
            step < warmup_steps,
            lambda: warmup_lr(step),
            lambda: cosine_lr(step - warmup_steps)
        )
    class CustomLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            return lr_schedule(step)
        def get_config(self):
            return {
                "epochs": epochs,
                "train_steps": train_steps,
                "batch_size": batch_size,
                "initial_lr": initial_lr,
                "warmup_epochs": warmup_epochs
            }
    return CustomLRSchedule()

def _get_current_lr(optimizer):
    """Helper function to get the current learning rate from different optimizer types."""
    try:
        # For Keras 3 optimizers with LearningRateSchedule
        if hasattr(optimizer, 'learning_rate'):
            lr = optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                return float(lr.numpy())
            elif callable(getattr(lr, '__call__', None)):
                return float(lr(optimizer.iterations).numpy())
            else:
                return float(lr)
        # For older optimizers
        elif hasattr(optimizer, 'lr'):
            lr = optimizer.lr
            if hasattr(lr, 'numpy'):
                return float(lr.numpy())
            elif callable(getattr(lr, '__call__', None)):
                return float(lr(optimizer.iterations).numpy())
            else:
                return float(lr)
        else:
            return 0.0
    except Exception as e:
        print(f"Error getting learning rate: {e}")
        return 0.0

def plot_training_curves(history, save_path='miou_curves.png'):
    if not history.history or not history.history.keys():
        print("Warning: Empty training history. Creating placeholder plot.")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No training history available.", horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        plt.savefig(save_path, dpi=300)
        print(f"Empty plot saved to {save_path}")
        return
    plt.figure(figsize=(12, 5))
    available_keys = list(history.history.keys())
    train_iou_key = next((key for key in available_keys if 'iou' in key.lower() and not key.startswith('val_')), None)
    val_iou_key = next((key for key in available_keys if 'iou' in key.lower() and key.startswith('val_')), None)
    if not train_iou_key or not val_iou_key:
        print("Warning: IoU metrics not found. Using available metrics.")
        train_metrics = [key for key in available_keys if not key.startswith('val_')]
        val_metrics = [key for key in available_keys if key.startswith('val_')]
        if train_metrics and val_metrics:
            train_iou_key = train_metrics[0]
            val_iou_key = val_metrics[0]
        else:
            print("No metrics to plot.")
            plt.text(0.5, 0.5, "No metrics available.", horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            plt.savefig(save_path, dpi=300)
            print(f"Empty plot saved to {save_path}")
            return
    plt.subplot(1, 2, 1)
    plt.plot(history.history[train_iou_key], label=f'Training {train_iou_key}')
    plt.plot(history.history[val_iou_key], label=f'Validation {val_iou_key}')
    plt.title('Mean IoU Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    train_loss_key = 'loss' if 'loss' in available_keys else None
    val_loss_key = 'val_loss' if 'val_loss' in available_keys else None
    if train_loss_key and val_loss_key:
        plt.subplot(1, 2, 2)
        plt.plot(history.history[train_loss_key], label='Training Loss')
        plt.plot(history.history[val_loss_key], label='Validation Loss')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")

def save_checkpoint(model, epoch, is_best=False):
    """
    Save simplified checkpoint: latest weights, best weights, and epoch number.

    Args:
        model: The Keras model to save weights from.
        epoch: Current epoch number (1-indexed).
        is_best: Whether this is the best model so far (based on validation metric).
    """
    # Ensure the main checkpoints directory exists
    os.makedirs('checkpoints', exist_ok=True)

    # 1. Save latest weights directly to the root checkpoints directory
    latest_weights_path = 'checkpoints/segformer_b2_latest.weights.h5'
    model.save_weights(latest_weights_path)

    # 2. If this is the best model, save weights to the best path
    if is_best:
        best_weights_path = 'checkpoints/segformer_b2_best.weights.h5'
        model.save_weights(best_weights_path)

    # 3. Save only the epoch number to checkpoint_info.txt
    try:
        with open('checkpoints/checkpoint_info.txt', 'w') as f:
            f.write(str(epoch))
    except Exception as e:
        colored_print(f"Warning: Failed to save epoch number: {e}", color=TermColors.YELLOW, bold=True)

    return latest_weights_path # Return the path to the latest saved weights file

def load_checkpoint(model, optimizer=None, checkpoint_path=None):
    """
    Load a saved checkpoint (weights only) and epoch number.
    Optimizer state is no longer loaded.

    Args:
        model: The Keras model to load weights into.
        optimizer: No longer used, kept for compatibility.
        checkpoint_path: Path to the checkpoint weights file (if None, will use latest).

    Returns:
        tuple: (success, initial_epoch, history={})
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        # Try the conventional location
        checkpoint_path = 'checkpoints/segformer_b2_latest.weights.h5'
        if not os.path.exists(checkpoint_path):
            # Try the best model
            checkpoint_path = 'checkpoints/segformer_b2_best.weights.h5'
            if not os.path.exists(checkpoint_path):
                colored_print("No checkpoint found", color=TermColors.YELLOW, bold=True)
                return False, 0, {}

    # Load the model weights
    colored_print(f"Loading checkpoint weights from {checkpoint_path}", color=TermColors.CYAN, bold=True)

    # Use Keras built-in load_weights method directly instead of weight_loader
    try:
        model.load_weights(checkpoint_path)
        success = True
    except Exception as e:
        colored_print(f"Failed to load checkpoint weights: {e}", color=TermColors.RED, bold=True)
        return False, 0, {}

    # Load only the epoch number from checkpoint_info.txt
    epoch = 0
    history = {} # Return empty history as it's no longer saved
    info_path = 'checkpoints/checkpoint_info.txt'
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                epoch = int(f.read().strip())
        except Exception as e:
            colored_print(f"Warning: Failed to parse epoch from {info_path}: {e}", color=TermColors.YELLOW, bold=True)
            epoch = 0 # Reset to 0 if parsing fails

    colored_print(f"Successfully loaded checkpoint weights. Resuming from epoch {epoch}", color=TermColors.GREEN, bold=True)
    return True, epoch, history

class ModelCheckpointWithHistory(tf.keras.callbacks.Callback):
    """Custom ModelCheckpoint callback that saves comprehensive checkpoints including history."""

    def __init__(self, timestamp, monitor='val_loss', mode='min', verbose=0, save_freq='epoch'):
        super().__init__()
        self.timestamp = timestamp
        self.monitor = monitor
        self.verbose = verbose
        self.save_freq = save_freq
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.mode = mode
        self.epochs_since_improvement = 0
        self.max_patience = 30  # Match early stopping patience

    def on_epoch_end(self, epoch, logs=None):
        if self.save_freq != 'epoch':
            return

        logs = logs or {}
        current_value = logs.get(self.monitor)

        if current_value is None:
            if self.verbose > 0:
                colored_print(f"Warning: Metric '{self.monitor}' not found in logs",
                             color=TermColors.YELLOW)
            return

        is_improvement = False
        if self.mode == 'min' and current_value < self.best_value:
            self.best_value = current_value
            is_improvement = True
        elif self.mode == 'max' and current_value > self.best_value:
            self.best_value = current_value
            is_improvement = True

        # Update patience counter
        if is_improvement:
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        # Calculate metrics for checkpoint
        metrics = {key: float(value) for key, value in logs.items()}

        # Save checkpoint
        weights_path = save_checkpoint(
            model=self.model,
            epoch=epoch + 1,  # epoch is 0-indexed, but we store as 1-indexed
            is_best=is_improvement
        )

        if self.verbose > 0:
            if is_improvement:
                colored_print(f"Epoch {epoch+1}: {self.monitor} improved to {current_value:.4f}, saving model to {weights_path}",
                            color=TermColors.GREEN)
            else:
                msg = f"Epoch {epoch+1}: {self.monitor} did not improve from {self.best_value:.4f}"
                if self.epochs_since_improvement > 0:
                    msg += f" (no improvement for {self.epochs_since_improvement} epochs)"
                colored_print(msg, color=TermColors.YELLOW)

def train_and_evaluate():
    NUM_CLASSES = 5
    BATCH_SIZE = 2  # Reduced batch size for memory efficiency with augmentations
    EPOCHS = 600
    LEARNING_RATE = 3e-5
    IMG_SIZE = (384, 384)

    # Clear GPU memory before starting
    tf.keras.backend.clear_session()
    import gc
    gc.collect()

    # Set GPU memory growth and optimization settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for GPUs to prevent OOM errors")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")

    # Set TensorFlow to use fewer threads to reduce memory pressure
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    print(f"Limited TensorFlow to 2 threads per operation for better memory management")

    os.makedirs('checkpoints', exist_ok=True)
    latest_weights = 'checkpoints/segformer_b2_latest.weights.h5'
    initial_epoch = 0

    print("Loading datasets...")
    train_ds, class_weights_dict, train_steps = load_dataset(data_dir='dataset', split='train', batch_size=BATCH_SIZE)
    test_ds, _, val_steps = load_dataset(data_dir='dataset', split='test', batch_size=BATCH_SIZE)

    # Debug dataset shapes
    for images, labels in train_ds.take(1):
        print(f"Train batch - Images shape: {images.shape}, dtype: {images.dtype}")
        print(f"Train batch - Labels shape: {labels.shape}, dtype: {labels.dtype}")
    for images, labels in test_ds.take(1):
        print(f"Test batch - Images shape: {images.shape}, dtype: {images.dtype}")
        print(f"Test batch - Labels shape: {labels.shape}, dtype: {labels.dtype}")

    colored_print("Applying advanced augmentation with explicit shape checking...", color=TermColors.YELLOW, bold=True)
    augmented_train_ds = apply_augmentation(train_ds, batch_size=BATCH_SIZE, cutmix_prob=0.3)

    # Debug dataset shapes after augmentation
    for images, labels in augmented_train_ds.take(1):
        print(f"Augmented train batch - Images shape: {images.shape}, dtype: {images.dtype}")
        print(f"Augmented train batch - Labels shape: {labels.shape}, dtype: {labels.dtype}")

    # Ensure shapes are valid for training
    @tf.function
    def ensure_shapes(images, labels):
        images = tf.ensure_shape(images, [None, 384, 384, 1])
        labels = tf.ensure_shape(labels, [None, 384, 384, 1])
        return images, labels

    augmented_train_ds = augmented_train_ds.map(ensure_shapes, num_parallel_calls=tf.data.AUTOTUNE)

    print(f"Dataset loaded. Training steps: {train_steps}, Validation steps: {val_steps}")
    if train_steps == 0 or val_steps == 0:
        raise ValueError("Empty dataset detected. Check data loading pipeline.")

    # Import the model and loss function
    from model import OilSpillSegformer
    from loss import HybridSegmentationLoss

    # Create the model explicitly without pretrained weights
    colored_print("Creating OilSpillSegformer model..", color=TermColors.CYAN, bold=True)
    model = OilSpillSegformer(
        input_shape=(*IMG_SIZE, 1),
        num_classes=NUM_CLASSES,
        drop_rate=0.1,
        use_cbam=False,  # Disable CBAM to reduce memory usage
        pretrained_weights=None  # Explicitly set to None to train from scratch
    )

    # Set up the learning rate schedule and optimizer
    lr_schedule = create_cosine_decay_with_warmup(
        epochs=EPOCHS,
        train_steps=train_steps,
        batch_size=BATCH_SIZE,
        initial_lr=LEARNING_RATE,
        warmup_epochs=10
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Check if checkpoint exists (for resuming training, not pretrained weights)
    checkpoint_path = latest_weights if os.path.exists(latest_weights) else None
    if checkpoint_path:
        colored_print(f"Found checkpoint at {checkpoint_path}, but training from scratch as requested.",
                     color=TermColors.YELLOW, bold=True)

    # Set up the loss function
    loss_fn = HybridSegmentationLoss(
        class_weights=class_weights_dict,
        ce_weight=0.4,     # Cross-entropy weight (equivalent to alpha)
        focal_weight=0.3,  # Focal loss weight (equivalent to beta)
        dice_weight=0.3,   # Dice loss weight
        from_logits=True
    )

    # Compile the model with optimized settings for SAR imagery
    colored_print("Compiling model with HybridSegmentationLoss...", color=TermColors.CYAN, bold=True)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[IoUMetric(num_classes=NUM_CLASSES)]
    )

    # Set up callbacks for training
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "fit", f"segformer_b2_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Instantiate the ClassWiseIoUCallback so we can access it later
    class_iou_callback = ClassWiseIoUCallback(
        validation_data=test_ds,
        steps=val_steps,
        num_classes=NUM_CLASSES
    )

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        ),
        ModelCheckpointWithHistory(
            timestamp=timestamp,
            monitor='val_iou_metric',
            mode='max',
            verbose=1,
            save_freq='epoch'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_iou_metric',
            patience=30,
            verbose=1,
            mode='max',
            restore_best_weights=True
        ),
        ProgressCallback(
            total_epochs=EPOCHS,
            steps_per_epoch=train_steps,
            validation_steps=val_steps
        ),
        class_iou_callback  # Add the instance to the list
    ]

    # Train the model
    colored_print(f"Starting model training for {EPOCHS} epochs from epoch {initial_epoch}...",
                 color=TermColors.GREEN, bold=True)
    history = model.fit(
        augmented_train_ds,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        steps_per_epoch=train_steps,
        validation_data=test_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=0  # Let our custom callback handle the outputs
    )

    # Save training curves
    plot_training_curves(history, save_path=f'logs/fit/segformer_b2_{timestamp}/training_curves.png')

    # Save the final epoch number for resuming later
    with open('checkpoints/checkpoint_info.txt', 'w') as f:
        f.write(str(initial_epoch + len(history.history.get('loss', []))))

    # Evaluate the model on validation set
    colored_print("Evaluating final model performance...", color=TermColors.CYAN, bold=True)

    # Run model.evaluate to get the standard final loss and mean IoU reported by Keras
    results = model.evaluate(test_ds, steps=val_steps, verbose=1)
    loss, mean_iou_keras = results  # Get loss and mean IoU from Keras evaluation
    colored_print(f"Final Keras Metrics - Loss: {loss:.4f}, Mean IoU: {mean_iou_keras:.4f}",
                 color=TermColors.GREEN, bold=True)

    # Get the final class-wise IoU results directly from the callback's metric instance
    # This uses the confusion matrix accumulated by the callback during the last epoch's evaluation
    colored_print("Final Class-wise IoU (from callback):", color=TermColors.CYAN, bold=True)
    try:
        # Access the iou_metric instance within the callback
        final_class_ious = class_iou_callback.iou_metric.get_class_iou()
        final_mean_iou_callback = class_iou_callback.iou_metric.result().numpy()

        for class_name, iou_val in final_class_ious.items():
            print(f"  {class_name}: {iou_val:.4f}")
        print(f"  Mean IoU (Callback): {final_mean_iou_callback:.4f}")

        # Save the final class IoUs to a file
        if not os.path.exists('logs/class_ious'):
            os.makedirs('logs/class_ious')

        with open(f'logs/class_ious/final_class_ious.csv', 'w') as f:
            f.write("Class,IoU\n")
            for class_name, iou_val in final_class_ious.items():
                f.write(f"{class_name},{iou_val}\n")
            f.write(f"Mean,{final_mean_iou_callback}\n")
        colored_print("Final class-wise IoU saved to logs/class_ious/final_class_ious.csv",
                     color=TermColors.GREEN)

    except Exception as e:
        colored_print(f"Could not retrieve final class-wise IoU from callback: {e}",
                     color=TermColors.RED, bold=True)
        # Fallback: Try getting from model.metrics again, though likely won't work well
        iou_metric_keras = None
        for metric in model.metrics:
            if isinstance(metric, IoUMetric):
                iou_metric_keras = metric
                break
        if iou_metric_keras:
            try:
                class_ious_keras = iou_metric_keras.get_class_iou()
                colored_print("Class-wise IoU (from Keras metric - may be inaccurate):", color=TermColors.YELLOW, bold=True)
                for class_name, iou_val in class_ious_keras.items():
                    print(f"  {class_name}: {iou_val:.4f}")
            except Exception as inner_e:
                colored_print(f"Could not retrieve class-wise IoU from Keras metric either: {inner_e}",
                             color=TermColors.RED, bold=True)
        else:
            colored_print("IoUMetric not found in model.metrics.", color=TermColors.RED, bold=True)

    # Save the final model
    model.save_weights('checkpoints/segformer_b2_final.weights.h5')
    colored_print("Training complete. Final weights saved to checkpoints/segformer_b2_final.weights.h5",
                 color=TermColors.GREEN, bold=True)

    return history

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    print("____________________________________________________________________")
    print("Starting SegFormer-B2 training for oil spill detection...")
    history = train_and_evaluate()
    print("SegFormer-B2 training completed successfully")
    print("____________________________________________________________________")
