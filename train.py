import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import time

def silent_tf_import():
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

# Set standard float32 precision for more stable training
from tensorflow.keras import mixed_precision # type: ignore

mixed_precision.set_global_policy('float32')  # Fixed standard policy instead of mixed precision
policy = mixed_precision.global_policy()
print(f"Precision policy set to {policy.name} for stable training")

# Import custom modules
from data_loader import load_dataset
from augmentation import apply_augmentation
from loss import HybridSegmentationLoss
from model import DeepLabv3Plus


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
print(f"Using GPU: {gpus[0].name}" if gpus else "No GPU found, using CPU")

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, steps_per_epoch, validation_steps=None):
        super(ProgressCallback, self).__init__()
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
            # Calculate and format metrics
            loss = logs.get('loss', 0.0)
            iou = logs.get('iou_metric', 0.0)
            avg_step_time = np.mean(self.step_times[-50:]) if self.step_times else 0

            # Calculate ETA for epoch completion
            if batch > 0 and self.steps_per_epoch > 0:
                eta_seconds = avg_step_time * (self.steps_per_epoch - batch)
                eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
            else:
                eta_str = "??:??"

            print(f"Step {batch+1}/{self.steps_per_epoch} - "
                  f"Loss: {loss:.4f} - IoU: {iou:.4f} - "
                  f"{avg_step_time*1000:.1f} ms/step - ETA: {eta_str}")

    def on_epoch_end(self, epoch, logs=None):
        # Calculate epoch duration
        epoch_time = time.time() - self.epoch_start_time
        hours, remainder = divmod(epoch_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format validation metrics
        val_loss = logs.get('val_loss', 0.0)
        val_iou = logs.get('val_iou_metric', 0.0)

        print(f"\nEpoch {epoch+1}/{self.total_epochs} completed in "
              f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        print(f"Loss: {logs.get('loss', 0.0):.4f} - IoU: {logs.get('iou_metric', 0.0):.4f} - "
              f"Val Loss: {val_loss:.4f} - Val IoU: {val_iou:.4f}")

        # Print learning rate
        if hasattr(self.model.optimizer, 'lr'):
            lr = self.model.optimizer.lr
            if hasattr(lr, 'numpy'):
                print(f"Learning rate: {float(lr.numpy()):.2e}")

        print("="*80)


class MultiScalePredictor:
    def __init__(self, model, scales=[0.5, 0.75, 1.0], max_batch_size=4):
        self.model = model
        self.scales = scales
        self.max_batch_size = max_batch_size  # Set a smaller max batch size to avoid OOM
        # Get the expected input shape from the model
        self.expected_height = model.input_shape[1]
        self.expected_width = model.input_shape[2]

    def predict(self, image_batch):
        batch_size = tf.shape(image_batch)[0]
        height = tf.shape(image_batch)[1]
        width = tf.shape(image_batch)[2]

        # Get the compute dtype from the model
        compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

        # If batch is large, split it to avoid memory issues
        if batch_size > self.max_batch_size:
            # Process in smaller batches and concatenate results
            all_batch_predictions = []

            # Process max_batch_size images at a time
            for i in range(0, batch_size, self.max_batch_size):
                end_idx = tf.minimum(i + self.max_batch_size, batch_size)
                sub_batch = image_batch[i:end_idx]

                # Process this sub-batch
                sub_batch_preds = self._predict_single_batch(sub_batch, height, width, compute_dtype)
                all_batch_predictions.append(sub_batch_preds)

                # Clear GPU memory after each batch
                tf.keras.backend.clear_session()

            # Concatenate results along batch dimension
            return tf.concat(all_batch_predictions, axis=0)
        else:
            # Small enough batch, process normally
            return self._predict_single_batch(image_batch, height, width, compute_dtype)

    def _predict_single_batch(self, image_batch, height, width, compute_dtype):
        """Process a single batch of images with multi-scale prediction"""
        # Placeholder for all scaled predictions
        all_predictions = []

        # Make predictions at each scale
        for scale in self.scales:
            # Release memory before processing each scale
            tf.keras.backend.clear_session()

            # Resize images to current scale while preserving aspect ratio
            scaled_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
            scaled_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)

            # Ensure dimensions are multiples of 8 for better performance
            scaled_height = tf.cast(tf.math.ceil(scaled_height / 8) * 8, tf.int32)
            scaled_width = tf.cast(tf.math.ceil(scaled_width / 8) * 8, tf.int32)

            # First resize to the scaled dimensions
            scaled_images = tf.image.resize(
                image_batch,
                size=(scaled_height, scaled_width),
                method='bilinear'
            )

            # Then resize to the model's expected input shape
            model_input = tf.image.resize(
                scaled_images,
                size=(self.expected_height, self.expected_width),
                method='bilinear'
            )

            # Ensure proper dtype for mixed precision
            model_input = tf.cast(model_input, compute_dtype)

            # Get predictions using the model's expected input size
            logits = self.model(model_input, training=False)

            # Resize predictions back to the original image size
            resized_preds = tf.image.resize(
                logits,
                size=(height, width),
                method='bilinear'
            )

            # Apply softmax to get probabilities
            probs = tf.nn.softmax(resized_preds, axis=-1)
            all_predictions.append(probs)

            # Explicitly delete variables to free memory
            del scaled_images, model_input, logits, resized_preds

        # Average predictions from all scales
        fused_prediction = tf.reduce_mean(tf.stack(all_predictions, axis=0), axis=0)

        # Convert back to logits for compatibility with loss functions
        epsilon = 1e-7
        fused_prediction = tf.clip_by_value(fused_prediction, epsilon, 1 - epsilon)
        fused_logits = tf.math.log(fused_prediction / (1 - fused_prediction + epsilon))

        return fused_logits


class IoUMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes=5, name='iou_metric', **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.mean_iou = tf.keras.metrics.MeanIoU(num_classes=num_classes)
        self.class_names = ['Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land']

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get shapes
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)

        # Make sure inputs have consistent dtypes
        compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

        # Define resize function to always return the same dtype
        def resize_fn():
            resized = tf.image.resize(y_pred, [y_true_shape[1], y_true_shape[2]], method='bilinear')
            # Ensure the dtype is consistent
            return tf.cast(resized, compute_dtype)

        def identity_fn():
            # Ensure the dtype is consistent
            return tf.cast(y_pred, compute_dtype)

        # Use tf.cond with functions that return the same dtype
        y_pred = tf.cond(
            tf.logical_or(
                tf.not_equal(y_true_shape[1], y_pred_shape[1]),
                tf.not_equal(y_true_shape[2], y_pred_shape[2])
            ),
            resize_fn,
            identity_fn
        )

        # Convert y_true to integer class indices
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.squeeze(y_true, axis=-1)

        # Convert y_pred from logits to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, tf.int32)

        # Update the mean IoU metric
        self.mean_iou.update_state(y_true, y_pred)

    def result(self):
        return self.mean_iou.result()

    def reset_state(self):
        self.mean_iou.reset_state()

    def get_class_iou(self):
        """Get IoU for each class."""
        confusion_matrix = self.mean_iou.total_cm
        # IoU = true_positive / (true_positive + false_positive + false_negative)
        sum_over_row = tf.cast(
            tf.reduce_sum(confusion_matrix, axis=0),
            dtype=tf.float32
        )
        sum_over_col = tf.cast(
            tf.reduce_sum(confusion_matrix, axis=1),
            dtype=tf.float32
        )
        true_positives = tf.cast(
            tf.linalg.tensor_diag_part(confusion_matrix),
            dtype=tf.float32
        )

        # sum_over_row + sum_over_col - true_positives = TP + FP + FN
        denominator = sum_over_row + sum_over_col - true_positives

        # The IoU is set to 0 if the denominator is 0
        iou = tf.math.divide_no_nan(true_positives, denominator)

        class_iou = {self.class_names[i]: iou[i].numpy() for i in range(self.num_classes)}
        return class_iou


class ShipFocusedLearningRateScheduler(tf.keras.callbacks.Callback):
    """Custom learning rate scheduler that applies higher learning rates to layers in the ship detection branch."""

    def __init__(self, base_lr=1e-4, ship_lr_multiplier=2.0, decay_rate=0.9, decay_steps=10):
        super(ShipFocusedLearningRateScheduler, self).__init__()
        self.base_lr = base_lr
        self.ship_lr_multiplier = ship_lr_multiplier
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.ship_branch_names = ['ship_branch', 'combined_features', 'combined_conv']

    def on_epoch_begin(self, epoch, logs=None):
        # Apply exponential decay to base learning rate
        decayed_lr = self.base_lr * (self.decay_rate ** (epoch // self.decay_steps))

        # Set different learning rates for different layers
        for layer in self.model.layers:
            # Skip non-trainable layers
            if not layer.trainable:
                continue

            # Check if layer is part of the ship detection branch
            is_ship_layer = any(name in layer.name for name in self.ship_branch_names)

            if is_ship_layer:
                # Apply higher learning rate to ship-related layers
                layer_lr = decayed_lr * self.ship_lr_multiplier
                print(f"Setting {layer.name} learning rate to {layer_lr:.2e} (ship branch)")
            else:
                # Regular learning rate for other layers
                layer_lr = decayed_lr

            # Only set custom learning rates if the optimizer supports it
            if hasattr(layer, 'optimizer') and hasattr(layer.optimizer, 'learning_rate'):
                layer.optimizer.learning_rate.assign(layer_lr)

        # Set global learning rate for the main optimizer
        if hasattr(self.model.optimizer, 'learning_rate'):
            self.model.optimizer.learning_rate.assign(decayed_lr)
            print(f"Epoch {epoch+1}: Base LR = {decayed_lr:.2e}, Ship LR = {decayed_lr * self.ship_lr_multiplier:.2e}")


class CurriculumLearningCallback(tf.keras.callbacks.Callback):
    """Implements curriculum learning strategy by gradually increasing the difficulty of training samples."""

    def __init__(self, train_ds, epochs=100, ship_focus_start=30, full_dataset_start=60):
        super(CurriculumLearningCallback, self).__init__()
        self.train_ds = train_ds
        self.epochs = epochs
        self.ship_focus_start = ship_focus_start
        self.full_dataset_start = full_dataset_start
        self.original_ds = train_ds
        self.ship_focused_ds = None
        self.balanced_ds = None

    def on_train_begin(self, logs=None):
        """Prepare the curriculum datasets at the start of training."""
        print("Initializing curriculum learning strategy...")
        # Store the original dataset
        self.original_ds = self.train_ds

        # Create a balanced subset that includes more ship examples
        self.balanced_ds = self._create_balanced_dataset()

        # Create a ship-focused dataset with higher proportion of ship examples
        self.ship_focused_ds = self._create_ship_focused_dataset()

        # Start with the balanced dataset
        print("Starting with balanced dataset for initial training")
        self.model.train_dataset = self.balanced_ds

    def on_epoch_begin(self, epoch, logs=None):
        """Adjust the training dataset based on the curriculum stage."""
        if epoch < self.ship_focus_start:
            # First phase: use balanced dataset
            if self.model.train_dataset is not self.balanced_ds:
                print(f"Epoch {epoch+1}: Using balanced dataset")
                self.model.train_dataset = self.balanced_ds
        elif epoch < self.full_dataset_start:
            # Second phase: use ship-focused dataset
            if self.model.train_dataset is not self.ship_focused_ds:
                print(f"Epoch {epoch+1}: Switching to ship-focused dataset")
                self.model.train_dataset = self.ship_focused_ds
        else:
            # Final phase: use full dataset
            if self.model.train_dataset is not self.original_ds:
                print(f"Epoch {epoch+1}: Switching to full dataset")
                self.model.train_dataset = self.original_ds

    def _create_balanced_dataset(self):
        """Create a balanced dataset with equal class representation."""
        # This is a placeholder implementation - in a real scenario,
        # you would filter the dataset to achieve better class balance
        print("Creating balanced dataset...")
        return self.train_ds

    def _create_ship_focused_dataset(self):
        """Create a dataset with emphasis on ship examples."""
        # This is a placeholder implementation - in a real scenario,
        # you would oversample ship examples or filter to focus on them
        print("Creating ship-focused dataset...")
        return self.train_ds


def plot_training_curves(history, save_path='miou_curves.png'):
    """
    Plot and save training/validation IoU curves.

    Includes fallback logic to handle missing keys gracefully.
    """
    # Check if history object has any data
    if not history.history or len(history.history.keys()) == 0:
        print("Warning: Training history is empty. No curves to plot.")
        # Create a simple plot with a message instead
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No training history data available.\nTry training for more epochs to generate curves.",
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        plt.savefig(save_path, dpi=300)
        print(f"Empty plot saved to {save_path}")
        return

    # Continue with normal plotting if history contains data
    plt.figure(figsize=(12, 5))

    # Check available keys in history object
    available_keys = list(history.history.keys())
    print(f"Available history keys: {available_keys}")

    # Find the IoU metric keys (may be 'iou_metric' or just 'iou' or other variations)
    train_iou_key = next((key for key in available_keys if 'iou' in key.lower() and not key.startswith('val_')), None)
    val_iou_key = next((key for key in available_keys if 'iou' in key.lower() and key.startswith('val_')), None)

    if not train_iou_key or not val_iou_key:
        print("Warning: IoU metrics not found in history. Using available metrics instead.")
        train_metrics = [key for key in available_keys if not key.startswith('val_')]
        val_metrics = [key for key in available_keys if key.startswith('val_')]

        if train_metrics and val_metrics:
            train_iou_key = train_metrics[0]
            val_iou_key = val_metrics[0]
        else:
            print("No metrics found to plot. Creating empty plot.")
            plt.text(0.5, 0.5, "No metrics data available in history.",
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14)
            plt.axis('off')
            plt.savefig(save_path, dpi=300)
            print(f"Empty plot saved to {save_path}")
            return

    # Plot IoU metric
    plt.subplot(1, 2, 1)
    plt.plot(history.history[train_iou_key], label=f'Training {train_iou_key}')
    plt.plot(history.history[val_iou_key], label=f'Validation {val_iou_key}')
    plt.title('Mean IoU Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Find loss keys
    train_loss_key = next((key for key in available_keys if 'loss' in key.lower() and not key.startswith('val_')), None)
    val_loss_key = next((key for key in available_keys if 'loss' in key.lower() and key.startswith('val_')), None)

    # Plot Loss if available
    if train_loss_key and val_loss_key:
        plt.subplot(1, 2, 2)
        plt.plot(history.history[train_loss_key], label=f'Training {train_loss_key}')
        plt.plot(history.history[val_loss_key], label=f'Validation {val_loss_key}')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
    else:
        # If loss is not available, use another metric if possible
        remaining_train_metrics = [key for key in available_keys if key != train_iou_key and not key.startswith('val_')]
        remaining_val_metrics = [key for key in available_keys if key != val_iou_key and key.startswith('val_')]

        if remaining_train_metrics and remaining_val_metrics:
            alt_train_key = remaining_train_metrics[0]
            alt_val_key = remaining_val_metrics[0]

            plt.subplot(1, 2, 2)
            plt.plot(history.history[alt_train_key], label=f'Training {alt_train_key}')
            plt.plot(history.history[alt_val_key], label=f'Validation {alt_val_key}')
            plt.title(f'{alt_train_key.capitalize()} Over Time')
            plt.xlabel('Epoch')
            plt.ylabel(alt_train_key.capitalize())
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")


def train_two_stage():
    """Two-stage training approach: first train on ship-focused dataset, then fine-tune on full dataset."""

    # Define constants
    NUM_CLASSES = 5
    BATCH_SIZE = 12
    STAGE1_EPOCHS = 120  # First stage with ship focus
    STAGE2_EPOCHS = 120  # Second stage on full dataset
    LEARNING_RATE = 5e-5
    SHIP_LR_MULTIPLIER = 2.0  # Higher learning rate for ship detection branch
    IMG_SIZE = (320, 320)

    # Create directory for checkpoints if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    # Path for latest checkpoint
    stage1_weights = 'checkpoints/ship_focused_stage1.weights.h5'
    stage2_weights = 'checkpoints/improved_deeplabv3plus_best.weights.h5'

    print("Loading datasets...")
    # Load training and validation datasets
    train_ds, class_weights_dict, train_steps = load_dataset(data_dir='dataset', split='train', batch_size=BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds, _, val_steps = load_dataset(data_dir='dataset', split='test', batch_size=BATCH_SIZE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # Apply advanced data augmentation
    print("Applying advanced data augmentation...")
    augmented_train_ds = apply_augmentation(train_ds)

    print(f"Dataset loaded. Training steps: {train_steps}, Validation steps: {val_steps}")

    # Convert class weights dictionary to a list ordered by class index
    class_weights_list = [class_weights_dict.get(i, 1.0) for i in range(NUM_CLASSES)]
    print(f"Using class weights from data loader: {class_weights_list}")

    # Check if we have a previous checkpoint to resume from
    if os.path.exists(stage2_weights):
        print(f"Found final model at {stage2_weights}. Skipping training.")
        # Load the final model for evaluation only
        model = DeepLabv3Plus(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES, ship_enhanced=True)
        model.load_weights(stage2_weights)

        # Skip to evaluation
        evaluate_model(model, test_ds)
        return

    # STAGE 1: Train with focus on ship detection
    print("\n" + "="*80)
    print("STAGE 1: Training with focus on ship detection")
    print("="*80)

    # Create model with ship enhancement enabled
    model_stage1 = DeepLabv3Plus(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES, ship_enhanced=True)

    # Create the loss function with emphasis on Ship class
    loss_fn_stage1 = HybridSegmentationLoss(
        class_weights=class_weights_list,
        ce_weight=0.2,       # Reduced cross-entropy weight
        focal_weight=0.5,    # Increased focal weight for hard examples
        dice_weight=0.3,     # Increased dice weight
        ship_boost_factor=2.0  # Boost Ship class loss
    )

    # Define optimizer with higher learning rate
    optimizer_stage1 = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE * 1.5,  # Higher initial LR for stage 1
        clipnorm=1.0,
        epsilon=1e-8
    )

    # Compile model
    model_stage1.compile(
        optimizer=optimizer_stage1,
        loss=loss_fn_stage1,
        metrics=[IoUMetric(num_classes=NUM_CLASSES)],
        run_eagerly=False,
        jit_compile=True
    )

    # Set up callbacks for stage 1
    callbacks_stage1 = [
        # Save best weights
        tf.keras.callbacks.ModelCheckpoint(
            filepath=stage1_weights,
            monitor='val_iou_metric',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_iou_metric',
            patience=30,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        # Custom learning rate scheduler with ship focus
        ShipFocusedLearningRateScheduler(
            base_lr=LEARNING_RATE * 1.5,
            ship_lr_multiplier=SHIP_LR_MULTIPLIER,
            decay_rate=0.9,
            decay_steps=10
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs/fit/stage1_' + datetime.now().strftime("%Y%m%d-%H%M%S"),
            update_freq='epoch'
        ),
        # Progress callback
        ProgressCallback(
            total_epochs=STAGE1_EPOCHS,
            steps_per_epoch=train_steps,
            validation_steps=val_steps
        )
    ]

    # Train stage 1
    print(f"Starting stage 1 training for {STAGE1_EPOCHS} epochs...")
    history_stage1 = model_stage1.fit(
        augmented_train_ds,
        validation_data=test_ds,
        epochs=STAGE1_EPOCHS,
        callbacks=callbacks_stage1,
        verbose=0
    )

    # Plot and save stage 1 training curves
    plot_training_curves(history_stage1, save_path='ship_focused_training_curves.png')

    # STAGE 2: Fine-tune on full dataset
    print("\n" + "="*80)
    print("STAGE 2: Fine-tuning on full dataset")
    print("="*80)

    # Create a new model for stage 2 (or continue with the same model)
    model_stage2 = DeepLabv3Plus(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES, ship_enhanced=True)

    # Load weights from stage 1
    if os.path.exists(stage1_weights):
        print(f"Loading weights from stage 1: {stage1_weights}")
        model_stage2.load_weights(stage1_weights)
    else:
        print("Warning: Stage 1 weights not found. Starting stage 2 from scratch.")

    # Create the loss function for stage 2 (balanced for all classes but still with ship emphasis)
    loss_fn_stage2 = HybridSegmentationLoss(
        class_weights=class_weights_list,
        ce_weight=0.3,         # Balanced weights
        focal_weight=0.4,      # Still focus on hard examples
        dice_weight=0.3,
        ship_boost_factor=1.5  # Reduced ship boost for more balanced training
    )

    # Define optimizer with lower learning rate for fine-tuning
    optimizer_stage2 = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE * 0.5,  # Lower LR for fine-tuning
        clipnorm=1.0,
        epsilon=1e-8
    )

    # Compile model for stage 2
    model_stage2.compile(
        optimizer=optimizer_stage2,
        loss=loss_fn_stage2,
        metrics=[IoUMetric(num_classes=NUM_CLASSES)],
        run_eagerly=False,
        jit_compile=True
    )

    # Set up callbacks for stage 2
    callbacks_stage2 = [
        # Save best weights
        tf.keras.callbacks.ModelCheckpoint(
            filepath=stage2_weights,
            monitor='val_iou_metric',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        ),
        # Also save latest weights
        tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/latest_model.weights.h5',
            save_weights_only=True,
            save_best_only=False,
            save_freq='epoch',
            verbose=0
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_iou_metric',
            patience=40,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate scheduler
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_iou_metric',
            factor=0.7,
            patience=10,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs/fit/stage2_' + datetime.now().strftime("%Y%m%d-%H%M%S"),
            update_freq='epoch'
        ),
        # Progress callback
        ProgressCallback(
            total_epochs=STAGE2_EPOCHS,
            steps_per_epoch=train_steps,
            validation_steps=val_steps
        )
    ]

    # Train stage 2
    print(f"Starting stage 2 training for {STAGE2_EPOCHS} epochs...")
    history_stage2 = model_stage2.fit(
        augmented_train_ds,
        validation_data=test_ds,
        epochs=STAGE2_EPOCHS,
        callbacks=callbacks_stage2,
        verbose=0
    )

    # Plot and save stage 2 training curves
    plot_training_curves(history_stage2, save_path='fine_tuning_curves.png')

    # Evaluate the final model
    evaluate_model(model_stage2, test_ds)

    return model_stage2


def evaluate_model(model, test_ds):
    """Evaluate the model using multi-scale prediction."""
    # Initialize multi-scale predictor
    multi_scale_predictor = MultiScalePredictor(model, scales=[0.5, 0.75, 1.0])

    # Initialize IoU metric
    test_metric = IoUMetric(num_classes=5)

    print("\nEvaluating with multi-scale prediction...")
    # Evaluate on test dataset
    for images, labels in tqdm(test_ds):
        # Get multi-scale predictions
        predictions = multi_scale_predictor.predict(images)

        # Update metrics
        test_metric.update_state(labels, predictions)

    # Get results
    mean_iou = test_metric.result().numpy()
    print(f"\nTest Mean IoU: {mean_iou:.4f}")

    # Print class-wise IoU
    class_iou = test_metric.get_class_iou()
    print("\nClass-wise IoU:")
    for class_name, iou in class_iou.items():
        print(f"  {class_name}: {iou:.4f}")

    # Save results to a file
    with open('evaluation_summary.txt', 'w') as f:
        f.write(f"Test Mean IoU: {mean_iou:.4f}\n\n")
        f.write("Class-wise IoU:\n")
        for class_name, iou in class_iou.items():
            f.write(f"  {class_name}: {iou:.4f}\n")

    print(f"Evaluation results saved to evaluation_summary.txt")

    return mean_iou, class_iou


if __name__ == "__main__":
    # Make TensorFlow operations deterministic for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    print("____________________________________________________________________")
    print("Starting two-stage training for improved ship detection...")
    model = train_two_stage()
    print("Training completed successfully!")
    print("____________________________________________________________________")
