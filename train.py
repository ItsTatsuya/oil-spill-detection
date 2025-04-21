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
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"Using GPU: {gpus[0].name}")
else:
    print("No GPU found, using CPU")

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
    def __init__(self, model, scales=[0.5, 0.75, 1.0], batch_size=1):
        self.model = model
        self.scales = scales
        # Get the expected input shape from the model
        self.expected_height = model.input_shape[1]
        self.expected_width = model.input_shape[2]
        # Use smaller batch size for prediction to reduce memory usage
        self.batch_size = batch_size

    def predict(self, image_batch):
        import gc  # Import garbage collector

        original_batch_size = tf.shape(image_batch)[0]
        height = tf.shape(image_batch)[1]
        width = tf.shape(image_batch)[2]
        num_classes = self.model.output_shape[-1]

        # Create a placeholder for the final prediction with the correct shape
        # This avoids storing all scale predictions in memory at once
        final_prediction = tf.zeros([original_batch_size, height, width, num_classes],
                                   dtype=tf.float32)

        # Process each image in the batch individually to reduce memory usage
        for i in range(original_batch_size):
            # Extract single image and add batch dimension back
            single_image = tf.expand_dims(image_batch[i], axis=0)

            # Initialize accumulator for this single image
            accumulated_pred = tf.zeros([1, height, width, num_classes], dtype=tf.float32)

            # Process each scale separately and accumulate results
            for scale in self.scales:
                # Explicitly release memory
                tf.keras.backend.clear_session()
                gc.collect()

                # Resize images to current scale while preserving aspect ratio
                scaled_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
                scaled_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)

                # Ensure dimensions are multiples of 8 for better performance
                scaled_height = tf.cast(tf.math.ceil(scaled_height / 8) * 8, tf.int32)
                scaled_width = tf.cast(tf.math.ceil(scaled_width / 8) * 8, tf.int32)

                # Resize to the scaled dimensions
                scaled_image = tf.image.resize(
                    single_image,
                    size=(scaled_height, scaled_width),
                    method='bilinear'
                )

                # Resize to model's expected input shape
                model_input = tf.image.resize(
                    scaled_image,
                    size=(self.expected_height, self.expected_width),
                    method='bilinear'
                )

                # Get predictions
                with tf.device('/CPU:0'):  # Use CPU for post-processing to free GPU memory
                    logits = self.model(model_input, training=False)

                    # Resize predictions back to original size
                    resized_preds = tf.image.resize(
                        logits,
                        size=(height, width),
                        method='bilinear'
                    )

                    # Apply softmax to get probabilities
                    probs = tf.nn.softmax(resized_preds, axis=-1)

                    # Accumulate probabilities (no need to keep all scales in memory)
                    accumulated_pred += probs

                # Explicitly delete tensors to free memory
                del scaled_image, model_input, logits, resized_preds, probs
                gc.collect()

            # Average the accumulated predictions for this image
            average_pred = accumulated_pred / len(self.scales)

            # Update the corresponding slice of the final prediction tensor
            final_prediction = tf.tensor_scatter_nd_update(
                final_prediction,
                indices=[[i]],
                updates=average_pred
            )

            # Clean up
            del single_image, accumulated_pred, average_pred
            gc.collect()

        # Convert averaged predictions back to logits for compatibility with loss functions
        epsilon = 1e-7
        final_prediction = tf.clip_by_value(final_prediction, epsilon, 1 - epsilon)
        final_logits = tf.math.log(final_prediction / (1 - final_prediction + epsilon))

        return final_logits


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


def create_callbacks(checkpoint_path, save_best_only=True):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    callbacks = [
        # Model checkpoint to save the best model based on validation IoU
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_iou_metric',
            save_best_only=save_best_only,
            mode='max',
            verbose=1
        ),
        # Also save a checkpoint after each epoch to enable resuming training
        tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/latest_model.h5',
            save_weights_only=False,
            save_best_only=False,
            save_freq='epoch',
            verbose=1
        ),
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_iou_metric',
            patience=60,  # Increased from 30 to 60 to allow more time for model improvement
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate scheduler
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_iou_metric',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
            update_freq='epoch'
        )
    ]

    return callbacks


def plot_training_curves(history, save_path='miou_curves.png'):
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

    # Find the IoU metric keys (may be 'iou_metric' or just 'iou')
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
    train_loss_key = 'loss' if 'loss' in available_keys else None
    val_loss_key = 'val_loss' if 'val_loss' in available_keys else None

    # Plot Loss if available
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


def train_and_evaluate():
    # Define constants
    NUM_CLASSES = 5
    BATCH_SIZE = 8
    EPOCHS = 600
    LEARNING_RATE = 5e-5
    IMG_SIZE = (320, 320)

    # Create directory for checkpoints if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    # Path for latest checkpoint - using correct .weights.h5 extension for weights-only files
    latest_weights = 'checkpoints/latest_model.weights.h5'
    initial_epoch = 0

    print("Loading datasets...")
    # Load training and validation datasets with performance optimization
    train_ds, class_weights_dict, train_steps = load_dataset(data_dir='dataset', split='train', batch_size=BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)  # Add prefetch for better performance

    test_ds, _, val_steps = load_dataset(data_dir='dataset', split='test', batch_size=BATCH_SIZE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)  # Add prefetch for validation dataset

    # Apply advanced data augmentation from augmentation.py
    print("Applying advanced data augmentation...")
    augmented_train_ds = apply_augmentation(train_ds)

    print(f"Dataset loaded. Training steps: {train_steps}, Validation steps: {val_steps}")
    if train_steps == 0:
        raise ValueError("Training dataset is empty! Please check your data loading pipeline.")

    if val_steps == 0:
        raise ValueError("Validation dataset is empty! Please check your data loading pipeline.")

    # Create the DeepLabv3+ model
    print("Creating model...")
    model = DeepLabv3Plus(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)

    # Check if checkpoint exists to resume training
    if os.path.exists(latest_weights):
        print(f"Found checkpoint at {latest_weights}. Loading weights...")
        try:
            # Load model weights only, not the full architecture
            model.load_weights(latest_weights)

            # Get the initial epoch from checkpoint info
            checkpoint_info_path = 'checkpoints/checkpoint_info.txt'
            if os.path.exists(checkpoint_info_path):
                with open(checkpoint_info_path, 'r') as f:
                    initial_epoch = int(f.read().strip())
                print(f"Resuming from epoch {initial_epoch}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Training from scratch instead.")
            initial_epoch = 0
    else:
        print("No checkpoint found. Training from scratch.")

    # Define optimizer with gradient clipping to prevent exploding gradients
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0,  # Add gradient clipping to prevent extreme weight updates
        epsilon=1e-8,  # Increase epsilon to improve numerical stability
        beta_1=0.9,    # Default Adam momentum parameters
        beta_2=0.999   # Default Adam RMSprop parameter
    )

    # Convert class weights dictionary to a list ordered by class index
    class_weights_list = [class_weights_dict.get(i, 1.0) for i in range(NUM_CLASSES)]
    print(f"Using class weights from data loader: {class_weights_list}")

    # Create the loss function with the calculated class weights
    loss_fn = HybridSegmentationLoss(
        class_weights=class_weights_list,
        ce_weight=0.4,
        focal_weight=0.3,
        dice_weight=0.3
    )

    # Use standard float32 precision for training instead of mixed precision
    # This helps avoid issues with mixed data types during ReluGrad operations
    print("Setting standard float32 precision for training stability")
    tf.keras.mixed_precision.set_global_policy('float32')

    # Compile model with XLA optimization enabled
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[IoUMetric(num_classes=NUM_CLASSES)],
        run_eagerly=False,
        jit_compile=True,  # Enable XLA JIT compilation
    )

    print("Model compiled with XLA JIT compilation enabled for training")

    # Set up callbacks for saving weights only, not full model
    checkpoint_path = 'checkpoints/improved_deeplabv3plus_best.weights.h5'

    # Create and configure callbacks
    callbacks = [
        # Model checkpoint to save the best weights based on validation IoU
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_iou_metric',
            save_best_only=True,
            save_weights_only=True,  # Save weights only, not full model
            mode='max',
            verbose=1
        ),
        # Also save weights after each epoch to enable resuming training
        tf.keras.callbacks.ModelCheckpoint(
            filepath=latest_weights,
            save_weights_only=True,  # Save weights only, not full model
            save_best_only=False,
            save_freq='epoch',
            verbose=1
        ),
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_iou_metric',
            patience=60,  # Increased from 30 to 60 to allow more time for model improvement
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate scheduler
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_iou_metric',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
            update_freq='epoch'
        )
    ]

    # Add custom callback to save epoch information
    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            with open('checkpoints/checkpoint_info.txt', 'w') as f:
                f.write(str(epoch + 1))  # +1 because next epoch will be epoch+1

    callbacks.append(EpochLogger())

    # Add our custom progress callback
    progress_callback = ProgressCallback(
        total_epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=val_steps
    )
    callbacks.append(progress_callback)

    # Train the model with validation data
    print(f"Starting training for {EPOCHS} epochs from epoch {initial_epoch}...")
    history = model.fit(
        augmented_train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0,  # Set to 0 as we're using our custom progress logger
        initial_epoch=initial_epoch  # Start from this epoch
    )

    # Save the final weights
    final_weights_path = 'improved_deeplabv3plus.weights.h5'
    model.save_weights(final_weights_path)
    print(f"Model weights saved as '{final_weights_path}'")

    # Plot and save training curves
    plot_training_curves(history)

    # Evaluate on test set using multi-scale prediction with smaller evaluation batch size
    print("\nEvaluating with memory-efficient multi-scale prediction...")
    # Use batch size of 1 for evaluation to minimize memory usage
    multi_scale_predictor = MultiScalePredictor(model, scales=[0.5, 0.75, 1.0], batch_size=1)

    # Initialize IoU metric for evaluation
    test_metric = IoUMetric(num_classes=NUM_CLASSES)

    # Import gc for explicit garbage collection during evaluation
    import gc

    # Perform evaluation in smaller chunks to reduce memory pressure
    for images, labels in tqdm(test_ds):
        # Get multi-scale predictions
        predictions = multi_scale_predictor.predict(images)

        # Update metrics
        test_metric.update_state(labels, predictions)

        # Force garbage collection after each batch to free memory
        tf.keras.backend.clear_session()
        gc.collect()

    # Print evaluation results
    mean_iou = test_metric.result().numpy()
    print(f"\nTest Mean IoU: {mean_iou:.4f}")

    # Print class-wise IoU
    class_iou = test_metric.get_class_iou()
    print("\nClass-wise IoU:")
    for class_name, iou in class_iou.items():
        print(f"  {class_name}: {iou:.4f}")

    return history


if __name__ == "__main__":
    # Make TensorFlow operations deterministic for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    print("____________________________________________________________________")
    print("Starting model training with mixed precision...")
    history = train_and_evaluate()
    print("Training completed successfully with mixed precision")
    print("____________________________________________________________________")
