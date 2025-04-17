"""
Oil Spill Detection - Training Script

This script trains an improved DeepLabv3+ model on the oil spill detection dataset.
Features:
1. Multi-scale training (50%, 75%, 100% of 320x320) with prediction fusion
2. Data augmentation for training
3. Hybrid loss function
4. EfficientNet-B4 backbone with enhanced decoder

The model is trained for 600 epochs with batch size 8, learning rate 5e-5, and Adam optimizer.
Model performance is evaluated using mIoU and class-wise IoU for the 5 classes.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Set TensorFlow logging level to reduce verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Import custom modules
from data_loader import load_dataset
from augmentation import apply_augmentation
from loss import hybrid_loss
from model import DeepLabv3Plus


# Configure GPU memory growth to avoid OOM errors
def configure_gpu():
    """Configure GPU memory settings to avoid memory allocation issues."""
    # First, check for GPU devices
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Only use GPU 0 if available and more than one GPU exists
            # For multiple GPUs, can be configured to use more
            if len(gpus) > 0:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Configure memory growth
            for gpu in tf.config.experimental.get_visible_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"Available GPUs: {len(logical_gpus)}")

            # Set memory limit if needed for older GPU versions
            # Uncomment the following line if you're having OOM issues
            # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

            return True
        except RuntimeError as e:
            # Handle errors
            print(f"GPU configuration error: {e}")
            print("Using CPU instead")
            return False
    else:
        print("No GPU found. Using CPU.")
        return False


class MultiScalePredictor:
    """
    Multi-scale prediction with fusion for semantic segmentation.
    Accepts images at multiple scales and combines predictions.
    """
    def __init__(self, model, scales=[0.5, 0.75, 1.0]):
        self.model = model
        self.scales = scales

    def predict(self, image_batch):
        """
        Predict using multiple scales and fuse results.

        Args:
            image_batch: Batch of images with shape [batch_size, height, width, channels]

        Returns:
            Fused predictions with shape [batch_size, height, width, num_classes]
        """
        batch_size = tf.shape(image_batch)[0]
        height = tf.shape(image_batch)[1]
        width = tf.shape(image_batch)[2]

        # Placeholder for all scaled predictions
        all_predictions = []

        # Make predictions at each scale
        for scale in self.scales:
            # Resize images to current scale
            scaled_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
            scaled_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)

            # Ensure dimensions are multiples of 8 for better performance
            scaled_height = tf.cast(tf.math.ceil(scaled_height / 8) * 8, tf.int32)
            scaled_width = tf.cast(tf.math.ceil(scaled_width / 8) * 8, tf.int32)

            scaled_images = tf.image.resize(
                image_batch,
                size=(scaled_height, scaled_width),
                method='bilinear'
            )

            # Get predictions for scaled images
            scaled_preds = self.model(scaled_images, training=False)

            # Resize predictions back to original size
            resized_preds = tf.image.resize(
                scaled_preds,
                size=(height, width),
                method='bilinear'
            )

            # Apply softmax to get probabilities
            probs = tf.nn.softmax(resized_preds, axis=-1)

            all_predictions.append(probs)

        # Average predictions from all scales
        fused_prediction = tf.reduce_mean(tf.stack(all_predictions, axis=0), axis=0)

        # Convert back to logits for compatibility with loss functions
        epsilon = 1e-7
        fused_prediction = tf.clip_by_value(fused_prediction, epsilon, 1 - epsilon)
        fused_logits = tf.math.log(fused_prediction / (1 - fused_prediction + epsilon))

        return fused_logits


class IoUMetric(tf.keras.metrics.Metric):
    """
    Custom IoU metric for semantic segmentation.
    Calculates mean IoU and class-wise IoU for evaluation.
    """
    def __init__(self, num_classes=5, name='iou_metric', **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.mean_iou = tf.keras.metrics.MeanIoU(num_classes=num_classes)
        self.class_names = ['Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land']

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get shapes
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)

        # Use tf.cond instead of Python if statement for shape checking
        y_pred = tf.cond(
            tf.logical_or(
                tf.not_equal(y_true_shape[1], y_pred_shape[1]),
                tf.not_equal(y_true_shape[2], y_pred_shape[2])
            ),
            lambda: tf.image.resize(y_pred, [y_true_shape[1], y_true_shape[2]], method='bilinear'),
            lambda: y_pred
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
    """Create training callbacks."""
    # Create checkpoint directory if it doesn't exist
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
            patience=30,
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
    """Plot and save training/validation IoU curves."""
    plt.figure(figsize=(12, 5))

    # Plot IoU metric
    plt.subplot(1, 2, 1)
    plt.plot(history.history['iou_metric'], label='Training IoU')
    plt.plot(history.history['val_iou_metric'], label='Validation IoU')
    plt.title('Mean IoU Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")


def train_and_evaluate():
    """Train and evaluate the DeepLabv3+ model on the oil spill detection dataset."""
    # Define constants
    NUM_CLASSES = 5
    BATCH_SIZE = 4
    EPOCHS = 1
    LEARNING_RATE = 5e-5
    IMG_SIZE = (320, 320)

    # Create directory for checkpoints if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    # Path for latest checkpoint
    latest_checkpoint = 'checkpoints/latest_model.h5'
    initial_epoch = 0

    print("Loading datasets...")
    # Load training and validation datasets
    train_ds = load_dataset(data_dir='dataset', split='train')
    test_ds = load_dataset(data_dir='dataset', split='test')

    # Apply data augmentation to training dataset
    print("Applying data augmentation...")
    # Fix: Don't unbatch the dataset since apply_augmentation will batch it again
    train_ds = apply_augmentation(train_ds, batch_size=BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Create the DeepLabv3+ model
    print("Creating model...")
    model = DeepLabv3Plus(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)

    # Check if checkpoint exists to resume training
    if os.path.exists(latest_checkpoint):
        print(f"Found checkpoint at {latest_checkpoint}. Resuming training...")
        try:
            # Load model weights
            model = tf.keras.models.load_model(
                latest_checkpoint,
                custom_objects={'hybrid_loss': hybrid_loss, 'IoUMetric': IoUMetric}
            )
            # Get the initial epoch from the model name (if saved with epoch in name)
            # or you might need to store this in a separate file
            checkpoint_info_path = 'checkpoints/checkpoint_info.txt'
            if os.path.exists(checkpoint_info_path):
                with open(checkpoint_info_path, 'r') as f:
                    initial_epoch = int(f.read().strip())
                print(f"Resuming from epoch {initial_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training from scratch instead.")
            initial_epoch = 0

    # Define optimizer with gradient clipping to prevent exploding gradients
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0  # Add gradient clipping to prevent extreme weight updates
    )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=hybrid_loss,
        metrics=[IoUMetric(num_classes=NUM_CLASSES)]
    )

    # Set up callbacks
    checkpoint_path = 'checkpoints/improved_deeplabv3plus_best.h5'
    callbacks = create_callbacks(checkpoint_path)

    # Add custom callback to save epoch information
    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            with open('checkpoints/checkpoint_info.txt', 'w') as f:
                f.write(str(epoch + 1))  # +1 because next epoch will be epoch+1

    callbacks.append(EpochLogger())

    # Train the model with validation data
    print(f"Starting training for {EPOCHS} epochs from epoch {initial_epoch}...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=initial_epoch  # Start from this epoch
    )

    # Save the final model
    model.save('improved_deeplabv3plus.h5')
    print("Model saved as 'improved_deeplabv3plus.h5'")

    # Plot and save training curves
    plot_training_curves(history)

    # Evaluate on test set using multi-scale prediction
    print("\nEvaluating with multi-scale prediction...")
    multi_scale_predictor = MultiScalePredictor(model, scales=[0.5, 0.75, 1.0])

    # Initialize IoU metric for evaluation
    test_metric = IoUMetric(num_classes=NUM_CLASSES)

    # Perform evaluation
    for images, labels in tqdm(test_ds):
        # Get multi-scale predictions
        predictions = multi_scale_predictor.predict(images)

        # Update metrics
        test_metric.update_state(labels, predictions)

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

    # Configure GPU
    gpu_available = configure_gpu()

    # Don't use mixed precision as it causes issues with the model architecture
    # TensorFlow is trying to mix float32 and float16 in ways not supported in this model
    print("Using default precision (float32)")

    # Train and evaluate the model
    train_and_evaluate()
