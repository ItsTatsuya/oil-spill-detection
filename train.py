import os
import numpy as np
import matplotlib.pyplot as plt
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

def train_and_evaluate():
    NUM_CLASSES = 5
    BATCH_SIZE = 2  # Reduced batch size for memory efficiency with augmentations
    EPOCHS = 800
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
    augmented_train_ds = apply_augmentation(train_ds, batch_size=BATCH_SIZE)

    # Debug dataset shapes
    for images, labels in augmented_train_ds.take(1):
        print(f"Augmented train batch - Images shape: {images.shape}, dtype: {images.dtype}")
        print(f"Augmented train batch - Labels shape: {labels.shape}, dtype: {labels.dtype}")

    # Add extra validation to ensure model compatibility
    @tf.function
    def ensure_shapes(images, labels):
        """Ensure images and labels have correct shapes for model input."""
        # Get rank and shape dynamically
        images_rank = tf.rank(images)
        labels_rank = tf.rank(labels)
        images_shape = tf.shape(images)
        labels_shape = tf.shape(labels)

        # Define expected shape
        expected_height = 384
        expected_width = 384
        expected_channels = 1

        # Handle rank 4 or 5 tensors
        def reshape_if_rank_5(tensor, tensor_name):
            rank = tf.rank(tensor)
            shape = tf.shape(tensor)
            return tf.cond(
                tf.equal(rank, 5),
                lambda: tf.reshape(tensor, [shape[0] * shape[1], expected_height, expected_width, expected_channels]),
                lambda: tensor
            )

        images = reshape_if_rank_5(images, "images")
        labels = reshape_if_rank_5(labels, "labels")

        # Verify rank 4 and channels
        tf.assert_equal(tf.rank(images), 4, message="Images must be rank 4 after reshaping")
        tf.assert_equal(tf.rank(labels), 4, message="Labels must be rank 4 after reshaping")
        tf.assert_equal(images_shape[-1], expected_channels, message="Images must have 1 channel")
        tf.assert_equal(labels_shape[-1], expected_channels, message="Labels must have 1 channel")

        # Ensure height and width
        tf.assert_equal(images_shape[1], expected_height, message="Images height must be 384")
        tf.assert_equal(images_shape[2], expected_width, message="Images width must be 384")
        tf.assert_equal(labels_shape[1], expected_height, message="Labels height must be 384")
        tf.assert_equal(labels_shape[2], expected_width, message="Labels width must be 384")

        return images, labels
    # Apply shape validation to both datasets
    augmented_train_ds = augmented_train_ds.map(ensure_shapes)
    test_ds = test_ds.map(ensure_shapes)

    print(f"Dataset loaded. Training steps: {train_steps}, Validation steps: {val_steps}")
    if train_steps == 0 or val_steps == 0:
        raise ValueError("Empty dataset detected. Check data loading pipeline.")

    pretrained_weights_path = 'pretrained_weights/segformer_b2_pretrain.weights.h5'
    use_pretrained = os.path.exists(pretrained_weights_path)
    print(f"Pre-trained weights {'found' if use_pretrained else 'not found'} at {pretrained_weights_path}")

    from model import OilSpillSegformer
    from loss import HybridSegmentationLoss
    model = OilSpillSegformer(
        input_shape=(*IMG_SIZE, 1),
        num_classes=NUM_CLASSES,
        drop_rate=0.1,
        use_cbam=False,  # Disable CBAM to reduce memory usage
        pretrained_weights=pretrained_weights_path if use_pretrained else None
    )

    if os.path.exists(latest_weights):
        print(f"Loading checkpoint from {latest_weights}...")
        try:
            model.load_weights(latest_weights, by_name=True, skip_mismatch=True)
            checkpoint_info_path = 'checkpoints/checkpoint_info.txt'
            if os.path.exists(checkpoint_info_path):
                with open(checkpoint_info_path, 'r') as f:
                    initial_epoch = int(f.read().strip())
                print(f"Resuming from epoch {initial_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Continuing with pre-trained weights or from scratch")
            initial_epoch = 0
    else:
        print("No checkpoint found. Using pre-trained weights or training from scratch.")

    lr_schedule = create_cosine_decay_with_warmup(
        epochs=EPOCHS,
        train_steps=train_steps,
        batch_size=BATCH_SIZE,
        initial_lr=LEARNING_RATE,
        warmup_epochs=10
    )

    # Check TensorFlow/Keras version and use appropriate optimizer
    # Detect if we're using Keras 3 (which has built-in weight decay for Adam)
    using_keras3 = hasattr(tf.keras, '__version__') and tf.keras.__version__.startswith('3')
    print(f"Using Keras version: {tf.keras.__version__}")

    # Create optimizer based on Keras version
    if using_keras3:
        # For Keras 3, use Adam with built-in weight decay parameter
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,  # Increased for better numerical stability
            amsgrad=True,  # Enable AMSGrad for more stable training
            weight_decay=1e-4,  # Built-in weight decay in Keras 3
            clipnorm=1.0,  # Gradient clipping to prevent exploding gradients
            name="adam_optimizer"
        )
        print("Using Keras 3 Adam optimizer with built-in weight decay")
        use_weight_decay_callback = False
    else:
        # For older Keras versions, fall back to AdamW or Adam + custom weight decay callback
        try:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=1e-4,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                clipnorm=1.0
            )
            use_weight_decay_callback = False
            print("Using AdamW optimizer")
        except (ImportError, AttributeError):
            # Fall back to standard Adam + weight decay callback
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                clipnorm=1.0
            )
            use_weight_decay_callback = True
            print("Using Adam optimizer with weight decay callback")

    # Keep the weight decay callback definition for compatibility with older Keras versions
    weight_decay_rate = 1e-4

    class WeightDecayCallback(tf.keras.callbacks.Callback):
        def __init__(self, weight_decay=1e-4):
            super().__init__()
            self.weight_decay = weight_decay

        def on_train_batch_begin(self, batch, logs=None):
            # Apply weight decay to all trainable weights except biases and normalization layers
            for var in self.model.trainable_weights:
                # Skip biases, normalization layers, and embeddings
                if (len(var.shape) <= 1 or
                    'bias' in var.name or
                    'gamma' in var.name or
                    'beta' in var.name or
                    'norm' in var.name.lower() or
                    'embedding' in var.name.lower()):
                    continue

                # Get the current learning rate
                lr = self.model.optimizer.lr
                if hasattr(lr, '__call__'):
                    lr = lr(self.model.optimizer.iterations)

                # Apply weight decay using assign_sub
                var.assign_sub(
                    self.weight_decay * lr * var,
                    use_locking=True
                )

    print(f"Using {'standard Adam with weight decay callback' if use_weight_decay_callback else 'AdamW optimizer'}")

    # Use the improved hybrid segmentation loss for better performance on oil spill detection
    class_weights_list = [class_weights_dict[i] for i in range(NUM_CLASSES)]
    loss_fn = HybridSegmentationLoss(
        class_weights=class_weights_list,
        ce_weight=0.4,
        focal_weight=0.3,
        dice_weight=0.3,
        boundary_weight=0.0,  # Temporarily disabled until we can fix the boundary loss component
        gamma=3.0,
        from_logits=True
    )
    print("Using fixed HybridSegmentationLoss for better oil spill and ship detection")

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[IoUMetric(num_classes=NUM_CLASSES)],
        run_eagerly=False,  # Disable eager execution for improved performance
        jit_compile=True    # Enable XLA JIT compilation for faster training now that loss function is compatible
    )

    print("Model compiled with XLA JIT compilation enabled for faster training with the XLA-compatible loss function")
    model.summary()

    checkpoint_path = 'checkpoints/segformer_b2_best.weights.h5'
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_iou_metric',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=latest_weights,
            save_weights_only=True,
            save_best_only=False,
            save_freq='epoch',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_iou_metric',
            patience=100,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs/fit/segformer_b2_' + datetime.now().strftime("%Y%m%d-%H%M%S"),
            update_freq='epoch'
        ),
        ProgressCallback(total_epochs=EPOCHS, steps_per_epoch=train_steps, validation_steps=val_steps),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: open('checkpoints/checkpoint_info.txt', 'w').write(str(epoch + 1))
        ),
        # Learning rate reporting callback that works with Keras 3
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(f"Current learning rate: {_get_current_lr(optimizer):.2e}")
        )
    ]

    # Add weight decay callback if using standard Adam
    if use_weight_decay_callback:
        callbacks.append(WeightDecayCallback(weight_decay=weight_decay_rate))

    print(f"Starting SegFormer-B2 training for {EPOCHS} epochs from epoch {initial_epoch}...")
    history = model.fit(
        augmented_train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0,
        initial_epoch=initial_epoch
    )

    final_weights_path = 'segformer_b2_final.weights.h5'
    model.save_weights(final_weights_path)
    print(f"Model weights saved as '{final_weights_path}'")

    plot_training_curves(history, save_path='segformer_b2_miou_curves.png')

    print("\nEvaluating with multi-scale prediction...")
    eval_batch_size = 2
    multi_scale_predictor = MultiScalePredictor(model, scales=[0.75, 1.0, 1.25], batch_size=eval_batch_size)
    test_metric = IoUMetric(num_classes=NUM_CLASSES)
    eval_start_time = time.time()

    for images, labels in tqdm(test_ds, desc="Evaluating"):
        predictions = multi_scale_predictor.predict(images)
        test_metric.update_state(labels, predictions)

    eval_time = time.time() - eval_start_time
    minutes, seconds = divmod(eval_time, 60)
    print(f"\nEvaluation completed in {int(minutes)}m {seconds:.1f}s")
    mean_iou = test_metric.result().numpy()
    print(f"Test Mean IoU: {mean_iou:.4f}")
    class_iou = test_metric.get_class_iou()
    print("\nClass-wise IoU:")
    for class_name, iou in class_iou.items():
        print(f"  {class_name}: {iou:.4f}")

    return history

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    print("____________________________________________________________________")
    print("Starting SegFormer-B2 training for oil spill detection...")
    history = train_and_evaluate()
    print("SegFormer-B2 training completed successfully")
    print("____________________________________________________________________")
