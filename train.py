"""
Training script for SegFormer-B2 oil spill detection.

Key improvements over the original:
- 8x A100 distributed training via MirroredStrategy (global batch 256)
- Proper train/val split (test set reserved for final evaluation only)
- Learning rate scaled via sqrt rule for large-batch training
- EMA of weights for better generalisation
- ReduceLROnPlateau alongside cosine schedule
- Re-enabled boundary loss, label smoothing, corrected focal weighting
- Lower focal gamma (2.0 vs 3.0), lower cosine alpha (0.001 vs 0.01)
- Thread parallelism auto-detected instead of hardcoded to 2
- Full reproducibility (PYTHONHASHSEED, TF_DETERMINISTIC_OPS, random.seed)
"""

import os
import gc
import time
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import silent_tf_import, set_reproducibility, colored_print, TermColors
from config import TrainConfig, LossConfig, DataConfig

# ---- Reproducibility (before any TF import) --------------------------------
cfg = TrainConfig()
set_reproducibility(cfg.seed)

tf = silent_tf_import()
from tensorflow.keras import mixed_precision  # type: ignore

logger = logging.getLogger('oil_spill')

# ---- Mixed precision --------------------------------------------------------
mixed_precision.set_global_policy('mixed_float16')
print(f"Mixed precision policy: {mixed_precision.global_policy().name}")

from data.data_loader import load_dataset
from data.augmentation import apply_augmentation


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, steps_per_epoch, validation_steps=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.step_times = []
        self.epoch_start_time = None

    def on_train_begin(self, logs=None):
        print("\n" + "=" * 80)
        print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total epochs: {self.total_epochs}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        if self.validation_steps:
            print(f"Validation steps: {self.validation_steps}")
        print("=" * 80 + "\n")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.step_times = []
        print(f"\nEpoch {epoch + 1}/{self.total_epochs} - "
              f"Starting at {datetime.now().strftime('%H:%M:%S')}")
        self.last_step_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        now = time.time()
        self.step_times.append(now - self.last_step_time)
        self.last_step_time = now
        report_every = max(1, self.steps_per_epoch // 20)
        if batch % report_every == 0 or batch == self.steps_per_epoch - 1:
            loss = logs.get('loss', 0.0)
            iou = logs.get('iou_metric', 0.0)
            avg_ms = np.mean(self.step_times[-50:]) * 1000 if self.step_times else 0
            eta_s = avg_ms / 1000 * (self.steps_per_epoch - batch) if batch > 0 else 0
            eta = f"{int(eta_s // 60):02d}:{int(eta_s % 60):02d}" if batch > 0 else "??:??"
            print(f"  Step {batch + 1}/{self.steps_per_epoch} - "
                  f"Loss: {loss:.4f} - IoU: {iou:.4f} - "
                  f"{avg_ms:.1f} ms/step - ETA: {eta}")

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        val_loss = logs.get('val_loss', 0.0)
        val_iou = logs.get('val_iou_metric', 0.0)
        print(f"\nEpoch {epoch + 1}/{self.total_epochs} "
              f"completed in {int(h):02d}:{int(m):02d}:{int(s):02d}")
        print(f"  Loss: {logs.get('loss', 0.0):.4f} - "
              f"IoU: {logs.get('iou_metric', 0.0):.4f} - "
              f"Val Loss: {val_loss:.4f} - Val IoU: {val_iou:.4f}")
        if hasattr(self.model.optimizer, 'lr'):
            lr = self.model.optimizer.lr
            if hasattr(lr, 'numpy'):
                lr_val = float(lr.numpy())
            elif callable(getattr(lr, '__call__', None)):
                lr_val = float(lr(self.model.optimizer.iterations).numpy())
            else:
                lr_val = float(lr)
            print(f"  Learning rate: {lr_val:.2e}")
        print("=" * 80)


class EMACallback(tf.keras.callbacks.Callback):
    """Exponential Moving Average of model weights — swap in during validation."""

    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema_weights = None
        self.backup_weights = None

    def on_train_batch_end(self, batch, logs=None):
        if self.ema_weights is None:
            self.ema_weights = [tf.Variable(w) for w in self.model.trainable_weights]
        for ema_w, w in zip(self.ema_weights, self.model.trainable_weights):
            ema_w.assign(self.decay * ema_w + (1.0 - self.decay) * w)

    def on_test_begin(self, logs=None):
        if self.ema_weights is None:
            return
        self.backup_weights = [tf.identity(w) for w in self.model.trainable_weights]
        for w, ema_w in zip(self.model.trainable_weights, self.ema_weights):
            w.assign(ema_w)

    def on_test_end(self, logs=None):
        if self.backup_weights is None:
            return
        for w, bak in zip(self.model.trainable_weights, self.backup_weights):
            w.assign(bak)
        self.backup_weights = None

    def apply_ema_weights(self):
        """Permanently replace model weights with EMA weights (call after training)."""
        if self.ema_weights is not None:
            for w, ema_w in zip(self.model.trainable_weights, self.ema_weights):
                w.assign(ema_w)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def create_cosine_decay_with_warmup(
    epochs, train_steps, initial_lr=3.4e-4, warmup_epochs=25, alpha=0.001
):
    total_steps = epochs * train_steps
    warmup_steps = warmup_epochs * train_steps

    warmup_lr = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.0,
        decay_steps=max(warmup_steps, 1),
        end_learning_rate=initial_lr,
    )

    if total_steps <= warmup_steps:
        class FallbackLR(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __call__(self, step):
                return tf.keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=initial_lr,
                    decay_steps=total_steps,
                    end_learning_rate=1e-6,
                )(step)
            def get_config(self):
                return {"initial_lr": initial_lr, "total_steps": total_steps}
        return FallbackLR()

    cosine_lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=alpha,
    )

    class CosineWarmupLR(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            return tf.cond(
                step < warmup_steps,
                lambda: warmup_lr(step),
                lambda: cosine_lr(step - warmup_steps),
            )

        def get_config(self):
            return {
                "epochs": epochs,
                "train_steps": train_steps,
                "initial_lr": initial_lr,
                "warmup_epochs": warmup_epochs,
                "alpha": alpha,
            }

    return CosineWarmupLR()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_curves(history, save_path='miou_curves.png'):
    if not history.history:
        print("Warning: Empty training history.")
        return

    plt.figure(figsize=(12, 5))
    keys = list(history.history.keys())

    t_iou = next((k for k in keys if 'iou' in k.lower() and not k.startswith('val_')), None)
    v_iou = next((k for k in keys if 'iou' in k.lower() and k.startswith('val_')), None)

    if t_iou and v_iou:
        plt.subplot(1, 2, 1)
        plt.plot(history.history[t_iou], label=f'Train {t_iou}')
        plt.plot(history.history[v_iou], label=f'Val {v_iou}')
        plt.title('Mean IoU Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Mean IoU')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.6)

    if 'loss' in keys and 'val_loss' in keys:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")


# ---------------------------------------------------------------------------
# Main: train_and_evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate():
    # ---- Config -------------------------------------------------------------
    NUM_CLASSES = cfg.num_classes
    BATCH_SIZE = cfg.batch_size
    PER_GPU_BS = cfg.per_gpu_batch_size
    EPOCHS = cfg.epochs
    LEARNING_RATE = cfg.learning_rate
    IMG_SIZE = (384, 384)

    tf.keras.backend.clear_session()
    gc.collect()

    # ---- GPU setup ----------------------------------------------------------
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        print(f"Found {len(gpus)} GPU(s), memory growth enabled")
    else:
        print("No GPU found — using CPU")

    # Thread parallelism: auto-detect (0 = let TF decide)
    if cfg.inter_op_threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(cfg.inter_op_threads)
    if cfg.intra_op_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(cfg.intra_op_threads)

    # ---- Distribution strategy ----------------------------------------------
    if cfg.use_distributed and len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"MirroredStrategy: {strategy.num_replicas_in_sync} replicas")
        BATCH_SIZE = PER_GPU_BS * strategy.num_replicas_in_sync
    else:
        strategy = tf.distribute.get_strategy()  # default (single device)
        if len(gpus) == 1:
            BATCH_SIZE = PER_GPU_BS
        print(f"Single-device strategy, batch_size={BATCH_SIZE}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    latest_weights = os.path.join(cfg.checkpoint_dir, 'segformer_b2_latest.weights.h5')
    initial_epoch = 0

    # ---- Data ---------------------------------------------------------------
    print("Loading datasets …")
    dcfg = DataConfig()
    train_ds, class_weights_dict, train_steps = load_dataset(
        data_dir=dcfg.data_dir, split='train', batch_size=BATCH_SIZE,
        val_split=dcfg.val_split, class_weights_cache=dcfg.class_weights_cache,
    )
    val_ds, _, val_steps = load_dataset(
        data_dir=dcfg.data_dir, split='val', batch_size=BATCH_SIZE,
        val_split=dcfg.val_split,
    )

    # Debug shapes
    for images, labels in train_ds.take(1):
        print(f"Train batch — images: {images.shape} ({images.dtype}), labels: {labels.shape}")
    for images, labels in val_ds.take(1):
        print(f"Val batch   — images: {images.shape} ({images.dtype}), labels: {labels.shape}")

    colored_print("Applying augmentation pipeline …", color=TermColors.YELLOW, bold=True)
    augmented_train_ds = apply_augmentation(train_ds, batch_size=BATCH_SIZE)

    for images, labels in augmented_train_ds.take(1):
        print(f"Augmented   — images: {images.shape} ({images.dtype}), labels: {labels.shape}")

    # Shape validation
    @tf.function
    def ensure_shapes(images, labels):
        expected_h, expected_w, expected_c = 384, 384, 1

        def reshape5(tensor):
            rank = tf.rank(tensor)
            shape = tf.shape(tensor)
            return tf.cond(
                tf.equal(rank, 5),
                lambda: tf.reshape(tensor, [shape[0] * shape[1], expected_h, expected_w, expected_c]),
                lambda: tensor,
            )

        images = reshape5(images)
        labels = reshape5(labels)
        return images, labels

    augmented_train_ds = augmented_train_ds.map(ensure_shapes)
    val_ds = val_ds.map(ensure_shapes)

    print(f"Train steps: {train_steps}, Val steps: {val_steps}")
    if train_steps == 0 or val_steps == 0:
        raise ValueError("Empty dataset detected — check data loading pipeline.")

    # ---- Model (inside strategy scope) --------------------------------------
    pretrained_path = cfg.pretrained_weights
    use_pretrained = os.path.exists(pretrained_path)
    print(f"Pre-trained weights {'found' if use_pretrained else 'not found'} at {pretrained_path}")

    with strategy.scope():
        from model.model import OilSpillSegformer
        from model.loss import HybridSegmentationLoss
        from model.metrics import IoUMetric

        model = OilSpillSegformer(
            input_shape=(*IMG_SIZE, 1),
            num_classes=NUM_CLASSES,
            drop_rate=0.1,
            use_cbam=False,
            pretrained_weights=pretrained_path if use_pretrained else None,
        )

        # ---- Attempt checkpoint resume --------------------------------------
        if os.path.exists(latest_weights):
            print(f"Loading checkpoint from {latest_weights} …")
            try:
                model.load_weights(latest_weights)
                info_path = os.path.join(cfg.checkpoint_dir, 'checkpoint_info.txt')
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        initial_epoch = int(f.read().strip())
                    print(f"Resuming from epoch {initial_epoch}")
            except Exception as e:
                print(f"Checkpoint loading failed: {e} — training from scratch")
                initial_epoch = 0
        else:
            print("No checkpoint found — starting from scratch / pre-trained weights.")

        # ---- LR schedule ----------------------------------------------------
        lr_schedule = create_cosine_decay_with_warmup(
            epochs=EPOCHS,
            train_steps=train_steps,
            initial_lr=LEARNING_RATE,
            warmup_epochs=cfg.warmup_epochs,
            alpha=cfg.cosine_alpha,
        )

        print(f"LR schedule: peak={LEARNING_RATE:.2e}, warmup={cfg.warmup_epochs} epochs, "
              f"alpha={cfg.cosine_alpha}")

        # ---- Optimizer -------------------------------------------------------
        using_keras3 = hasattr(tf.keras, '__version__') and tf.keras.__version__.startswith('3')
        if using_keras3:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=cfg.beta_1,
                beta_2=cfg.beta_2,
                epsilon=cfg.epsilon,
                amsgrad=cfg.amsgrad,
                weight_decay=cfg.weight_decay,
                clipnorm=cfg.clipnorm,
                name="adam_optimizer",
            )
            print("Keras 3 Adam with built-in weight decay")
        else:
            try:
                optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=lr_schedule,
                    weight_decay=cfg.weight_decay,
                    beta_1=cfg.beta_1,
                    beta_2=cfg.beta_2,
                    epsilon=cfg.epsilon,
                    clipnorm=cfg.clipnorm,
                )
                print("AdamW optimizer")
            except (ImportError, AttributeError):
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=lr_schedule,
                    beta_1=cfg.beta_1,
                    beta_2=cfg.beta_2,
                    epsilon=cfg.epsilon,
                    clipnorm=cfg.clipnorm,
                )
                print("Adam optimizer (no built-in weight decay)")

        # ---- Loss ------------------------------------------------------------
        lcfg = LossConfig()
        class_weights_list = [class_weights_dict[i] for i in range(NUM_CLASSES)]
        loss_fn = HybridSegmentationLoss(
            class_weights=class_weights_list,
            ce_weight=lcfg.ce_weight,
            focal_weight=lcfg.focal_weight,
            dice_weight=lcfg.dice_weight,
            boundary_weight=lcfg.boundary_weight,
            gamma=lcfg.focal_gamma,
            label_smoothing=lcfg.label_smoothing,
            from_logits=lcfg.from_logits,
        )
        print(f"HybridSegmentationLoss: CE={lcfg.ce_weight}, Focal={lcfg.focal_weight} (γ={lcfg.focal_gamma}), "
              f"Dice={lcfg.dice_weight}, Boundary={lcfg.boundary_weight}, "
              f"label_smoothing={lcfg.label_smoothing}")

        # ---- Compile ---------------------------------------------------------
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[IoUMetric(num_classes=NUM_CLASSES)],
            run_eagerly=False,
            jit_compile=True,
        )

    print("Model compiled with XLA JIT compilation")
    model.summary()

    # ---- Callbacks -----------------------------------------------------------
    ema_cb = EMACallback(decay=cfg.ema_decay) if cfg.use_ema else None

    checkpoint_path = os.path.join(cfg.checkpoint_dir, 'segformer_b2_best.weights.h5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_iou_metric',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=latest_weights,
            save_weights_only=True,
            save_best_only=False,
            save_freq='epoch',
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_iou_metric',
            patience=cfg.early_stopping_patience,
            mode='max',
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_iou_metric',
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            mode='max',
            min_lr=cfg.reduce_lr_min,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(cfg.log_dir, 'fit',
                                 'segformer_b2_' + datetime.now().strftime("%Y%m%d-%H%M%S")),
            update_freq='epoch',
        ),
        ProgressCallback(total_epochs=EPOCHS, steps_per_epoch=train_steps,
                         validation_steps=val_steps),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: open(
                os.path.join(cfg.checkpoint_dir, 'checkpoint_info.txt'), 'w'
            ).write(str(epoch + 1)),
        ),
    ]
    if ema_cb is not None:
        callbacks.append(ema_cb)

    # ---- Train ---------------------------------------------------------------
    print(f"\nStarting training for {EPOCHS} epochs from epoch {initial_epoch} …")
    print(f"Global batch size: {BATCH_SIZE}  |  GPUs: {strategy.num_replicas_in_sync if hasattr(strategy, 'num_replicas_in_sync') else 1}")

    history = model.fit(
        augmented_train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0,
        initial_epoch=initial_epoch,
    )

    # Apply EMA weights permanently before saving
    if ema_cb is not None:
        ema_cb.apply_ema_weights()
        print("Applied EMA weights to model")

    # Save final
    final_path = 'segformer_b2_final.weights.h5'
    model.save_weights(final_path)
    print(f"Final weights saved to {final_path}")

    plot_training_curves(history, save_path='segformer_b2_miou_curves.png')

    # ---- Post-training evaluation with TTA on validation set ----------------
    print("\nEvaluating on validation set with TTA …")
    from model.prediction import MultiScalePredictor
    from model.metrics import IoUMetric as _IoU

    tta_predictor = MultiScalePredictor(
        model, scales=[0.75, 1.0, 1.25], batch_size=BATCH_SIZE, use_tta=True,
    )
    test_metric = _IoU(num_classes=NUM_CLASSES)
    t0 = time.time()

    for images, labels in tqdm(val_ds, desc="Evaluating with TTA"):
        preds = tta_predictor.predict(images)
        test_metric.update_state(labels, preds)

    elapsed = time.time() - t0
    m, s = divmod(elapsed, 60)
    print(f"\nTTA Evaluation completed in {int(m)}m {s:.1f}s")
    mean_iou = test_metric.result().numpy()
    print(f"Val Mean IoU (TTA): {mean_iou:.4f}")
    class_iou = test_metric.get_class_iou()
    print("Class-wise IoU:")
    for cn, v in class_iou.items():
        print(f"  {cn}: {v:.4f}")

    # Basic multi-scale comparison
    print("\nComparing with basic multi-scale (no TTA) …")
    ms_predictor = MultiScalePredictor(model, scales=[0.75, 1.0, 1.25], use_tta=False)
    baseline_metric = _IoU(num_classes=NUM_CLASSES)
    for images, labels in tqdm(val_ds, desc="Multi-scale eval"):
        preds = ms_predictor.predict(images)
        baseline_metric.update_state(labels, preds)
    bl_iou = baseline_metric.result().numpy()
    print(f"Val Mean IoU (multi-scale only): {bl_iou:.4f}")

    diff = mean_iou - bl_iou
    colored_print(
        f"TTA improvement: {diff:+.4f} ({diff * 100:+.1f}%)",
        TermColors.GREEN if diff > 0 else TermColors.RED,
    )

    # Save CSV
    import pandas as pd

    bl_class = baseline_metric.get_class_iou()
    df = pd.DataFrame({
        'Class': list(class_iou.keys()),
        'IoU_TTA': [class_iou[c] for c in class_iou],
        'IoU_baseline': [bl_class[c] for c in bl_class],
        'Improvement': [class_iou[c] - bl_class[c] for c in class_iou],
    })
    os.makedirs('logs/class_ious', exist_ok=True)
    df.to_csv('logs/class_ious/final_class_ious.csv', index=False)
    print("IoU results saved to logs/class_ious/final_class_ious.csv")

    return history


if __name__ == "__main__":
    print("=" * 70)
    print("Starting SegFormer-B2 training for oil spill detection …")
    print("=" * 70)
    history = train_and_evaluate()
    print("Training completed successfully ✓")
