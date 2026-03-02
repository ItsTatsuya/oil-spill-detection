"""
Training script for SegFormer-B2 oil spill detection — PyTorch implementation.

Key features:
- Explicit training loop with AMP (autocast + GradScaler)
- Auto-detects GPUs: single-GPU, CPU, or multi-GPU via torchrun / DDP
- Cosine LR with linear warmup; ReduceLROnPlateau safety net
- EMA (torch-ema) — swapped in during validation
- TensorBoard logging
- Checkpoint resume + early stopping
- Post-training TTA evaluation

Single GPU / CPU usage:
    python train.py

Multi-GPU (2 GPUs on one machine):
    torchrun --nproc_per_node=2 train.py
"""

import os
import gc
import time
import logging
from datetime import datetime
from typing import Optional, cast

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ReduceLROnPlateau, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage

from utils import set_reproducibility, colored_print, TermColors
from config import TrainConfig, LossConfig, DataConfig

# ---------------------------------------------------------------------------
# Config + reproducibility (before DDP init)
# ---------------------------------------------------------------------------
cfg = TrainConfig()
set_reproducibility(cfg.seed)

logger = logging.getLogger('oil_spill')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------
def build_scheduler(optimizer, epochs: int, train_steps: int,
                    warmup_epochs: int = 25, cosine_alpha: float = 0.001):
    """Return a SequentialLR: LinearLR warmup then CosineAnnealingLR."""
    warmup_iters = max(1, warmup_epochs * train_steps)
    total_iters  = epochs * train_steps

    warmup = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_iters,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_iters - warmup_iters),
        eta_min=cfg.learning_rate * cosine_alpha,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_iters],
    )
    return scheduler


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_curves(train_losses, val_losses, train_ious, val_ious,
                         save_path='miou_curves.png'):
    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if train_ious and val_ious:
        axes[0].plot(epochs, train_ious, label='Train IoU')
        axes[0].plot(epochs[:len(val_ious)], val_ious, label='Val IoU')
        axes[0].set_title('Mean IoU Over Time')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Mean IoU')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, linestyle='--', alpha=0.6)

    if train_losses and val_losses:
        axes[1].plot(epochs, train_losses, label='Train Loss')
        axes[1].plot(epochs[:len(val_losses)], val_losses, label='Val Loss')
        axes[1].set_title('Loss Over Time')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, scaler, ema, epoch, best_iou):
    """Save full training state."""
    raw_model = model.module if hasattr(model, 'module') else model
    torch.save({
        'epoch':      epoch,
        'model':      raw_model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'scaler':     scaler.state_dict(),
        'ema':        ema.state_dict(),
        'best_iou':   best_iou,
    }, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, ema=None, device='cpu'):
    """Load training state. Returns the epoch to resume from."""
    ckpt = torch.load(path, map_location=device)
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    if ema is not None and 'ema' in ckpt:
        ema.load_state_dict(ckpt['ema'])
    epoch    = ckpt.get('epoch', 0)
    best_iou = ckpt.get('best_iou', 0.0)
    return epoch, best_iou


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------
@torch.inference_mode()
def validate(model, val_loader, criterion, device, metric, use_amp=True):
    """Run one validation pass. Returns (mean_loss, mean_iou)."""
    model.eval()
    metric.reset_state()
    total_loss = 0.0
    n_batches  = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        with autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)
        total_loss += loss.item()
        metric.update_state(labels, logits)
        n_batches  += 1

    mean_loss = total_loss / max(n_batches, 1)
    mean_iou  = float(metric.result().item())
    model.train()
    return mean_loss, mean_iou


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train_and_evaluate():
    # ---- DDP setup ---------------------------------------------------------
    rank       = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_main    = (rank == 0)

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if is_main:
        print(f"\n{'='*70}")
        print(f"PyTorch SegFormer-B2  —  oil spill detection")
        print(f"Device: {device}  |  World size: {world_size}  |  Seed: {cfg.seed}")
        print(f"{'='*70}\n")

    # ---- Config ------------------------------------------------------------
    NUM_CLASSES = cfg.num_classes
    PER_GPU_BS  = cfg.per_gpu_batch_size
    BATCH_SIZE  = PER_GPU_BS * world_size
    EPOCHS      = cfg.epochs
    LR          = cfg.learning_rate
    IMG_SIZE    = (384, 384)
    use_amp     = torch.cuda.is_available()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ---- Data --------------------------------------------------------------
    dcfg = DataConfig()

    # DataLoader for DDP: DistributedSampler handles per-rank sharding
    from torch.utils.data import DataLoader as _DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from data.data_loader import OilSpillDataset, analyze_class_distribution, _worker_init_fn
    import glob

    train_sampler = None  # set below when world_size > 1

    def _make_loaders():
        nonlocal train_sampler
        base_dir  = os.path.join(dcfg.data_dir, 'train')
        image_dir = os.path.join(base_dir, 'images')
        label_dir = os.path.join(base_dir, 'labels_1D')
        all_images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        all_labels = sorted(glob.glob(os.path.join(label_dir, '*.png')))

        n_total  = len(all_images)
        n_val    = max(1, int(dcfg.val_split * n_total))
        tr_imgs  = all_images[:-n_val];  tr_lbls = all_labels[:-n_val]
        va_imgs  = all_images[-n_val:];  va_lbls = all_labels[-n_val:]

        cw = analyze_class_distribution(
            tr_lbls, num_classes=NUM_CLASSES,
            cache_path=dcfg.class_weights_cache,
        )

        tr_ds = OilSpillDataset(tr_imgs, tr_lbls, split='train', augment=True)
        va_ds = OilSpillDataset(va_imgs, va_lbls, split='val',   augment=False)

        if world_size > 1:
            train_sampler = DistributedSampler(tr_ds, num_replicas=world_size,
                                               rank=rank, shuffle=True)
            val_sampler   = DistributedSampler(va_ds, num_replicas=world_size,
                                               rank=rank, shuffle=False)
        else:
            train_sampler = None
            val_sampler   = None

        tr_loader = _DataLoader(
            tr_ds, batch_size=PER_GPU_BS,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=dcfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=(dcfg.num_workers > 0),
            worker_init_fn=_worker_init_fn if dcfg.num_workers > 0 else None,
        )
        va_loader = _DataLoader(
            va_ds, batch_size=PER_GPU_BS,
            shuffle=False,
            sampler=val_sampler,
            num_workers=dcfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(dcfg.num_workers > 0),
            worker_init_fn=_worker_init_fn if dcfg.num_workers > 0 else None,
        )
        return tr_loader, va_loader, cw, len(tr_loader), len(va_loader)

    if is_main:
        print("Loading datasets …")
    train_loader, val_loader, class_weights_dict, train_steps, val_steps = _make_loaders()

    if is_main:
        for images, labels in train_loader:
            print(f"Train batch — images: {tuple(images.shape)} ({images.dtype}), "
                  f"labels: {tuple(labels.shape)}")
            break
        if train_steps == 0 or val_steps == 0:
            raise ValueError("Empty dataset detected — check data path.")

    gc.collect()

    # ---- Model -------------------------------------------------------------
    from model.model import OilSpillSegformer
    from model.loss import HybridSegmentationLoss
    from model.metrics import IoUMetric

    pretrained_path = cfg.pretrained_weights
    use_pretrained  = os.path.exists(pretrained_path)
    if is_main:
        print(f"Pre-trained weights {'found' if use_pretrained else 'not found'} "
              f"at {pretrained_path}")

    model = OilSpillSegformer(
        input_shape=(*IMG_SIZE, 1),
        num_classes=NUM_CLASSES,
        drop_rate=0.1,
        use_cbam=False,
        pretrained_weights=pretrained_path if use_pretrained else None,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ---- Loss + metrics ----------------------------------------------------
    lcfg = LossConfig()
    class_weights_list = [class_weights_dict[i] for i in range(NUM_CLASSES)] \
                         if class_weights_dict else None
    criterion = HybridSegmentationLoss(
        class_weights=class_weights_list,
        ce_weight=lcfg.ce_weight,
        focal_weight=lcfg.focal_weight,
        dice_weight=lcfg.dice_weight,
        boundary_weight=lcfg.boundary_weight,
        gamma=lcfg.focal_gamma,
        label_smoothing=lcfg.label_smoothing,
    ).to(device)

    metric = IoUMetric(num_classes=NUM_CLASSES, device=device)

    if is_main:
        print(f"HybridSegmentationLoss: CE={lcfg.ce_weight}, "
              f"Focal={lcfg.focal_weight} (γ={lcfg.focal_gamma}), "
              f"Dice={lcfg.dice_weight}, Boundary={lcfg.boundary_weight}, "
              f"label_smoothing={lcfg.label_smoothing}")

    # ---- Optimizer + scheduler + scaler + EMA ------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(cfg.beta_1, cfg.beta_2),
        eps=cfg.epsilon,
        weight_decay=cfg.weight_decay,
        amsgrad=cfg.amsgrad,
    )

    scheduler      = build_scheduler(optimizer, EPOCHS, train_steps,
                                     warmup_epochs=cfg.warmup_epochs,
                                     cosine_alpha=cfg.cosine_alpha)
    plateau_sched  = ReduceLROnPlateau(optimizer, mode='max',
                                       factor=cfg.reduce_lr_factor,
                                       patience=cfg.reduce_lr_patience,
                                       min_lr=cfg.reduce_lr_min)
    scaler = GradScaler('cuda', enabled=use_amp)
    raw_model = model.module if hasattr(model, 'module') else model
    assert isinstance(raw_model, torch.nn.Module)
    ema = ExponentialMovingAverage(raw_model.parameters(), decay=cfg.ema_decay)

    # ---- Checkpoint resume -------------------------------------------------
    initial_epoch = 0
    best_iou      = 0.0
    latest_ckpt   = os.path.join(cfg.checkpoint_dir, 'segformer_b2_latest.pt')
    best_ckpt     = os.path.join(cfg.checkpoint_dir, 'segformer_b2_best.pt')

    if os.path.exists(latest_ckpt):
        if is_main:
            print(f"Resuming from {latest_ckpt} …")
        try:
            initial_epoch, best_iou = load_checkpoint(
                latest_ckpt, model, optimizer, scaler, ema, device=str(device)
            )
            if is_main:
                print(f"  → epoch {initial_epoch}, best_iou={best_iou:.4f}")
        except Exception as e:
            if is_main:
                print(f"Checkpoint load failed ({e}) — training from scratch")
            initial_epoch = 0
    else:
        if is_main:
            print("No checkpoint found — starting from scratch / pre-trained backbone.")

    # ---- TensorBoard -------------------------------------------------------
    writer: Optional[SummaryWriter] = None
    if is_main:
        run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        tb_dir = os.path.join(cfg.log_dir, 'fit', f'segformer_b2_{run_id}')
        writer = SummaryWriter(log_dir=tb_dir)

    # ---- History buffers ---------------------------------------------------
    train_losses, val_losses  = [], []
    train_ious,   val_ious    = [], []
    patience_counter          = 0

    # ---- Training loop -----------------------------------------------------
    if is_main:
        print(f"\nStarting training for {EPOCHS} epochs from epoch {initial_epoch} …")
        print(f"Batch/GPU: {PER_GPU_BS}  |  Global batch: {BATCH_SIZE}  |  "
              f"GPUs: {world_size}  |  AMP: {use_amp}\n")

    for epoch in range(initial_epoch, EPOCHS):
        # DistributedSampler must be re-seeded each epoch for proper shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        metric.reset_state()
        t_ep = time.time()

        if is_main:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        else:
            pbar = train_loader

        for step, (images, labels) in enumerate(pbar):
            images, labels = images.to(device, non_blocking=True), \
                             labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=use_amp):
                logits = model(images)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), cfg.clipnorm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update()

            epoch_loss += loss.item()
            metric.update_state(labels, logits.detach())

            if is_main and step % max(1, train_steps // 20) == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                cast(tqdm, pbar).set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.2e}")

        # Capture training IoU BEFORE validate() resets+overwrites the metric
        train_iou = float(metric.result().item())

        # ---- Validation with EMA weights -----------------------------------
        with ema.average_parameters():
            val_loss, val_iou = validate(model, val_loader, criterion,
                                         device, metric, use_amp)
        mean_loss = epoch_loss / max(train_steps, 1)
        elapsed   = time.time() - t_ep
        cur_lr    = optimizer.param_groups[0]['lr']

        train_losses.append(mean_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        if is_main:
            print(f"Epoch {epoch+1}/{EPOCHS} — "
                  f"Loss: {mean_loss:.4f}  IoU: {train_iou:.4f}  |  "
                  f"Val Loss: {val_loss:.4f}  Val IoU: {val_iou:.4f}  |  "
                  f"LR: {cur_lr:.2e}  |  {elapsed:.1f}s")
            assert writer is not None
            writer.add_scalars('Loss',
                {'train': mean_loss, 'val': val_loss}, epoch)
            writer.add_scalars('IoU',
                {'train': train_iou, 'val': val_iou}, epoch)
            writer.add_scalar('lr', cur_lr, epoch)

        # ---- Plateau scheduler (operates per epoch) ------------------------
        plateau_sched.step(val_iou)

        # ---- Checkpoint + early stopping -----------------------------------
        if is_main:
            if val_iou > best_iou:
                best_iou = val_iou
                save_checkpoint(best_ckpt, model, optimizer, scaler, ema,
                                epoch + 1, best_iou)
                print(f"  ↑ New best IoU: {best_iou:.4f} — saved to {best_ckpt}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    print(f"Early stopping: no improvement for "
                          f"{cfg.early_stopping_patience} epochs")
                    break
            # Save latest after best_iou is updated so resume is consistent
            save_checkpoint(latest_ckpt, model, optimizer, scaler, ema,
                            epoch + 1, best_iou)

    # ---- Apply EMA and save final model ------------------------------------
    if is_main:
        with ema.average_parameters():
            final_path = 'segformer_b2_final.pt'
            raw_model = model.module if hasattr(model, 'module') else model
            assert isinstance(raw_model, torch.nn.Module)
            torch.save(raw_model.state_dict(), final_path)
            print(f"\nFinal model saved to {final_path}")

        plot_training_curves(train_losses, val_losses, train_ious, val_ious,
                             save_path='segformer_b2_miou_curves.png')
        assert writer is not None
        writer.close()

    # ---- Post-training TTA evaluation (main process only) ------------------
    if is_main:
        print("\nEvaluating on validation set with TTA …")
        from model.prediction import MultiScalePredictor

        raw_model = model.module if hasattr(model, 'module') else model
        with ema.average_parameters():
            tta_predictor  = MultiScalePredictor(raw_model, scales=[0.75, 1.0, 1.25],
                                                  use_tta=True, device=device)
            ms_predictor   = MultiScalePredictor(raw_model, scales=[0.75, 1.0, 1.25],
                                                  use_tta=False, device=device)
            tta_metric     = IoUMetric(num_classes=NUM_CLASSES, device=device)
            ms_metric      = IoUMetric(num_classes=NUM_CLASSES, device=device)
            t0 = time.time()

            for images, labels in tqdm(val_loader, desc="TTA eval"):
                images = images.to(device)
                labels = labels.to(device)
                preds  = tta_predictor.predict(images)
                tta_metric.update_state(labels, preds)
                preds  = ms_predictor.predict(images)
                ms_metric.update_state(labels, preds)

        elapsed = time.time() - t0
        print(f"TTA Evaluation completed in {elapsed:.1f}s")
        tta_iou  = float(tta_metric.result().item())
        ms_iou   = float(ms_metric.result().item())
        diff     = tta_iou - ms_iou
        print(f"Val Mean IoU — TTA: {tta_iou:.4f}  |  Multi-scale: {ms_iou:.4f}  "
              f"|  Δ: {diff:+.4f}")
        print("Class-wise IoU (TTA):")
        for cn, v in tta_metric.get_class_iou().items():
            print(f"  {cn}: {v:.4f}")

        # Save CSV
        import pandas as pd
        tta_cls = tta_metric.get_class_iou()
        ms_cls  = ms_metric.get_class_iou()
        df = pd.DataFrame({
            'Class':       list(tta_cls.keys()),
            'IoU_TTA':     list(tta_cls.values()),
            'IoU_ms':      [ms_cls[c] for c in tta_cls],
            'Improvement': [tta_cls[c] - ms_cls[c] for c in tta_cls],
        })
        os.makedirs('logs/class_ious', exist_ok=True)
        df.to_csv('logs/class_ious/final_class_ious.csv', index=False)
        print("IoU results saved to logs/class_ious/final_class_ious.csv")

    # ---- Cleanup DDP -------------------------------------------------------
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    print("=" * 70)
    print("Starting SegFormer-B2 training for oil spill detection …")
    print("=" * 70)
    train_and_evaluate()
    print("Training completed successfully ✓")
