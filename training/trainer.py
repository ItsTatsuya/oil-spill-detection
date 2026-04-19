from __future__ import annotations

import contextlib
import logging
import time
import warnings
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from constants import CLASS_NAMES
from data.augmentation import SARSegmentationAugmentation, get_progressive_crop_size
from data.dataloader import CurriculumDataLoaderFactory
from inference.pipeline import InferencePipeline
from losses.combined_loss import CombinedLoss
from training.callbacks import CheckpointCallback, ModelEMA
from training.metrics import SegmentationMetrics
from utils.distributed import get_rank, get_world_size, is_main_process

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Any,
        val_dataset: Any,
        config: dict,
        is_distributed: bool = False,
        local_rank: int = 0,
        exp_logger: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.is_distributed = is_distributed
        self.local_rank = local_rank
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.logger = exp_logger
        self.device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        if not self.is_distributed:
            self.model = self.model.to(self.device)

        train_cfg = config.get("training", {})
        self.num_epochs = int(train_cfg.get("num_epochs", 220))
        self.gradient_clip_norm = float(train_cfg.get("gradient_clip_norm", 1.0))
        self.validate_every = int(train_cfg.get("validate_every_n_epochs", 2))
        self.save_every = int(train_cfg.get("save_every_n_epochs", 10))
        self.checkpoint_dir = str(
            train_cfg.get("checkpoint_dir", "./checkpoints/segformer")
        )
        self.precision = str(train_cfg.get("precision", "bf16")).lower()
        self.grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
        self.batch_size = int(train_cfg.get("batch_size_per_gpu", 4))
        self.num_workers = int(train_cfg.get("num_workers", 4))
        self.val_num_workers = int(train_cfg.get("val_num_workers", 2))
        self.pin_memory = bool(train_cfg.get("pin_memory", True))
        self.persistent_workers = bool(train_cfg.get("persistent_workers", True))
        self.progressive_schedule = {
            int(k): int(v)
            for k, v in train_cfg.get("progressive_schedule", {1: 512, 41: 576}).items()
        }
        self.early_stop_patience_epochs = int(
            train_cfg.get("early_stop_patience_epochs", 30)
        )
        self.early_stop_start_epoch = int(train_cfg.get("early_stop_start_epoch", 100))
        self.early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))

        ema_cfg = train_cfg.get("model_ema", {})
        self.use_model_ema = bool(ema_cfg.get("enabled", True))
        self.use_ema_for_validation = bool(ema_cfg.get("use_for_validation", True))
        self.save_best_ema = bool(ema_cfg.get("save_best_ema", True))
        self.model_ema = (
            ModelEMA(
                self.model,
                decay=float(ema_cfg.get("decay", 0.9998)),
                update_after_step=int(ema_cfg.get("update_after_step", 0)),
            )
            if self.use_model_ema
            else None
        )

        self.use_amp = self.device.type == "cuda" and self.precision in {"fp16", "bf16"}
        self.amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.precision == "fp16"))

        self.loss_fn = CombinedLoss(config).to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.checkpoint_cb = CheckpointCallback(
            self.checkpoint_dir,
            save_every_n_epochs=self.save_every,
            keep_top_k_ema_checkpoints=int(
                train_cfg.get("keep_top_k_ema_checkpoints", 0)
            ),
        )
        self.primary_validation_profile = (
            str(train_cfg.get("validation_profile", "fast")).strip().lower()
        )
        if self.primary_validation_profile not in {"fast", "full"}:
            logger.warning(
                "Unknown training.validation_profile=%r. Falling back to 'fast'.",
                self.primary_validation_profile,
            )
            self.primary_validation_profile = "fast"
        self.evaluation_profile = (
            str(config.get("evaluation", {}).get("profile", "full")).strip().lower()
        )
        if self.evaluation_profile not in {"fast", "full"}:
            logger.warning(
                "Unknown evaluation.profile=%r. Falling back to 'full'.",
                self.evaluation_profile,
            )
            self.evaluation_profile = "full"
        logger.info(
            "Training validation profile=%s; evaluation profile=%s.",
            self.primary_validation_profile,
            self.evaluation_profile,
        )
        if self.primary_validation_profile != self.evaluation_profile:
            logger.warning(
                "training.validation_profile (%s) differs from evaluation.profile (%s). "
                "This is allowed but can yield different train-time vs final-eval metrics.",
                self.primary_validation_profile,
                self.evaluation_profile,
            )
        self.best_val_miou = 0.0
        self._best_val_miou_for_patience = 0.0
        self.start_epoch = 1
        self.global_step = 0
        self._epochs_since_improvement = 0
        self._last_improvement_epoch = 0
        self._last_val_metrics: dict[str, float] = {}
        self._last_val_metrics_by_profile: dict[str, dict[str, float]] = {}
        self._last_train_perf_metrics: dict[str, float] = {}
        self._last_val_perf_metrics: dict[str, float] = {}
        self._current_crop_size: Optional[int] = None
        self._train_sampler: Optional[DistributedSampler] = None
        self._train_loader: Optional[DataLoader] = None
        self.curriculum_loader_factory: Optional[CurriculumDataLoaderFactory] = None

        if config.get("curriculum"):
            self.curriculum_loader_factory = CurriculumDataLoaderFactory(
                dataset=self.train_dataset,
                config=self.config,
                is_distributed=self.is_distributed,
                rank=self.rank,
                world_size=self.world_size,
            )

        self.val_loader = self._build_validation_dataloader()

    def _build_optimizer(self) -> optim.Optimizer:
        opt_cfg = self.config.get("optimizer", {})
        optimizer_name = str(opt_cfg.get("name", "adamw")).strip().lower()
        base_lr = float(opt_cfg.get("lr", 6e-5))
        head_lr_mult = float(opt_cfg.get("head_lr_mult", 10.0))
        head_lr = base_lr * head_lr_mult
        head_tokens = (
            "decoder.",
            "projections.",
            "fuse.",
            "dropout.",
            "classifier.",
            "enhancers.",
            "aspp.",
            "fuse_s16.",
            "fuse_s8.",
            "fuse_s4.",
            "esem_s16.",
            "esem_s8.",
            "esem_s4.",
            "aux_head_s16.",
            "aux_head_s8.",
            "decode_head.",
            "segmentation_head.",
        )
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name for token in head_tokens):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": base_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr})
        if not param_groups:
            raise RuntimeError("No trainable parameters found for optimizer.")

        logger.info(
            "Optimizer param groups: backbone=%d (lr=%.2e), head=%d (lr=%.2e), head_lr_mult=%.2f",
            len(backbone_params),
            base_lr,
            len(head_params),
            head_lr,
            head_lr_mult,
        )

        fused_cfg_raw = opt_cfg.get("fused", "auto")
        fused_mode: bool | str
        if isinstance(fused_cfg_raw, bool):
            fused_mode = fused_cfg_raw
        elif isinstance(fused_cfg_raw, str):
            normalized = fused_cfg_raw.strip().lower()
            if normalized in {"auto"}:
                fused_mode = "auto"
            elif normalized in {"true", "1", "yes", "on"}:
                fused_mode = True
            elif normalized in {"false", "0", "no", "off"}:
                fused_mode = False
            else:
                logger.warning(
                    "Unknown optimizer.fused=%r. Expected true/false/'auto'; using 'auto'.",
                    fused_cfg_raw,
                )
                fused_mode = "auto"
        elif fused_cfg_raw is None:
            fused_mode = "auto"
        else:
            logger.warning(
                "Unknown optimizer.fused=%r. Expected true/false/'auto'; using 'auto'.",
                fused_cfg_raw,
            )
            fused_mode = "auto"

        use_fused = fused_mode is True or (
            fused_mode == "auto" and self.device.type == "cuda"
        )
        if use_fused and self.device.type != "cuda":
            logger.warning(
                "optimizer.fused requested on %s device; falling back to standard AdamW.",
                self.device.type,
            )
            use_fused = False

        optimizer_kwargs: dict[str, Any] = {
            "lr": base_lr,
            "weight_decay": float(opt_cfg.get("weight_decay", 0.01)),
        }
        if optimizer_name in {"adamw", "adam"}:
            optimizer_kwargs["betas"] = tuple(opt_cfg.get("betas", [0.9, 0.999]))
            optimizer_kwargs["eps"] = float(opt_cfg.get("eps", 1e-8))
            if optimizer_name == "adamw" and use_fused:
                optimizer_kwargs["fused"] = True
        elif optimizer_name == "sgd":
            if use_fused:
                logger.warning(
                    "optimizer.fused is only supported for AdamW in this trainer; ignoring it for SGD."
                )
            optimizer_kwargs["momentum"] = float(opt_cfg.get("momentum", 0.9))
            optimizer_kwargs["nesterov"] = bool(opt_cfg.get("nesterov", True))
        else:
            logger.warning(
                "Unknown optimizer.name=%r. Falling back to 'adamw'.",
                optimizer_name,
            )
            optimizer_name = "adamw"
            optimizer_kwargs["betas"] = tuple(opt_cfg.get("betas", [0.9, 0.999]))
            optimizer_kwargs["eps"] = float(opt_cfg.get("eps", 1e-8))
            if use_fused:
                optimizer_kwargs["fused"] = True

        optimizer_cls = {
            "adamw": optim.AdamW,
            "adam": optim.Adam,
            "sgd": optim.SGD,
        }[optimizer_name]

        try:
            optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        except (TypeError, RuntimeError) as exc:
            if optimizer_name == "adamw" and "fused" in optimizer_kwargs:
                logger.warning(
                    "Fused AdamW unavailable in this torch/runtime (%s). Falling back to standard AdamW.",
                    exc,
                )
                optimizer_kwargs.pop("fused", None)
                optimizer = optim.AdamW(param_groups, **optimizer_kwargs)
            else:
                raise

        logger.info(
            "Using %s optimizer (fused=%s).",
            optimizer_name.upper(),
            bool(optimizer_kwargs.get("fused")),
        )
        return optimizer

    def _build_scheduler(self):
        sched_cfg = self.config.get("scheduler", {})
        scheduler_type = str(sched_cfg.get("type", "cosine")).strip().lower()
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 5))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))
        base_lr = float(self.config.get("optimizer", {}).get("lr", 6e-5))
        min_lr_ratio = max(min_lr / max(base_lr, 1e-12), 1e-6)

        if scheduler_type not in {"cosine", "cosine_restarts"}:
            logger.warning(
                "Unknown scheduler.type=%r. Falling back to 'cosine'.",
                scheduler_type,
            )
            scheduler_type = "cosine"

        if scheduler_type == "cosine_restarts":
            restart_t0 = int(sched_cfg.get("restart_t0", 40))
            restart_t_mult = int(sched_cfg.get("restart_t_mult", 2))
            restart = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(restart_t0, 1),
                T_mult=max(restart_t_mult, 1),
                eta_min=min_lr,
            )
            if warmup_epochs <= 0:
                return restart

            warmup = LinearLR(
                self.optimizer,
                start_factor=min_lr_ratio,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "Detected call of `lr_scheduler.step\\(\\)` before"
                )
                return SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, restart],
                    milestones=[warmup_epochs],
                )

        if warmup_epochs <= 0:
            return CosineAnnealingLR(
                self.optimizer,
                T_max=max(self.num_epochs, 1),
                eta_min=min_lr,
            )

        warmup = LinearLR(
            self.optimizer,
            start_factor=min_lr_ratio,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.num_epochs - warmup_epochs, 1),
            eta_min=min_lr,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Detected call of `lr_scheduler.step\\(\\)` before"
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )

    def _build_train_dataloader(self) -> DataLoader:
        sampler = None
        shuffle = True
        if self.is_distributed and self.world_size > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            shuffle = False
        self._train_sampler = sampler

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def _set_train_epoch(self, epoch: int) -> None:
        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)

    def _shutdown_loader_workers(self, loader: Optional[DataLoader]) -> None:
        if loader is None:
            return
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
            loader._iterator = None

    def _get_train_dataloader(self, epoch: int, rebuild: bool = False) -> DataLoader:
        if self.curriculum_loader_factory is not None:
            if hasattr(self.train_dataset, "set_epoch"):
                self.train_dataset.set_epoch(epoch)
            if rebuild:
                self.curriculum_loader_factory.clear_cache()
            loader = self.curriculum_loader_factory.get_dataloader(epoch)
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            return loader

        if rebuild and self._train_loader is not None:
            self._shutdown_loader_workers(self._train_loader)
            self._train_loader = None

        if self._train_loader is None:
            self._train_loader = self._build_train_dataloader()

        self._set_train_epoch(epoch)
        return self._train_loader

    def _build_validation_dataloader(self) -> DataLoader:
        dataset = self.val_dataset
        if self.is_distributed and self.world_size > 1:
            dataset = Subset(
                self.val_dataset,
                list(range(self.rank, len(self.val_dataset), self.world_size)),
            )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.val_num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.val_num_workers > 0,
        )

    def _refresh_train_augmentation(self, epoch: int) -> bool:
        crop_size = get_progressive_crop_size(epoch, self.progressive_schedule)
        if crop_size == self._current_crop_size:
            return False
        aug = SARSegmentationAugmentation(self.config)
        self.train_dataset.transform = aug.get_train_transform(
            crop_size_override=crop_size
        )
        if hasattr(self.train_dataset, "set_crop_size"):
            self.train_dataset.set_crop_size(crop_size)
        self._current_crop_size = crop_size
        logger.info("Epoch %d training resize -> %dx%d", epoch, crop_size, crop_size)
        return True

    def _reset_cuda_peak_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _cuda_memory_metrics(self, prefix: str) -> dict[str, float]:
        if self.device.type != "cuda":
            return {}
        return {
            f"mem/{prefix}_cuda_max_allocated_mb": float(
                torch.cuda.max_memory_allocated(self.device) / (1024**2)
            ),
            f"mem/{prefix}_cuda_max_reserved_mb": float(
                torch.cuda.max_memory_reserved(self.device) / (1024**2)
            ),
        }

    def _build_resume_state(self) -> dict[str, Any]:
        pixel_counts_source = str(
            self.config.get("training", {}).get("pixel_counts_source", "split")
        )
        return {
            "best_val_miou": float(self.best_val_miou),
            "epochs_since_improvement": int(self._epochs_since_improvement),
            "last_improvement_epoch": int(self._last_improvement_epoch),
            "best_val_miou_for_patience": float(self._best_val_miou_for_patience),
            "last_val_metrics": dict(self._last_val_metrics),
            "pixel_counts_source": pixel_counts_source,
        }

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        ignore_loss_state: bool = False,
    ) -> None:
        if checkpoint_path is None:
            latest = self.checkpoint_cb.find_latest()
            if latest is None:
                logger.info("No checkpoint found. Starting from scratch.")
                return
            checkpoint_path = str(latest)

        info = self.checkpoint_cb.load(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            current_config=self.config,
            scaler=self.scaler,
            loss_fn=self.loss_fn,
            model_ema=self.model_ema,
            ignore_loss_state=ignore_loss_state,
        )
        self.start_epoch = int(info["epoch"]) + 1
        self.global_step = int(info.get("global_step", 0))
        trainer_state = info.get("trainer_state", {}) or {}
        # Prefer explicit historical best if present; fall back for old checkpoints.
        self.best_val_miou = float(
            trainer_state.get("best_val_miou", info.get("val_miou", 0.0))
        )
        self._best_val_miou_for_patience = float(
            trainer_state.get("best_val_miou_for_patience", self.best_val_miou)
        )
        self._epochs_since_improvement = int(
            trainer_state.get("epochs_since_improvement", 0)
        )
        self._last_improvement_epoch = int(
            trainer_state.get("last_improvement_epoch", 0)
        )
        checkpoint_pixel_counts_source = trainer_state.get("pixel_counts_source")
        if isinstance(checkpoint_pixel_counts_source, str):
            self.config.setdefault("training", {})["pixel_counts_source"] = (
                checkpoint_pixel_counts_source
            )
            logger.info(
                "Resumed training.pixel_counts_source=%s from checkpoint metadata.",
                checkpoint_pixel_counts_source,
            )
        if isinstance(trainer_state.get("last_val_metrics"), dict):
            self._last_val_metrics = dict(trainer_state["last_val_metrics"])
        if isinstance(info.get("val_metrics_by_profile"), dict):
            self._last_val_metrics_by_profile = {
                str(profile): dict(metrics)
                for profile, metrics in info["val_metrics_by_profile"].items()
                if isinstance(metrics, dict)
            }
        checkpoint_profile = info.get("primary_validation_profile")
        if self.primary_validation_profile in self._last_val_metrics_by_profile:
            self._last_val_metrics = dict(
                self._last_val_metrics_by_profile[self.primary_validation_profile]
            )
        elif self._last_val_metrics_by_profile:
            first_profile, first_metrics = next(
                iter(self._last_val_metrics_by_profile.items())
            )
            self._last_val_metrics = dict(first_metrics)
            if isinstance(checkpoint_profile, str):
                logger.info(
                    "Keeping current validation profile '%s'; checkpoint history only has '%s'.",
                    self.primary_validation_profile,
                    first_profile,
                )
        logger.info(
            "Resumed from epoch %d with best val mIoU %.2f%%",
            self.start_epoch - 1,
            self.best_val_miou * 100.0,
        )

    def _sync_validation_state(
        self,
        best_val_miou: float,
        best_val_miou_for_patience: float,
        epochs_since_improvement: int,
        should_stop: bool,
    ) -> tuple[float, float, int, bool]:
        if not (self.is_distributed and self.world_size > 1):
            return (
                best_val_miou,
                best_val_miou_for_patience,
                epochs_since_improvement,
                should_stop,
            )
        state = torch.tensor(
            [
                best_val_miou,
                best_val_miou_for_patience,
                float(epochs_since_improvement),
                float(should_stop),
            ],
            device=self.device,
            dtype=torch.float64,
        )
        dist.broadcast(state, src=0)
        return (
            float(state[0]),
            float(state[1]),
            int(state[2].item()),
            bool(int(state[3].item())),
        )

    def train(self) -> None:
        logger.info(
            "Starting training from epoch %d to %d",
            self.start_epoch,
            self.num_epochs,
        )

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            transform_changed = self._refresh_train_augmentation(epoch)
            train_loader = self._get_train_dataloader(epoch, rebuild=transform_changed)
            train_metrics = self.train_epoch(train_loader, epoch)
            optimizer_steps_in_epoch = int(train_metrics.get("optimizer_steps", 0.0))
            if self.is_distributed and self.world_size > 1:
                step_tensor = torch.tensor(
                    float(optimizer_steps_in_epoch), device=self.device
                )
                dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
                optimizer_steps_in_epoch = int(step_tensor.item())

            if self.logger is not None and is_main_process():
                log_data = {f"train/{k}": v for k, v in train_metrics.items()}
                log_data["epoch"] = epoch
                log_data.update(self._last_train_perf_metrics)
                for i, pg in enumerate(self.optimizer.param_groups):
                    log_data[f"lr/group_{i}"] = float(pg["lr"])
                self.logger.log_metrics(log_data, step=epoch)

            should_stop = False
            if epoch % self.validate_every == 0:
                validation_results = self.validate(epoch)
                val_miou = float(validation_results["mean_iou"])

                if is_main_process():
                    previous_best = self.best_val_miou
                    previous_patience_best = self._best_val_miou_for_patience
                    is_best = val_miou > previous_best
                    meaningful_improvement = val_miou > (
                        previous_patience_best + self.early_stop_min_delta
                    )
                    if is_best:
                        self.best_val_miou = val_miou
                    if meaningful_improvement:
                        self._best_val_miou_for_patience = val_miou
                        self._epochs_since_improvement = 0
                        self._last_improvement_epoch = epoch
                    elif epoch >= self.early_stop_start_epoch:
                        stall_reference_epoch = max(
                            self._last_improvement_epoch,
                            self.early_stop_start_epoch,
                        )
                        self._epochs_since_improvement = epoch - stall_reference_epoch

                    self.checkpoint_cb.save(
                        epoch=epoch,
                        val_miou=val_miou,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        config=self.config,
                        is_best=is_best,
                        rank=self.rank,
                        scaler=self.scaler,
                        loss_fn=self.loss_fn,
                        model_ema=self.model_ema,
                        global_step=self.global_step,
                        save_best_ema=self.save_best_ema,
                        val_metrics_by_profile={
                            self.primary_validation_profile: self._last_val_metrics
                        },
                        primary_validation_profile=self.primary_validation_profile,
                        trainer_state=self._build_resume_state(),
                        track_top_k_ema=True,
                    )

                    if self.logger is not None:
                        val_log = {"val/mean_iou": val_miou}
                        for name in CLASS_NAMES:
                            key = f"{name}_iou"
                            if key in self._last_val_metrics:
                                val_log[f"val/iou_{name}"] = float(
                                    self._last_val_metrics[key]
                                )
                        val_log.update(self._last_val_perf_metrics)
                        self.logger.log_metrics(val_log, step=epoch)

                    should_stop = (
                        epoch >= self.early_stop_start_epoch
                        and self._epochs_since_improvement
                        >= self.early_stop_patience_epochs
                    )

                (
                    self.best_val_miou,
                    self._best_val_miou_for_patience,
                    self._epochs_since_improvement,
                    should_stop,
                ) = self._sync_validation_state(
                    self.best_val_miou,
                    self._best_val_miou_for_patience,
                    self._epochs_since_improvement,
                    should_stop,
                )
                if should_stop:
                    logger.warning(
                        "Early stopping triggered at epoch %d after %d stalled epochs.",
                        epoch,
                        self._epochs_since_improvement,
                    )
                    break
            elif epoch % self.save_every == 0 and is_main_process():
                self.checkpoint_cb.save(
                    epoch=epoch,
                    val_miou=self.best_val_miou,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    config=self.config,
                    is_best=False,
                    rank=self.rank,
                    scaler=self.scaler,
                    loss_fn=self.loss_fn,
                    model_ema=self.model_ema,
                    global_step=self.global_step,
                    save_best_ema=self.save_best_ema,
                    val_metrics_by_profile={
                        self.primary_validation_profile: self._last_val_metrics
                    },
                    primary_validation_profile=self.primary_validation_profile,
                    trainer_state=self._build_resume_state(),
                    track_top_k_ema=False,
                )

            if self.is_distributed:
                dist.barrier()
            if optimizer_steps_in_epoch > 0:
                self.scheduler.step()
            else:
                logger.warning(
                    "No optimizer step executed in epoch %d; skipping scheduler.step() to keep LR schedule aligned.",
                    epoch,
                )

        logger.info(
            "Training complete. Best validation mIoU: %.2f%%",
            self.best_val_miou * 100.0,
        )
        if self.curriculum_loader_factory is not None:
            self.curriculum_loader_factory.clear_cache()
        self._shutdown_loader_workers(self._train_loader)
        self._shutdown_loader_workers(self.val_loader)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.loss_fn.train()
        self._reset_cuda_peak_memory()
        epoch_start = time.perf_counter()
        self.optimizer.zero_grad()

        total_metrics = {
            "loss_total": 0.0,
            "loss_ce": 0.0,
            "loss_dice": 0.0,
            "loss_jaccard": 0.0,
            "loss_focal": 0.0,
            "loss_lovasz": 0.0,
            "loss_boundary": 0.0,
            "loss_confusion": 0.0,
            "loss_aux_s16": 0.0,
            "loss_aux_s8": 0.0,
        }
        total_batches = max(len(dataloader), 1)
        remainder_batches = total_batches % self.grad_accum_steps
        seen_batches = 0
        optimizer_steps_applied = 0
        skipped_optimizer_steps = 0

        pbar = tqdm(
            total=total_batches,
            desc=f"Epoch {epoch}/{self.num_epochs}",
            disable=not is_main_process(),
        )
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with torch.amp.autocast(
                "cuda",
                enabled=self.use_amp,
                dtype=self.amp_dtype,
            ):
                outputs = self.model(images)

            # --- NEW: Move outputs to FP32 for stable loss computation ---
            if self.use_amp:
                outputs = {k: v.float() for k, v in outputs.items()}

            # Compute loss outside of autocast
            with torch.amp.autocast("cuda", enabled=False):
                loss_dict = self.loss_fn(outputs, masks, epoch=epoch)

            if not torch.isfinite(loss_dict["total"]):
                logger.warning(
                    f"NaN loss at epoch {epoch}, batch {batch_idx}. Skipping bad batch."
                )
                self.optimizer.zero_grad()
                pbar.update(1)
                continue

            in_tail_group = remainder_batches > 0 and batch_idx >= (
                total_batches - remainder_batches
            )
            accum_divisor = (
                remainder_batches if in_tail_group else self.grad_accum_steps
            )
            scaled_loss = loss_dict["total"] / max(accum_divisor, 1)

            self.scaler.scale(scaled_loss).backward()
            should_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or (
                (batch_idx + 1) == total_batches
            )
            if should_step:
                self.scaler.unscale_(self.optimizer)

                # --- NEW: Check for non-finite gradients before stepping ---
                grads_are_finite = True
                first_bad_param_name: Optional[str] = None
                first_bad_param_nan_count = 0
                first_bad_param_inf_count = 0
                bad_param_count = 0
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        continue
                    grad_is_finite = torch.isfinite(param.grad)
                    if not bool(grad_is_finite.all()):
                        grads_are_finite = False
                        bad_param_count += 1
                        if first_bad_param_name is None:
                            first_bad_param_name = name
                            first_bad_param_nan_count = int(
                                torch.isnan(param.grad).sum().item()
                            )
                            first_bad_param_inf_count = int(
                                torch.isinf(param.grad).sum().item()
                            )

                if grads_are_finite:
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_norm,
                            error_if_nonfinite=True,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.global_step += 1
                    optimizer_steps_applied += 1
                    if self.model_ema is not None:
                        self.model_ema.update(self.model, self.global_step)
                else:
                    skipped_optimizer_steps += 1
                    logger.warning(
                        "Non-finite gradients detected at epoch %d, batch %d. "
                        "Skipping optimizer step. bad_param_count=%d first_bad_param=%s nan=%d inf=%d",
                        epoch,
                        batch_idx,
                        bad_param_count,
                        first_bad_param_name,
                        first_bad_param_nan_count,
                        first_bad_param_inf_count,
                    )
                    if self.scaler.is_enabled():
                        self.scaler.update()

                self.optimizer.zero_grad()

            total_metrics["loss_total"] += float(loss_dict["total"].detach())
            total_metrics["loss_ce"] += float(loss_dict["ce"])
            total_metrics["loss_dice"] += float(loss_dict["dice"])
            total_metrics["loss_jaccard"] += float(loss_dict["jaccard"])
            total_metrics["loss_focal"] += float(loss_dict["focal"])
            total_metrics["loss_lovasz"] += float(loss_dict.get("lovasz", 0.0))
            total_metrics["loss_boundary"] += float(loss_dict["boundary"])
            total_metrics["loss_confusion"] += float(loss_dict["confusion_penalty"])
            total_metrics["loss_aux_s16"] += float(loss_dict.get("aux_s16", 0.0))
            total_metrics["loss_aux_s8"] += float(loss_dict.get("aux_s8", 0.0))
            seen_batches += 1
            pbar.set_postfix({"loss": f"{float(loss_dict['total'].detach()):.4f}"})
            pbar.update(1)
        pbar.close()

        if self.is_distributed and self.world_size > 1:
            batch_count_tensor = torch.tensor(float(seen_batches), device=self.device)
            dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)
            global_batch_count = max(int(batch_count_tensor.item()), 1)
            reduced_totals = {}
            for key, value in total_metrics.items():
                tensor = torch.tensor(float(value), device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                reduced_totals[key] = float(tensor.item())
            total_metrics = reduced_totals
            seen_batches = global_batch_count

        avg_metrics = {k: v / max(seen_batches, 1) for k, v in total_metrics.items()}
        avg_metrics["optimizer_steps"] = float(optimizer_steps_applied)
        avg_metrics["skipped_optimizer_steps"] = float(skipped_optimizer_steps)
        self._last_train_perf_metrics = {
            "perf/train_epoch_seconds": time.perf_counter() - epoch_start,
            **self._cuda_memory_metrics("train"),
        }
        return avg_metrics

    @torch.no_grad()
    def validate(self, epoch: int, profile: Optional[str] = None) -> Dict[str, Any]:
        self.model.eval()
        self.loss_fn.eval()
        self._reset_cuda_peak_memory()

        # Use raw model for validation inference to avoid compiler-only graph tiling failures.
        inference_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        for _ in range(8):
            orig_mod = getattr(inference_model, "_orig_mod", None)
            if orig_mod is None or orig_mod is inference_model:
                break
            inference_model = orig_mod

        use_ema = bool(
            self.model_ema is not None
            and self.use_ema_for_validation
            and self.model_ema.num_updates >= self.model_ema.update_after_step
        )
        validation_ctx = (
            self.model_ema.average_parameters(self.model)
            if use_ema
            else contextlib.nullcontext()
        )

        if profile is None:
            profile = self.primary_validation_profile
        pipeline = InferencePipeline(self.config, profile=profile)
        metrics = SegmentationMetrics(num_classes=5, class_names=CLASS_NAMES)
        val_start = time.perf_counter()
        inference_seconds = 0.0
        metric_update_seconds = 0.0
        val_perf_metrics: dict[str, float] = {}

        with validation_ctx:
            pbar = tqdm(
                total=len(self.val_loader),
                desc=f"Validation ({profile}) epoch {epoch}",
                leave=False,
                disable=not is_main_process(),
            )
            for batch in self.val_loader:
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"]

                infer_start = time.perf_counter()
                pred_map = pipeline.predict_segmentation_map(
                    inference_model,
                    images,
                    target_size=tuple(masks.shape[-2:]),
                )
                inference_seconds += time.perf_counter() - infer_start

                metrics_start = time.perf_counter()
                metrics.update(
                    torch.from_numpy(pred_map.astype("int64")).unsqueeze(0), masks
                )
                metric_update_seconds += time.perf_counter() - metrics_start
                pbar.update(1)
            pbar.close()

            result = metrics.compute()
            if self.is_distributed and self.world_size > 1:
                confusion_tensor = torch.from_numpy(result["confusion_matrix"]).to(
                    self.device, dtype=torch.int64
                )
                dist.all_reduce(confusion_tensor, op=dist.ReduceOp.SUM)
                result = metrics.compute_from_confusion_matrix(
                    confusion_tensor.cpu().numpy()
                )

            total_val_seconds = time.perf_counter() - val_start
            val_perf_metrics[f"perf/val_{profile}_epoch_seconds"] = total_val_seconds
            val_perf_metrics[f"perf/val_{profile}_inference_seconds"] = (
                inference_seconds
            )
            val_perf_metrics[f"perf/val_{profile}_metrics_seconds"] = (
                metric_update_seconds
            )
            val_perf_metrics.update(self._cuda_memory_metrics(f"val_{profile}"))

            if is_main_process():
                source = "EMA" if use_ema else "raw"
                logger.info(
                    "Validation (%s) epoch %d used %s weights. Mean IoU: %.2f%%",
                    profile,
                    epoch,
                    source,
                    float(result["mean_iou"]) * 100.0,
                )
                logger.info(
                    "Validation (%s) timing: total=%.2fs, inference=%.2fs, metrics=%.2fs",
                    profile,
                    total_val_seconds,
                    inference_seconds,
                    metric_update_seconds,
                )
                for name in CLASS_NAMES:
                    logger.info(
                        "[%s] %s IoU: %.2f%%",
                        profile,
                        name,
                        float(result["class_iou"][name]) * 100.0,
                    )

        self._last_val_metrics = {
            **{f"{name}_iou": float(result["class_iou"][name]) for name in CLASS_NAMES},
            "mean_iou": float(result["mean_iou"]),
        }
        self._last_val_metrics_by_profile = {profile: dict(self._last_val_metrics)}
        self._last_val_perf_metrics = val_perf_metrics

        self.model.train()
        self.loss_fn.train()
        return result
