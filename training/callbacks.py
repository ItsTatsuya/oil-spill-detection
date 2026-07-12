import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelEMA:
    """EMA of model weights. Shadow buffers live on CPU to save VRAM (8GB GPUs)."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9998,
        update_after_step: int = 200,
        device: str | torch.device = "cpu",
        update_every: int = 1,
    ) -> None:
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.update_every = max(int(update_every), 1)
        self.num_updates = 0
        self.device = torch.device(device)
        model_to_track = self._unwrap_model(model)
        self.shadow_state = {
            name: tensor.detach().to(self.device, copy=True)
            for name, tensor in model_to_track.state_dict().items()
        }
        logger.info(
            "ModelEMA shadow on %s (update_every=%d).",
            self.device,
            self.update_every,
        )

    def update(self, model: nn.Module, global_step: int) -> bool:
        self.num_updates = int(global_step)
        if self.num_updates < self.update_after_step:
            return False
        # Skip most steps when EMA lives on CPU — full state copies dominate step time.
        if self.update_every > 1 and (self.num_updates % self.update_every) != 0:
            return False

        current_state = self._unwrap_model(model).state_dict()
        # Slightly higher effective decay when updating less often so EMA stays smooth.
        decay = self.decay
        if self.update_every > 1:
            decay = float(decay ** self.update_every)
        for name, tensor in current_state.items():
            shadow = self.shadow_state[name]
            src = tensor.detach().to(shadow.device, non_blocking=False)
            if torch.is_floating_point(tensor):
                shadow.mul_(decay).add_(src, alpha=1.0 - decay)
            else:
                shadow.copy_(src)
        return True

    @contextmanager
    def average_parameters(self, model: nn.Module):
        """Load EMA weights for validation, restore live weights after.

        Live weights are backed up on CPU so we do not hold two full GPU copies.
        """
        model_to_use = self._unwrap_model(model)
        live_state = model_to_use.state_dict()
        raw_state = {
            name: tensor.detach().to("cpu", copy=True)
            for name, tensor in live_state.items()
        }
        ema_on_device = {
            name: self.shadow_state[name].to(
                device=param.device, dtype=param.dtype, non_blocking=False
            )
            for name, param in live_state.items()
            if name in self.shadow_state
        }
        model_to_use.load_state_dict(ema_on_device, strict=True)
        try:
            yield
        finally:
            restore = {
                name: tensor.to(
                    device=live_state[name].device, dtype=live_state[name].dtype
                )
                for name, tensor in raw_state.items()
            }
            model_to_use.load_state_dict(restore, strict=True)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "num_updates": self.num_updates,
            "shadow_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.shadow_state.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.update_after_step = int(
            state_dict.get("update_after_step", self.update_after_step)
        )
        self.num_updates = int(state_dict.get("num_updates", self.num_updates))
        for name, tensor in state_dict.get("shadow_state_dict", {}).items():
            if name in self.shadow_state:
                self.shadow_state[name].copy_(
                    tensor.to(device=self.shadow_state[name].device)
                )

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model


class CheckpointCallback:
    def __init__(
        self,
        checkpoint_dir: str,
        save_every_n_epochs: int = 10,
        keep_top_k_ema_checkpoints: int = 0,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = save_every_n_epochs
        self.best_val_miou = 0.0
        self.keep_top_k_ema_checkpoints = max(int(keep_top_k_ema_checkpoints), 0)
        self._top_k_ema_records = self._load_top_k_ema_records()

    def _top_k_ema_path(self, epoch: int) -> Path:
        return self.checkpoint_dir / f"topk_ema_epoch_{epoch:04d}.pth"

    def _load_top_k_ema_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if self.keep_top_k_ema_checkpoints <= 0:
            return records

        for path in sorted(self.checkpoint_dir.glob("topk_ema_epoch_*.pth")):
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:  # pragma: no cover - best effort recovery path
                logger.warning(
                    "Skipping unreadable top-k EMA checkpoint %s: %s", path, exc
                )
                continue
            records.append(
                {
                    "epoch": int(checkpoint.get("epoch", 0)),
                    "val_miou": float(checkpoint.get("val_miou", 0.0)),
                    "path": path,
                }
            )
        records.sort(key=lambda item: (item["val_miou"], item["epoch"]), reverse=True)
        return records[: self.keep_top_k_ema_checkpoints]

    def _save_top_k_ema_checkpoint(
        self,
        *,
        checkpoint: Dict[str, Any],
        ema_state: Dict[str, Any],
        epoch: int,
        val_miou: float,
    ) -> None:
        if self.keep_top_k_ema_checkpoints <= 0:
            return

        candidate_path = self._top_k_ema_path(epoch)
        self._top_k_ema_records = [
            item for item in self._top_k_ema_records if item["path"] != candidate_path
        ]
        should_save = len(self._top_k_ema_records) < self.keep_top_k_ema_checkpoints
        if not should_save and self._top_k_ema_records:
            threshold = min(self._top_k_ema_records, key=lambda item: item["val_miou"])
            should_save = val_miou > float(threshold["val_miou"])
        if not should_save:
            return

        ema_checkpoint = dict(checkpoint)
        ema_checkpoint["model_state_dict"] = {
            name: tensor.detach().clone()
            for name, tensor in ema_state["shadow_state_dict"].items()
        }
        tmp_candidate = candidate_path.with_suffix(".tmp")
        torch.save(ema_checkpoint, tmp_candidate)
        tmp_candidate.replace(candidate_path)

        self._top_k_ema_records.append(
            {"epoch": int(epoch), "val_miou": float(val_miou), "path": candidate_path}
        )
        self._top_k_ema_records.sort(
            key=lambda item: (item["val_miou"], item["epoch"]),
            reverse=True,
        )

        while len(self._top_k_ema_records) > self.keep_top_k_ema_checkpoints:
            removed = self._top_k_ema_records.pop()
            try:
                removed["path"].unlink(missing_ok=True)
            except TypeError:  # pragma: no cover - Python <3.8 compatibility
                if removed["path"].exists():
                    removed["path"].unlink()

        logger.info(
            "Updated top-%d EMA checkpoints with epoch %d (mIoU=%.2f%%)",
            self.keep_top_k_ema_checkpoints,
            epoch,
            val_miou * 100.0,
        )

    def save(
        self,
        epoch: int,
        val_miou: float,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any,
        config: dict,
        is_best: bool = False,
        rank: int = 0,
        curriculum: Optional[Any] = None,
        scaler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        model_ema: Optional[ModelEMA] = None,
        global_step: Optional[int] = None,
        save_best_ema: bool = True,
        val_metrics_by_profile: Optional[Dict[str, Dict[str, float]]] = None,
        primary_validation_profile: Optional[str] = None,
        trainer_state: Optional[Dict[str, Any]] = None,
        track_top_k_ema: bool = True,
    ) -> None:
        if rank != 0:
            return

        model_to_save = model.module if hasattr(model, "module") else model

        checkpoint = {
            "epoch": epoch,
            "val_miou": val_miou,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None
            else None,
            "config": config,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        }
        if global_step is not None:
            checkpoint["global_step"] = int(global_step)

        if val_metrics_by_profile is not None:
            checkpoint["val_metrics_by_profile"] = val_metrics_by_profile
        if primary_validation_profile is not None:
            checkpoint["primary_validation_profile"] = primary_validation_profile
        if trainer_state is not None:
            checkpoint["trainer_state"] = trainer_state

        if curriculum is not None:
            checkpoint["curriculum_applied_transitions"] = (
                curriculum.get_applied_transitions()
            )

        if loss_fn is not None:
            checkpoint["loss_fn_state_dict"] = loss_fn.state_dict()

        ema_state = None
        if model_ema is not None:
            ema_state = model_ema.state_dict()
            checkpoint["ema_state_dict"] = ema_state
            if track_top_k_ema:
                self._save_top_k_ema_checkpoint(
                    checkpoint=checkpoint,
                    ema_state=ema_state,
                    epoch=epoch,
                    val_miou=val_miou,
                )

        latest_path = self.checkpoint_dir / "latest.pth"
        tmp_path = latest_path.with_suffix(".tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(latest_path)

        if epoch % self.save_every_n_epochs == 0:
            numbered_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pth"
            tmp_numbered = numbered_path.with_suffix(".tmp")
            torch.save(checkpoint, tmp_numbered)
            tmp_numbered.replace(numbered_path)
            logger.info(f"Checkpoint saved: {numbered_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            tmp_best = best_path.with_suffix(".tmp")
            torch.save(checkpoint, tmp_best)
            tmp_best.replace(best_path)
            self.best_val_miou = val_miou
            logger.info(
                f"New best model saved: mIoU={val_miou * 100:.2f}% -> {best_path}"
            )
            if ema_state is not None and save_best_ema:
                ema_checkpoint = dict(checkpoint)
                ema_checkpoint["model_state_dict"] = {
                    name: tensor.detach().clone()
                    for name, tensor in ema_state["shadow_state_dict"].items()
                }
                best_ema_path = self.checkpoint_dir / "best_ema.pth"
                tmp_best_ema = best_ema_path.with_suffix(".tmp")
                torch.save(ema_checkpoint, tmp_best_ema)
                tmp_best_ema.replace(best_ema_path)
                logger.info(
                    "New best EMA model saved: mIoU=%.2f%% -> %s",
                    val_miou * 100.0,
                    best_ema_path,
                )

    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        current_config: Optional[dict] = None,
        curriculum: Optional[Any] = None,
        scaler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        model_ema: Optional[ModelEMA] = None,
        ignore_loss_state: bool = False,
    ) -> Dict[str, Any]:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (
            scheduler is not None
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"] is not None
        ):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if (
            scaler is not None
            and "scaler_state_dict" in checkpoint
            and checkpoint["scaler_state_dict"] is not None
        ):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        checkpoint_config = checkpoint.get("config", {})
        if current_config is not None:
            drift = self._collect_config_drift(checkpoint_config, current_config)
            if drift:
                preview = "\n".join(f"  - {entry}" for entry in drift[:10])
                extra = ""
                if len(drift) > 10:
                    extra = f"\n  ... and {len(drift) - 10} more differences"
                logger.warning(
                    "Checkpoint config differs from current config. "
                    "State was restored from the checkpoint, so review these changes:\n"
                    f"{preview}{extra}"
                )

        epoch = checkpoint.get("epoch", 0)
        val_miou = checkpoint.get("val_miou", 0.0)

        if "curriculum_applied_transitions" in checkpoint and curriculum is not None:
            curriculum.restore_applied_transitions(
                checkpoint["curriculum_applied_transitions"]
            )

        if loss_fn is not None and "loss_fn_state_dict" in checkpoint:
            if ignore_loss_state:
                logger.info(
                    "Ignored CombinedLoss state from checkpoint due to --ignore-loss-state flag. "
                    "Using YAML config."
                )
            else:
                loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])
                logger.info("Restored CombinedLoss state from checkpoint.")

        if model_ema is not None and "ema_state_dict" in checkpoint:
            model_ema.load_state_dict(checkpoint["ema_state_dict"])
            logger.info("Restored model EMA state from checkpoint.")

        logger.info(f"Loaded checkpoint: epoch={epoch}, val_mIoU={val_miou * 100:.2f}%")

        return {
            "epoch": epoch,
            "val_miou": val_miou,
            "config": checkpoint_config,
            "global_step": int(checkpoint.get("global_step", 0)),
            "val_metrics_by_profile": checkpoint.get("val_metrics_by_profile", {}),
            "primary_validation_profile": checkpoint.get("primary_validation_profile"),
            "trainer_state": checkpoint.get("trainer_state", {}),
        }

    def _collect_config_drift(
        self,
        checkpoint_config: Any,
        current_config: Any,
        prefix: str = "",
    ) -> List[str]:
        drift: List[str] = []

        if isinstance(checkpoint_config, dict) and isinstance(current_config, dict):
            all_keys = sorted(set(checkpoint_config) | set(current_config))
            for key in all_keys:
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                if key not in checkpoint_config:
                    drift.append(
                        f"{next_prefix}: missing in checkpoint, current={current_config[key]!r}"
                    )
                elif key not in current_config:
                    drift.append(
                        f"{next_prefix}: checkpoint={checkpoint_config[key]!r}, missing in current config"
                    )
                else:
                    drift.extend(
                        self._collect_config_drift(
                            checkpoint_config[key], current_config[key], next_prefix
                        )
                    )
            return drift

        if checkpoint_config != current_config:
            drift.append(
                f"{prefix}: checkpoint={checkpoint_config!r}, current={current_config!r}"
            )

        return drift

    def find_latest(self) -> Optional[Path]:
        latest = self.checkpoint_dir / "latest.pth"
        if latest.exists():
            return latest
        return None
