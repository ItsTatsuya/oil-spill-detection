import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_logger = logging.getLogger(__name__)


class Logger:

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
    ) -> None:
        self.config = config
        if rank is None:
            rank = int(os.environ.get("RANK", 0))
        self.rank = rank
        self._step = 0
        self.run_id = self._resolve_run_id(config)
        self.console_log_mode = self._resolve_console_log_mode(config)

        if isinstance(config, dict):
            log_dir = config.get("training", {}).get("log_dir", "./logs")
        else:
            log_dir = getattr(config, "log_dir", None)
            if log_dir is None and hasattr(config, "training"):
                log_dir = config.training.get("log_dir", "./logs")
            elif log_dir is None:
                log_dir = "./logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        append_training_log = (
            bool(config.get("training", {}).get("append_training_log", False))
            if isinstance(config, dict)
            else False
        )
        self._training_log_file = self.log_dir / "training.log"
        if (
            self.rank == 0
            and self._training_log_file.exists()
            and not append_training_log
        ):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = self.log_dir / f"training_{ts}.log"
            self._training_log_file.replace(backup)
            _logger.info(f"Rotated existing training log to: {backup}")
        self._training_log_mode = "a" if append_training_log else "w"

        append_metrics_log = (
            bool(config.get("training", {}).get("append_metrics_log", False))
            if isinstance(config, dict)
            else False
        )
        self._metrics_file = self.log_dir / "metrics.jsonl"
        if self.rank == 0 and self._metrics_file.exists() and not append_metrics_log:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = self.log_dir / f"metrics_{ts}.jsonl"
            self._metrics_file.replace(backup)
            _logger.info(f"Rotated existing metrics log to: {backup}")
        self._metrics_log_mode = "a" if append_metrics_log else "w"

        self._setup_console_logging()

        self._metrics_fh = (
            open(self._metrics_file, self._metrics_log_mode, buffering=1)
            if self.rank == 0
            else None
        )

        if self.rank == 0:
            _logger.info(f"Logger initialized. Log dir: {self.log_dir}")
            _logger.info(f"Run ID: {self.run_id}")
            _logger.info(
                "Local logging enabled: console + training.log + metrics.jsonl."
            )

    def _resolve_run_id(self, config: Any) -> str:
        env_run_id = os.environ.get("RUN_ID")
        if env_run_id:
            return env_run_id

        if isinstance(config, dict):
            run_id = config.get("training", {}).get("run_id")
            if run_id:
                return str(run_id)

        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _resolve_console_log_mode(self, config: Any) -> str:
        return "verbose"

    def _setup_console_logging(self) -> None:
        fmt = logging.Formatter(
            fmt="[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        root.handlers = [
            h
            for h in root.handlers
            if not isinstance(h, logging.StreamHandler)
            or isinstance(h, logging.FileHandler)
        ]
        root.addHandler(ch)
        if self.rank == 0:
            existing_paths = {
                h.baseFilename
                for h in root.handlers
                if isinstance(h, logging.FileHandler)
            }
            log_path = str(self._training_log_file)
            if log_path not in existing_paths:
                fh = logging.FileHandler(
                    self._training_log_file, mode=self._training_log_mode
                )
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(fmt)
                root.addHandler(fh)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        if self.rank != 0:
            return

        if step is None:
            step = self._step
            self._step += 1
        else:
            self._step = step

        record = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "step": step,
            **metrics,
        }
        if self._metrics_fh is not None:
            self._metrics_fh.write(json.dumps(record) + "\n")

        def _format_metric_value(name: str, value: float) -> str:
            if "lr" in name:
                return f"{value:.8f}"
            return f"{value:.4f}"

        metric_str = " | ".join(
            f"{k}: {_format_metric_value(k, v)}"
            for k, v in metrics.items()
            if any(kw in k for kw in ["loss", "iou", "lr"])
        )
        if metric_str:
            _logger.debug(f"Step {step}: {metric_str}")

    def log_image(self, name: str, image_path: str, step: Optional[int] = None) -> None:
        if self.rank != 0:
            return

        _logger.info("Image artifact [%s] at step %s: %s", name, step, image_path)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        if self.rank != 0:
            return

        _logger.info("Hyperparameters:")
        for k, v in params.items():
            _logger.info(f"  {k}: {v}")

    def info(self, message: str) -> None:
        if self.rank == 0:
            _logger.info(message)

    def warning(self, message: str) -> None:
        _logger.warning(message)

    def error(self, message: str) -> None:
        _logger.error(message)

    def finish(self) -> None:
        if self._metrics_fh is not None and not self._metrics_fh.closed:
            self._metrics_fh.close()

        if self.rank == 0 and hasattr(self, "_training_log_file"):
            root = logging.getLogger()
            target = str(self._training_log_file)
            for handler in list(root.handlers):
                if (
                    isinstance(handler, logging.FileHandler)
                    and getattr(handler, "baseFilename", None) == target
                ):
                    root.removeHandler(handler)
                    handler.close()

    def __del__(self) -> None:
        try:
            if (
                hasattr(self, "_metrics_fh")
                and self._metrics_fh is not None
                and not self._metrics_fh.closed
            ):
                self._metrics_fh.close()
        except Exception:
            pass


def get_logger(name: str = __name__) -> logging.Logger:
    return logging.getLogger(name)
