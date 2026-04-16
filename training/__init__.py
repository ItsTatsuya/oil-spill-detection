from importlib import import_module
from typing import Any

__all__ = [
    "Trainer",
    "SegmentationMetrics",
    "CurriculumScheduler",
    "CheckpointCallback",
    "build_train_val_indices",
]

_EXPORTS = {
    "Trainer": "training.trainer",
    "SegmentationMetrics": "training.metrics",
    "CurriculumScheduler": "training.curriculum",
    "CheckpointCallback": "training.callbacks",
    "build_train_val_indices": "training.split",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'training' has no attribute {name!r}")

    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
