from importlib import import_module
from typing import Any

__all__ = [
    "OilSpillDataset",
    "SARSegmentationAugmentation",
    "CurriculumDataLoaderFactory",
    "SARFeatureEncoder",
    "CopyPasteAugmentation",
]

_EXPORTS = {
    "OilSpillDataset": "data.dataset",
    "SARSegmentationAugmentation": "data.augmentation",
    "CurriculumDataLoaderFactory": "data.dataloader",
    "SARFeatureEncoder": "data.sar_features",
    "CopyPasteAugmentation": "data.copy_paste",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'data' has no attribute {name!r}")

    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
