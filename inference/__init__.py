from importlib import import_module
from typing import Any

__all__ = [
    "Evaluator",
    "TestTimeAugmentation",
    "MultiScaleInference",
]

_EXPORTS = {
    "Evaluator": "inference.evaluator",
    "TestTimeAugmentation": "inference.tta",
    "MultiScaleInference": "inference.multiscale",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'inference' has no attribute {name!r}")

    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
