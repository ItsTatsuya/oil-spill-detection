from importlib import import_module
from typing import Any

__all__ = [
    "setup_distributed",
    "cleanup_distributed",
    "Logger",
    "load_config",
]

_EXPORTS = {
    "setup_distributed": "utils.distributed",
    "cleanup_distributed": "utils.distributed",
    "Logger": "utils.logger",
    "load_config": "utils.config",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'utils' has no attribute {name!r}")

    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
