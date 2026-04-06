"""Unified exports for FaceGuard adapters (lazy import)."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MAP = {
    "VideoAdapter": (".VideoAdapter", "VideoAdapter"),
    "ModelAdapter": (".ModelAdapter", "ModelAdapter"),
    "StorageAdapter": (".StorageAdapter", "StorageAdapter"),
    "FineTuneAdapter": (".FineTuneAdapter", "FineTuneAdapter"),
    "FineTuneBatch": (".FineTuneAdapter", "FineTuneBatch"),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod_name, attr_name = _EXPORT_MAP[name]
    mod = import_module(mod_name, __name__)
    value = getattr(mod, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
