from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import SizePrediction


def predict_size_from_path(*args: Any, **kwargs: Any):
    from .pipeline import predict_size_from_path as _predict_size_from_path

    return _predict_size_from_path(*args, **kwargs)


__all__ = ["predict_size_from_path", "SizePrediction"]
