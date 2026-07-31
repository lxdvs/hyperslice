"""Interpolation contracts and shared validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import xarray as xr


class Interpolator(ABC):
    """Evaluate fixed dimensions while retaining two displayed axes."""

    name: str
    interpolated: bool = False

    @abstractmethod
    def evaluate(
        self,
        data: xr.DataArray,
        *,
        x_dim: str,
        y_dim: str,
        selections: Mapping[str, Any],
    ) -> xr.DataArray:
        """Return a slice ordered as ``(y_dim, x_dim)``."""
