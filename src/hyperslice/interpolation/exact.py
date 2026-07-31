"""Exact grid selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from hyperslice.interpolation.base import Interpolator


class ExactGridInterpolator(Interpolator):
    """Select existing coordinate values exactly."""

    name = "exact"

    def evaluate(
        self, data: xr.DataArray, *, x_dim: str, y_dim: str, selections: Mapping[str, Any]
    ) -> xr.DataArray:
        return data.sel(dict(selections))
