"""Nearest-neighbor fixed-coordinate selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from hyperslice.interpolation.base import Interpolator


class NearestInterpolator(Interpolator):
    """Select the nearest existing coordinate on numeric monotonic axes."""

    name = "nearest"

    def evaluate(
        self, data: xr.DataArray, *, x_dim: str, y_dim: str, selections: Mapping[str, Any]
    ) -> xr.DataArray:
        numeric = {
            key: value
            for key, value in selections.items()
            if data.coords[key].dtype.kind not in "OUSb"
        }
        exact = {key: value for key, value in selections.items() if key not in numeric}
        return data.sel(exact).sel(numeric, method="nearest")
