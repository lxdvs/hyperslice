"""Strict linear interpolation for rectilinear coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from hyperslice.exceptions import SliceError
from hyperslice.interpolation.base import Interpolator


class LinearRectilinearInterpolator(Interpolator):
    """Linearly interpolate numeric fixed dimensions without extrapolation."""

    name = "linear"
    interpolated = True

    def evaluate(
        self, data: xr.DataArray, *, x_dim: str, y_dim: str, selections: Mapping[str, Any]
    ) -> xr.DataArray:
        exact: dict[str, Any] = {}
        linear: dict[str, Any] = {}
        working = data
        for dim, value in selections.items():
            coord = working.coords[dim]
            if coord.dtype.kind in "OUSb":
                exact[dim] = value
                continue
            values = np.asarray(coord.values)
            if len(values) > 1:
                delta = np.diff(values)
                if not (np.all(delta > 0) or np.all(delta < 0)):
                    raise SliceError(f"Linear interpolation requires monotonic coordinate '{dim}'.")
            low, high = np.min(values), np.max(values)
            if value < low or value > high:
                raise SliceError(
                    f"Selection {dim}={value!r} lies outside the coordinate domain [{low}, {high}]."
                )
            if np.any(values == value):
                exact[dim] = value
            else:
                linear[dim] = value
        if exact:
            working = working.sel(exact)
        # xarray supports descending axes, but sorting makes behavior consistent across backends.
        for dim in linear:
            values = np.asarray(working.coords[dim].values)
            if len(values) > 1 and values[0] > values[-1]:
                working = working.sortby(dim)
        return working.interp(linear, method="linear")
