"""Core two-dimensional slicing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import xarray as xr

from hyperslice.exceptions import SliceError
from hyperslice.interpolation import (
    ExactGridInterpolator,
    Interpolator,
    LinearRectilinearInterpolator,
    NearestInterpolator,
)

SliceMethod = Literal["exact", "nearest", "linear"]
INTERPOLATORS: dict[str, Interpolator] = {
    "exact": ExactGridInterpolator(),
    "nearest": NearestInterpolator(),
    "linear": LinearRectilinearInterpolator(),
}


def make_slice(
    data: xr.DataArray,
    *,
    x_dim: str,
    y_dim: str,
    selections: Mapping[str, Any],
    method: SliceMethod = "exact",
) -> xr.DataArray:
    """Select/interpolate all fixed dimensions and return ``(y_dim, x_dim)``."""
    if x_dim == y_dim:
        raise SliceError("x_dim and y_dim must be different.")
    missing = sorted({x_dim, y_dim} - set(data.dims))
    if missing:
        raise SliceError(f"Selected dimensions are not present in variable: {missing}")
    fixed_dims = [dim for dim in data.dims if dim not in {x_dim, y_dim}]
    absent = [dim for dim in fixed_dims if dim not in selections]
    if absent:
        raise SliceError(f"Selections are required for all non-displayed dimensions: {absent}")
    fixed = {dim: selections[dim] for dim in fixed_dims}
    interpolator = INTERPOLATORS.get(method)
    if interpolator is None:
        raise SliceError(f"Unsupported slicing method: {method}")
    try:
        result = interpolator.evaluate(data, x_dim=x_dim, y_dim=y_dim, selections=fixed)
    except SliceError:
        raise
    except (KeyError, ValueError, TypeError) as exc:
        raise SliceError(f"Could not make {method} slice: {exc}") from exc
    # Only scalar coordinates are dropped; singleton displayed axes must survive.
    unexpected = [dim for dim in result.dims if dim not in {x_dim, y_dim}]
    if unexpected:
        raise SliceError(f"Slice contains unexpected dimensions: {unexpected}")
    if x_dim not in result.dims or y_dim not in result.dims:
        raise SliceError("A displayed axis was unexpectedly reduced to a scalar.")
    result = result.transpose(y_dim, x_dim)
    result.attrs = dict(data.attrs)
    result.attrs["hyperslice_method"] = method
    return result
