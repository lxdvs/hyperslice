"""Linear interpolation surfaces for response-value contours."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

from hyperslice.exceptions import SliceError


def linear_contour_surface(
    data: xr.DataArray,
    *,
    x_dim: str,
    y_dim: str,
    resolution: int = 240,
    invalid_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Evaluate a linear response surface without extrapolation.

    Complete slices use rectilinear multilinear interpolation. Incomplete
    slices use a Delaunay-based linear interpolator over finite samples and
    remain missing outside the convex hull of that support.
    """
    if resolution < 2:
        raise ValueError("Contour resolution must be at least 2.")
    x = np.asarray(data.coords[x_dim].values)
    y = np.asarray(data.coords[y_dim].values)
    if x.dtype.kind not in "iuf" or y.dtype.kind not in "iuf":
        raise SliceError("Pareto contours require numeric X and Y coordinates.")
    if len(x) < 2 or len(y) < 2:
        raise SliceError("Pareto contours require at least 2 coordinate values on each axis.")
    values = np.asarray(data.transpose(y_dim, x_dim).values, dtype=float)
    if invalid_mask is not None:
        semantic_invalid = np.asarray(invalid_mask.transpose(y_dim, x_dim).values, dtype=bool)
        if semantic_invalid.any():
            raise SliceError(
                "Pareto contour unavailable: semantically invalid cells cannot "
                "be used as interpolation support."
            )
    else:
        semantic_invalid = np.zeros_like(values, dtype=bool)
    finite = np.isfinite(values) & ~semantic_invalid
    if not finite.any():
        raise SliceError("Pareto contour unavailable: the selected slice has no finite support.")
    if np.all(np.diff(x) < 0):
        x = x[::-1]
        values = values[:, ::-1]
        finite = finite[:, ::-1]
    elif not np.all(np.diff(x) > 0):
        raise SliceError("Pareto contours require a strictly monotonic X coordinate.")
    if np.all(np.diff(y) < 0):
        y = y[::-1]
        values = values[::-1, :]
        finite = finite[::-1, :]
    elif not np.all(np.diff(y) > 0):
        raise SliceError("Pareto contours require a strictly monotonic Y coordinate.")
    dense_x = np.linspace(float(x[0]), float(x[-1]), resolution)
    dense_y = np.linspace(float(y[0]), float(y[-1]), resolution)
    query_x, query_y = np.meshgrid(dense_x, dense_y)
    if finite.all():
        interpolator = RegularGridInterpolator(
            (y, x),
            values,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        dense_values = interpolator(np.column_stack((query_y.ravel(), query_x.ravel()))).reshape(
            query_x.shape
        )
        interpolation = "rectilinear linear"
    else:
        source_x, source_y = np.meshgrid(x, y)
        if (
            np.unique(source_x[finite]).size < 2
            or np.unique(source_y[finite]).size < 2
            or int(finite.sum()) < 3
        ):
            raise SliceError(
                "Pareto contour unavailable: incomplete support requires at "
                "least 3 finite samples spanning 2 coordinates on each axis."
            )
        interpolator = LinearNDInterpolator(
            np.column_stack((source_x[finite], source_y[finite])),
            values[finite],
            fill_value=np.nan,
        )
        dense_values = interpolator(query_x, query_y)
        interpolation = "scattered linear over incomplete rectilinear support"
    result = xr.DataArray(
        dense_values,
        dims=(y_dim, x_dim),
        coords={y_dim: dense_y, x_dim: dense_x},
        name=data.name,
        attrs=dict(data.attrs),
    )
    result.attrs["hyperslice_interpolation"] = interpolation
    result.attrs["hyperslice_support_samples"] = int(finite.sum())
    result.attrs["hyperslice_missing_samples"] = int((~finite).sum())
    result.attrs["hyperslice_extrapolated"] = False
    return result
