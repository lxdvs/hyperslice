from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hyperslice.exceptions import SliceError
from hyperslice.interpolation import linear_contour_surface


def test_complete_irregular_surface_is_linearly_interpolated_exactly() -> None:
    x = np.array([0.0, 0.5, 2.0, 4.5, 8.0])
    y = np.array([-2.0, -0.5, 1.0, 3.0])
    source_y, source_x = np.meshgrid(y, x, indexing="ij")
    data = xr.DataArray(
        3 * source_x - 2 * source_y + 4,
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name="response",
    )
    result = linear_contour_surface(data, x_dim="x", y_dim="y", resolution=50)
    expected = 3 * result.x - 2 * result.y + 4
    xr.testing.assert_allclose(result, expected.transpose("y", "x"))
    assert result.attrs["hyperslice_interpolation"] == "rectilinear linear"
    assert result.attrs["hyperslice_extrapolated"] is False


def test_incomplete_surface_reconstructs_isolated_missing_grid_point() -> None:
    x = np.array([0.0, 0.5, 2.0, 4.5, 8.0])
    y = np.array([-2.0, -0.5, 1.0, 3.0, 6.0])
    source_y, source_x = np.meshgrid(y, x, indexing="ij")
    values = 3 * source_x - 2 * source_y + 4
    values[2, 2] = np.nan
    data = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name="response",
    )
    result = linear_contour_surface(data, x_dim="x", y_dim="y", resolution=40)
    expected = 3 * result.x - 2 * result.y + 4
    xr.testing.assert_allclose(result, expected.transpose("y", "x"))
    assert result.attrs["hyperslice_missing_samples"] == 1
    assert "incomplete rectilinear support" in result.attrs["hyperslice_interpolation"]


def test_incomplete_surface_does_not_extrapolate_outside_convex_hull() -> None:
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape(5, 5),
        dims=("y", "x"),
        coords={"x": range(5), "y": range(5)},
    )
    data[0, :] = np.nan
    result = linear_contour_surface(data, x_dim="x", y_dim="y", resolution=30)
    assert result.isel(y=0).isnull().all()
    assert result.attrs["hyperslice_extrapolated"] is False


def test_descending_axes_are_supported() -> None:
    data = xr.DataArray(
        np.arange(20, dtype=float).reshape(4, 5),
        dims=("y", "x"),
        coords={"x": [8.0, 4.5, 2.0, 0.5, 0.0], "y": [3.0, 1.0, -0.5, -2.0]},
    )
    result = linear_contour_surface(data, x_dim="x", y_dim="y", resolution=20)
    assert np.all(np.diff(result.x) > 0)
    assert np.all(np.diff(result.y) > 0)
    assert np.isfinite(result).all()


def test_semantically_invalid_cell_is_not_bridged() -> None:
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape(5, 5),
        dims=("y", "x"),
        coords={"x": range(5), "y": range(5)},
    )
    invalid = xr.zeros_like(data, dtype=bool)
    invalid[2, 2] = True
    with pytest.raises(SliceError, match="semantically invalid"):
        linear_contour_surface(data, x_dim="x", y_dim="y", invalid_mask=invalid)


def test_insufficient_support_is_rejected() -> None:
    data = xr.DataArray(
        [[1.0, np.nan], [np.nan, 2.0]],
        dims=("y", "x"),
        coords={"x": [0, 1], "y": [0, 1]},
    )
    with pytest.raises(SliceError, match="at least 3 finite samples"):
        linear_contour_surface(data, x_dim="x", y_dim="y")
