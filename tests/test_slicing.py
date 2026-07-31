from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hyperslice.exceptions import SliceError
from hyperslice.slicing import make_slice


def selections() -> dict[str, float]:
    return {"pressure": 2.5, "flow_rate": 5.0, "burnup": 5.0}


def test_exact_and_transpose(dataset: xr.Dataset) -> None:
    result = make_slice(
        dataset.k_eff,
        x_dim="drum_angle",
        y_dim="fuel_temperature",
        selections=selections(),
    )
    assert result.dims == ("fuel_temperature", "drum_angle")
    assert result.attrs["units"] == "dimensionless"


def test_nearest_uses_actual_coordinate(dataset: xr.Dataset) -> None:
    fixed = selections() | {"pressure": 4.2}
    result = make_slice(
        dataset.k_eff,
        x_dim="drum_angle",
        y_dim="fuel_temperature",
        selections=fixed,
        method="nearest",
    )
    assert result.pressure.item() == 5.0


def test_linear_and_descending_coordinate(dataset: xr.Dataset) -> None:
    data = dataset.k_eff.sortby("pressure", ascending=False)
    fixed = selections() | {"pressure": 4.2}
    result = make_slice(
        data,
        x_dim="drum_angle",
        y_dim="fuel_temperature",
        selections=fixed,
        method="linear",
    )
    expected = dataset.k_eff.sortby("pressure").interp(pressure=4.2).sel(flow_rate=5, burnup=5)
    xr.testing.assert_allclose(result, expected.transpose("fuel_temperature", "drum_angle"))


@pytest.mark.parametrize(
    ("x", "y", "fixed", "message"),
    [
        ("drum_angle", "drum_angle", selections(), "different"),
        ("bad", "fuel_temperature", selections(), "not present"),
        ("drum_angle", "fuel_temperature", {"pressure": 2.5}, "required"),
    ],
)
def test_invalid_requests(
    dataset: xr.Dataset, x: str, y: str, fixed: dict[str, float], message: str
) -> None:
    with pytest.raises(SliceError, match=message):
        make_slice(dataset.k_eff, x_dim=x, y_dim=y, selections=fixed)


def test_outside_linear_domain(dataset: xr.Dataset) -> None:
    with pytest.raises(SliceError, match="outside"):
        make_slice(
            dataset.k_eff,
            x_dim="drum_angle",
            y_dim="fuel_temperature",
            selections=selections() | {"pressure": 100},
            method="linear",
        )


def test_singleton_display_axis() -> None:
    data = xr.DataArray(
        np.ones((1, 2, 2)), dims=("a", "b", "c"), coords={"a": [1], "b": [2, 3], "c": [4, 5]}
    )
    result = make_slice(data, x_dim="a", y_dim="b", selections={"c": 4})
    assert result.shape == (2, 1)
