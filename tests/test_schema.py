from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hyperslice.exceptions import DatasetSchemaError
from hyperslice.schema import inspect_dataset


def test_schema_metadata_and_irregular_grid(dataset: xr.Dataset) -> None:
    schema = inspect_dataset(dataset)
    assert {"k_eff", "peak_temperature"} == set(schema.variables)
    assert schema.coordinates["fuel_temperature"].units == "K"
    assert schema.coordinates["fuel_temperature"].monotonic
    assert "status" in schema.status_candidates
    spacing = np.diff(dataset.fuel_temperature)
    assert not np.all(spacing == spacing[0])


def test_categorical_coordinate() -> None:
    ds = xr.Dataset(
        {"value": (("material", "x"), np.ones((2, 2)))},
        coords={"material": ["steel", "salt"], "x": [1, 2]},
    )
    assert inspect_dataset(ds).coordinates["material"].categorical


def test_no_plottable_variables() -> None:
    with pytest.raises(DatasetSchemaError, match="no plottable"):
        inspect_dataset(xr.Dataset({"profile": ("x", [1, 2])}, coords={"x": [1, 2]}))


def test_duplicate_coordinate() -> None:
    ds = xr.Dataset({"v": (("x", "y"), np.ones((2, 2)))}, coords={"x": [1, 1], "y": [1, 2]})
    with pytest.raises(DatasetSchemaError, match="duplicate"):
        inspect_dataset(ds)
