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


def _constant_dataset() -> xr.Dataset:
    """Two varying outputs, one fixed output, one fixed output with gaps."""
    return xr.Dataset(
        {
            "varies": (("a", "b"), np.arange(6.0).reshape(2, 3)),
            "fixed": (("a", "b"), np.full((2, 3), 7.0)),
            "fixed_with_gaps": (("a", "b"), np.array([[2.0, np.nan, 2.0], [2.0, 2.0, np.nan]])),
            "all_missing": (("a", "b"), np.full((2, 3), np.nan)),
        },
        coords={"a": [0, 1], "b": [1, 2, 3]},
    )


def test_constant_variables_are_annotated() -> None:
    schema = inspect_dataset(_constant_dataset())
    assert schema.variables["fixed"].constant is True
    assert schema.variables["fixed"].constant_value == 7.0
    assert schema.variables["varies"].constant is False
    assert schema.variables["varies"].constant_value is None


def test_gapped_and_empty_variables_are_not_constant() -> None:
    schema = inspect_dataset(_constant_dataset())
    # One distinct value, but the present/missing split is still filterable.
    assert schema.variables["fixed_with_gaps"].constant is False
    # No values at all is not a value that never varies.
    assert schema.variables["all_missing"].constant is False


def test_single_valued_coordinate_is_annotated() -> None:
    dataset = xr.Dataset(
        {"varies": (("a", "b", "c"), np.arange(6.0).reshape(1, 2, 3))},
        coords={"a": [0], "b": [1, 2], "c": [4, 5, 6]},
    )
    schema = inspect_dataset(dataset)
    assert schema.coordinates["a"].constant is True
    assert schema.coordinates["a"].constant_value == 0
    assert schema.coordinates["b"].constant is False


def test_high_cardinality_outputs_are_annotated() -> None:
    dataset = xr.Dataset(
        {
            "independent": (("a", "b"), np.arange(6.0).reshape(2, 3)),
            "bucketed": (("a", "b"), np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])),
        },
        coords={"a": [0, 1], "b": [1, 2, 3]},
    )
    schema = inspect_dataset(dataset)
    independent = schema.variables["independent"]
    assert independent.high_cardinality is True
    assert independent.distinct_count == 6
    assert independent.distinct_ratio == 1.0
    bucketed = schema.variables["bucketed"]
    assert bucketed.high_cardinality is False
    assert bucketed.distinct_count == 2
    assert bucketed.distinct_ratio == pytest.approx(1 / 3)


def test_constant_output_is_never_high_cardinality() -> None:
    schema = inspect_dataset(_constant_dataset())
    assert schema.variables["fixed"].high_cardinality is False
    assert schema.variables["fixed"].distinct_count == 1
    assert schema.variables["all_missing"].distinct_count == 0
    assert schema.variables["all_missing"].high_cardinality is False
