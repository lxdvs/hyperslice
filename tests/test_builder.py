from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hyperslice import (
    DatasetBuilder,
    DatasetBuilderError,
    InputSchemaMismatchError,
    InvalidParameterError,
    OutputOverwriteError,
)
from hyperslice.loading import load_dataset


def test_repeated_input_point_adds_distinct_outputs() -> None:
    builder = DatasetBuilder()
    inputs = {"temperature": 300.0, "configuration": "baseline"}
    assert builder.add_point(inputs, {"efficiency": 0.8}) is builder
    builder.add_point(inputs, {"margin": 2.5})
    dataset = builder.to_dataset()
    assert builder.point_count == 1
    assert builder.input_names == ("temperature", "configuration")
    assert builder.output_names == ("efficiency", "margin")
    assert dataset.efficiency.item() == 0.8
    assert dataset.margin.item() == 2.5
    assert dataset.temperature.item() == 300
    assert dataset.configuration.item() == "baseline"
    assert dataset.efficiency.dims == ()


def test_output_overwrite_is_rejected_even_for_same_value() -> None:
    builder = DatasetBuilder().add_point({"x": 1, "y": 2}, {"response": 3})
    with pytest.raises(OutputOverwriteError, match="response"):
        builder.add_point({"y": 2, "x": 1}, {"response": 3})
    assert builder.to_dataset().response.item() == 3


def test_input_key_schema_must_match_first_point() -> None:
    builder = DatasetBuilder().add_point({"x": 1, "y": 2}, {"response": 3})
    with pytest.raises(InputSchemaMismatchError, match=r"Missing: \['y'\].*unexpected: \['z'\]"):
        builder.add_point({"x": 2, "z": 4}, {"response": 5})


def test_materialization_builds_incomplete_cartesian_grid() -> None:
    builder = DatasetBuilder()
    builder.add_point({"x": 0, "configuration": "A"}, {"response": 1})
    builder.add_point({"x": 1, "configuration": "B"}, {"response": 2})
    builder.add_point({"x": 1, "configuration": "B"}, {"margin": 4})
    dataset = builder.to_dataset()
    assert dataset.response.shape == (2, 2)
    assert int(dataset.response.notnull().sum()) == 2
    assert int(dataset.response.isnull().sum()) == 2
    assert int(dataset.margin.notnull().sum()) == 1
    assert dataset.configuration.values.tolist() == ["A", "B"]
    assert dataset.attrs["hyperslice_builder_point_count"] == 2


def test_metadata_is_applied() -> None:
    builder = DatasetBuilder(
        coordinate_attrs={"x": {"long_name": "Position", "units": "m"}},
        variable_attrs={"response": {"long_name": "Response", "units": "Pa"}},
        attrs={"title": "Builder test"},
    )
    dataset = builder.add_point({"x": 1, "y": 2}, {"response": 3}).to_dataset()
    assert dataset.x.attrs == {"long_name": "Position", "units": "m"}
    assert dataset.response.attrs == {"long_name": "Response", "units": "Pa"}
    assert dataset.attrs["title"] == "Builder test"


def test_constant_inputs_become_scalar_coordinates_not_dimensions() -> None:
    builder = DatasetBuilder()
    builder.add_point(
        {"x": 0, "temperature": 600, "configuration": "baseline"},
        {"response": 1},
    )
    builder.add_point(
        {"x": 1, "temperature": 600, "configuration": "baseline"},
        {"response": 2},
    )
    dataset = builder.to_dataset()
    assert dataset.response.dims == ("x",)
    assert dataset.sizes == {"x": 2}
    assert dataset.temperature.dims == ()
    assert dataset.configuration.dims == ()
    assert dataset.attrs["hyperslice_builder_constant_inputs"] == ("temperature, configuration")


@pytest.mark.parametrize(
    ("inputs", "outputs", "message"),
    [
        ({}, {"response": 1}, "Input parameters cannot be empty"),
        ({"x": [1, 2]}, {"response": 1}, "must be scalar"),
        ({"x": np.nan}, {"response": 1}, "cannot be None or NaN"),
        ({"x": 1}, {}, "Output parameters cannot be empty"),
        ({"x": 1}, {"response": "bad"}, "real numeric scalar"),
    ],
)
def test_invalid_parameters_are_rejected(
    inputs: dict[str, object], outputs: dict[str, object], message: str
) -> None:
    with pytest.raises(InvalidParameterError, match=message):
        DatasetBuilder().add_point(inputs, outputs)


def test_netcdf_and_zarr_round_trip(tmp_path: Path) -> None:
    builder = DatasetBuilder()
    for temperature in (300, 450):
        for configuration in ("baseline", "reinforced"):
            builder.add_point(
                {"temperature": temperature, "configuration": configuration},
                {"response": temperature / 100 + (1 if configuration == "reinforced" else 0)},
            )
    for suffix in ("dataset.nc", "dataset.zarr"):
        target = builder.write(tmp_path / suffix)
        reopened = load_dataset(target)
        xr.testing.assert_allclose(reopened.response, builder.to_dataset().response)


def test_write_rejects_unsupported_format(tmp_path: Path) -> None:
    builder = DatasetBuilder().add_point({"x": 1, "y": 2}, {"response": 3})
    with pytest.raises(DatasetBuilderError, match="Unsupported output format"):
        builder.write(tmp_path / "dataset.csv")
