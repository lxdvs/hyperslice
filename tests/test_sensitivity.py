from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hyperslice.schema import inspect_dataset
from hyperslice.sensitivity import (
    UNDEFINED,
    format_sensitivity,
    input_dimensions,
    labelled_frame,
    partial_derivative,
    relative_sensitivity,
    sensitivity_frame,
    value_at,
)


def _valid_point(dataset: xr.Dataset) -> dict[str, float]:
    """A design point away from the synthetic dataset's invalid regions."""
    return {
        "fuel_temperature": 850.0,
        "drum_angle": 30.0,
        "pressure": 2.5,
        "flow_rate": 5.0,
        "burnup": 5.0,
    }


def test_linear_terms_are_differenced_exactly(dataset: xr.Dataset) -> None:
    point = _valid_point(dataset)
    # k_eff = ... + 0.002 * pressure - 0.0005 * burnup + 0.000002 * angle * (ft - 800)
    k_eff = dataset["k_eff"]
    assert partial_derivative(k_eff, "pressure", point) == pytest.approx(0.002)
    assert partial_derivative(k_eff, "burnup", point) == pytest.approx(-0.0005)
    expected = 0.0018 + 0.000002 * (point["fuel_temperature"] - 800)
    assert partial_derivative(k_eff, "drum_angle", point) == pytest.approx(expected)
    assert partial_derivative(dataset["peak_temperature"], "burnup", point) == pytest.approx(0.8)


def test_unsampled_directions_are_undefined(dataset: xr.Dataset) -> None:
    point = _valid_point(dataset)
    # k_eff spans flow_rate without depending on it: a real, exactly flat slope.
    assert partial_derivative(dataset["k_eff"], "flow_rate", point) == pytest.approx(0.0)
    assert np.isnan(partial_derivative(dataset["k_eff"], "absent", point))
    del point["pressure"]
    assert np.isnan(partial_derivative(dataset["k_eff"], "burnup", point))


def test_missing_neighbours_make_a_derivative_undefined(dataset: xr.Dataset) -> None:
    point = _valid_point(dataset) | {"fuel_temperature": 1050.0, "flow_rate": 2.0}
    # The synthetic sweep marks fuel_temperature >= 1050 at flow_rate == 2 invalid.
    assert np.isnan(partial_derivative(dataset["k_eff"], "drum_angle", point))


def _mixed_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"response": (("scope", "level", "pinned"), np.arange(12.0).reshape(2, 3, 2))},
        coords={"scope": ["full", "partial"], "level": [1.0, 2.0, 4.0], "pinned": [7.0, 9.0]},
    )


def test_input_dimensions_skip_categorical_and_constant_axes() -> None:
    dataset = _mixed_dataset().isel(pinned=[0])
    assert input_dimensions(inspect_dataset(dataset)) == ["level"]


def test_value_at_samples_the_pinned_point(dataset: xr.Dataset) -> None:
    point = _valid_point(dataset)
    assert value_at(dataset["k_eff"], point) == pytest.approx(0.9775)
    del point["burnup"]
    assert np.isnan(value_at(dataset["k_eff"], point))


def test_relative_sensitivity_is_the_elasticity(dataset: xr.Dataset) -> None:
    point = _valid_point(dataset)
    k_eff = dataset["k_eff"]
    # (x / y) dy/dx with dy/dx = 0.002, x = 2.5 and y = 0.9775.
    assert relative_sensitivity(k_eff, "pressure", point) == pytest.approx(0.002 * 2.5 / 0.9775)
    assert relative_sensitivity(k_eff, "burnup", point) == pytest.approx(-0.0005 * 5.0 / 0.9775)
    peak = dataset["peak_temperature"]
    expected = 0.8 * point["burnup"] / value_at(peak, point)
    assert relative_sensitivity(peak, "burnup", point) == pytest.approx(expected)


def test_relative_sensitivity_is_dimensionless(dataset: xr.Dataset) -> None:
    point = _valid_point(dataset)
    scaled = dataset.assign_coords(pressure=dataset["pressure"] * 1000.0)
    scaled["k_eff"] = scaled["k_eff"] * 7.0
    scaled_point = point | {"pressure": point["pressure"] * 1000.0}
    assert relative_sensitivity(scaled["k_eff"], "pressure", scaled_point) == pytest.approx(
        relative_sensitivity(dataset["k_eff"], "pressure", point)
    )


def test_relative_sensitivity_undefined_cases(dataset: xr.Dataset) -> None:
    point = _valid_point(dataset)
    k_eff = dataset["k_eff"]
    # An input at zero contributes nothing in fractional terms, whatever its slope.
    assert relative_sensitivity(k_eff, "burnup", point | {"burnup": 0.0}) == 0.0
    assert np.isnan(relative_sensitivity(k_eff, "absent", point))
    zero_output = xr.Dataset(
        {"response": ("a", np.array([-1.0, 0.0, 1.0]))}, coords={"a": [1.0, 2.0, 3.0]}
    )
    assert np.isnan(relative_sensitivity(zero_output["response"], "a", {"a": 2.0}))
    assert relative_sensitivity(zero_output["response"], "a", {"a": 3.0}) == pytest.approx(3.0)


def test_sensitivity_frame_is_outputs_by_inputs(dataset: xr.Dataset) -> None:
    schema = inspect_dataset(dataset)
    point = _valid_point(dataset)
    frame = sensitivity_frame(dataset, schema, point)
    assert list(frame.index) == ["k_eff", "peak_temperature"]
    assert list(frame.columns) == input_dimensions(schema)
    assert frame.loc["k_eff", "pressure"] == pytest.approx(
        relative_sensitivity(dataset["k_eff"], "pressure", point)
    )


def test_outputs_missing_an_input_report_no_derivative() -> None:
    dataset = xr.Dataset(
        {
            "both": (("a", "b"), np.arange(6.0).reshape(3, 2)),
            "only_a": ("a", np.array([0.0, 2.0, 4.0])),
        },
        coords={"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0]},
    )
    schema = inspect_dataset(dataset)
    frame = sensitivity_frame(dataset, schema, {"a": 1.0, "b": 0.0})
    assert list(frame.index) == ["both"]
    assert np.isnan(partial_derivative(dataset["only_a"], "b", {"a": 1.0, "b": 0.0}))


def test_cells_are_formatted_and_labelled_without_units(dataset: xr.Dataset) -> None:
    schema = inspect_dataset(dataset)
    frame = sensitivity_frame(dataset, schema, _valid_point(dataset))
    display = labelled_frame(frame, schema)
    assert "Multiplication factor" in display.index
    assert "Control drum angle" in display.columns
    assert not any("[" in label for label in [*display.index, *display.columns])
    assert display.loc["Multiplication factor", "Pressure"] == "0.005115"
    unpinned = labelled_frame(sensitivity_frame(dataset, schema, {}), schema)
    assert (unpinned == UNDEFINED).all().all()


def test_format_sensitivity_marks_unusable_values() -> None:
    assert format_sensitivity(float("nan")) == UNDEFINED
    assert format_sensitivity(float("inf")) == UNDEFINED
    assert format_sensitivity(0.0) == "0"
    assert format_sensitivity(0.000123456) == "0.0001235"


def test_unsorted_coordinates_still_differentiate() -> None:
    values = np.array([0.0, 3.0, 1.0])
    dataset = xr.Dataset(
        {"response": (("a", "b"), np.array([[0.0, 6.0, 2.0], [1.0, 7.0, 3.0]]))},
        coords={"a": [0.0, 1.0], "b": values},
    )
    for target in values:
        derivative = partial_derivative(dataset["response"], "b", {"a": 0.0, "b": target})
        assert derivative == pytest.approx(2.0)
