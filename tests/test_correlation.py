from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import drag

from hyperslice.correlation import (
    BEND_DOWN,
    BEND_UP,
    CORRELATION_PALETTE,
    PROFILE_BINS,
    UNDEFINED_COLOR,
    ProfileFrames,
    band_color,
    bend_glyph,
    fit_profile,
    informative,
    mean_profile,
    profile_frames,
    render_matrix,
    row_fractions,
)
from hyperslice.filtering import FilterView
from hyperslice.schema import inspect_dataset
from hyperslice.sensitivity import UNDEFINED


def test_mean_profile_averages_each_level() -> None:
    x = np.array([2.0, 1.0, 2.0, 1.0, 3.0])
    y = np.array([10.0, 1.0, 20.0, 3.0, 7.0])
    positions, means = mean_profile(x, y)
    assert positions.tolist() == [1.0, 2.0, 3.0]
    assert means.tolist() == [2.0, 15.0, 7.0]


def test_mean_profile_bins_a_dense_input() -> None:
    x = np.linspace(0.0, 1.0, 1000)
    positions, means = mean_profile(x, 2.0 * x)
    assert positions.size == PROFILE_BINS
    assert np.all(np.diff(positions) > 0)
    assert means == pytest.approx(2.0 * positions)


def test_a_line_has_its_elasticity_and_no_nonlinearity() -> None:
    x = np.linspace(1.0, 3.0, 5)
    fit = fit_profile(x, 3.0 * x)
    # Proportional: a 1% change in x is a 1% change in y.
    assert fit.sensitivity == pytest.approx(1.0)
    assert fit.nonlinearity == pytest.approx(0.0, abs=1e-12)
    assert fit.correlation == pytest.approx(1.0)
    assert fit.levels == 5


def test_sensitivity_carries_magnitude_that_correlation_does_not() -> None:
    x = np.linspace(1.0, 3.0, 5)
    weak = fit_profile(x, 1000.0 + 0.001 * x)
    strong = fit_profile(x, 1.0 + 10.0 * x)
    assert weak.correlation == pytest.approx(strong.correlation)
    assert abs(weak.sensitivity) < 1e-5 < 0.5 < strong.sensitivity


def test_averaging_cancels_scatter_from_other_inputs() -> None:
    x = np.repeat([1.0, 2.0, 3.0], 2)
    other = np.tile([-5.0, 5.0], 3)
    fit = fit_profile(x, 10.0 + x + other)
    assert fit.nonlinearity == pytest.approx(0.0, abs=1e-12)
    assert fit.sensitivity == pytest.approx(2.0 / 12.0)
    assert fit.correlation < 0.5


def test_a_symmetric_parabola_is_fully_nonlinear_and_bends_up() -> None:
    x = np.linspace(-1.0, 1.0, 5)
    fit = fit_profile(x, 1.0 + x**2)
    assert fit.sensitivity == pytest.approx(0.0, abs=1e-12)
    assert fit.nonlinearity == pytest.approx(1.0)
    assert fit.bend == 1.0
    assert fit_profile(x, 1.0 - x**2).bend == -1.0


def test_undefined_fits() -> None:
    assert np.isnan(fit_profile(np.ones(5), np.arange(5.0)).sensitivity)
    assert np.isnan(fit_profile(np.arange(5.0), np.zeros(5)).sensitivity)
    assert np.isnan(fit_profile(np.array([1.0, np.nan]), np.array([1.0, 2.0])).sensitivity)
    # A flat output has no sensitivity and no curvature.
    flat = fit_profile(np.arange(5.0), np.full(5, 4.0))
    assert flat.sensitivity == 0.0 and flat.nonlinearity == 0.0
    # Two levels fix a line but not a curve.
    two = fit_profile(np.array([0.0, 0.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.isfinite(two.sensitivity) and np.isnan(two.nonlinearity)


def test_row_fractions_scale_to_the_strongest_input() -> None:
    fractions = row_fractions(pd.Series({"a": -4.0, "b": 2.0, "c": np.nan}))
    assert fractions["a"] == pytest.approx(-1.0)
    assert fractions["b"] == pytest.approx(0.5)
    assert np.isnan(fractions["c"])
    assert (row_fractions(pd.Series({"a": 0.0})) == 0.0).all()


def test_band_color_runs_blue_to_red() -> None:
    assert band_color(-1.0) == CORRELATION_PALETTE[0]
    assert band_color(0.0) == CORRELATION_PALETTE[5]
    assert band_color(1.0) == CORRELATION_PALETTE[-1]
    assert band_color(0.04) == CORRELATION_PALETTE[5]


def test_bend_glyph_only_for_readable_curvature() -> None:
    assert bend_glyph(0.5, 1.0) == f" {BEND_UP}"
    assert bend_glyph(0.5, -1.0) == f" {BEND_DOWN}"
    assert bend_glyph(0.01, 1.0) == ""
    assert bend_glyph(float("nan"), 1.0) == ""


def _frames(
    sensitivity: dict[str, list[float]], nonlinearity: dict[str, list[float]]
) -> ProfileFrames:
    index = ["y", "z", "w"]
    sens = pd.DataFrame(sensitivity, index=index, dtype=float)
    curve = pd.DataFrame(nonlinearity, index=index, dtype=float)
    return ProfileFrames(
        sensitivity=sens,
        nonlinearity=curve,
        bend=curve * 0.0 + 1.0,
        correlation=sens.clip(-1, 1),
        levels=sens * 0.0 + 3,
    )


def _matrix() -> ProfileFrames:
    return _frames(
        # y responds to a; z is flat everywhere; w is undefined everywhere; b
        # moves nothing; c moves nothing on average but bends y.
        {
            "a": [1.0, 0.0, np.nan],
            "b": [np.nan, 0.0, np.nan],
            "c": [0.0, 0.0, np.nan],
        },
        {
            "a": [0.0, 0.0, np.nan],
            "b": [np.nan, 0.0, np.nan],
            "c": [1.0, 0.0, np.nan],
        },
    )


def test_informative_cells_are_defined_and_nonzero_or_curved() -> None:
    shown = informative(_matrix())
    assert shown.loc["y"].tolist() == [True, False, True]
    assert not shown.loc["z"].any() and not shown.loc["w"].any()


def test_render_matrix_shows_both_values_and_marks_undefined() -> None:
    frames = _frames({"a": [1.0, 0.5, 2.0]}, {"a": [0.0, 0.2, np.nan]})
    frames.sensitivity.loc["w", "a"] = np.nan
    frames.sensitivity["b"] = [0.5, 0.25, 1.0]
    frames.nonlinearity["b"] = [0.0, 0.0, 0.0]
    for frame in (frames.bend, frames.correlation, frames.levels):
        frame["b"] = 1.0
    html = render_matrix(frames, {"a": "<A>"}, {"y": "Why"})
    assert "+1" in html and "NL 0%" in html
    assert f"background: {band_color(1.0)}" in html
    assert f"background: {UNDEFINED_COLOR}" in html and UNDEFINED in html
    assert "&lt;A&gt;" in html and "<A>" not in html
    assert "<th>Why</th>" in html and "<th>z</th>" in html
    assert "hs-corr-hidden" not in html


def test_render_matrix_hides_flat_and_undefined_outputs_and_inputs() -> None:
    html = render_matrix(_matrix(), {"a": "Alpha", "b": "Beta", "c": "Gamma"}, {"z": "Zed"})
    table = html.split("</table>")[0]
    assert "<th>y</th>" in table
    assert "Zed" not in table and "<th>w</th>" not in table
    assert "Alpha" in table and "Gamma" in table and "Beta" not in table
    note = html.split('<div class="hs-corr-hidden">')[1]
    assert "Outputs: Zed, w" in note and "Inputs: Beta" in note


def test_render_matrix_with_nothing_to_show() -> None:
    frames = _frames({"a": [0.0, np.nan, 0.0]}, {"a": [0.0, np.nan, np.nan]})
    html = render_matrix(frames, {}, {})
    assert "<table" not in html
    assert "No output responds" in html and "Outputs: y, z, w" in html


def test_frames_cover_numeric_inputs_and_order_outputs(dataset: xr.Dataset) -> None:
    schema = inspect_dataset(dataset)
    frame = FilterView(dataset, schema)._sample_frame()
    frames = profile_frames(frame, schema)
    assert list(frames.sensitivity.index) == [
        name
        for name in sorted(schema.variables, key=lambda n: -schema.variables[n].cardinality)
        if name in frame.columns
    ]
    assert all(not schema.coordinates[name].categorical for name in frames.sensitivity.columns)
    assert frames.sensitivity.shape == frames.nonlinearity.shape
    assert np.isfinite(frames.sensitivity.to_numpy()).any()


def test_filter_view_fits_only_the_samples_inside_the_filters(dataset: xr.Dataset) -> None:
    schema = inspect_dataset(dataset)
    view = FilterView(dataset, schema)
    frame = view._sample_frame()
    assert f"{len(frame):,} samples" in view._correlation_title.object
    before = view._correlation.object

    dim = next(name for name in view._filter_widgets if name in schema.coordinates)
    widget = view._filter_widgets[dim]
    low, high = widget.start, widget.end
    drag(widget, (low, low + (high - low) / 2))
    included = view._included_mask(frame)
    assert included.sum() < len(frame)
    assert f"{int(included.sum()):,} samples" in view._correlation_title.object
    assert view._correlation.object != before
    assert view._correlation.object == render_matrix(
        profile_frames(frame.loc[included], schema),
        {name: info.long_name for name, info in schema.coordinates.items()},
        {name: info.long_name for name, info in schema.variables.items()},
    )
