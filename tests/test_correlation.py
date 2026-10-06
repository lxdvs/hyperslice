from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import drag

from hyperslice.correlation import (
    BEND_DOWN,
    BEND_UP,
    CORRELATION_PALETTE,
    CORRELATION_STYLES,
    MINUS_SIGN,
    PROFILE_BINS,
    UNDEFINED_COLOR,
    ProfileFrames,
    band_color,
    bend_glyph,
    best_fit,
    fit_profile,
    informative,
    mean_profile,
    profile_frames,
    render_matrix,
    row_fractions,
)
from hyperslice.correlation_view import SensitivityMatrix
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
    before = view._correlation.content

    dim = next(name for name in view._filter_widgets if name in schema.coordinates)
    widget = view._filter_widgets[dim]
    low, high = widget.start, widget.end
    drag(widget, (low, low + (high - low) / 2))
    included = view._included_mask(frame)
    assert included.sum() < len(frame)
    assert f"{int(included.sum()):,} samples" in view._correlation_title.object
    assert view._correlation.content != before
    assert view._correlation.content == render_matrix(
        profile_frames(frame.loc[included], schema),
        {name: info.long_name for name, info in schema.coordinates.items()},
        {name: info.long_name for name, info in schema.variables.items()},
    )


def test_best_fit_recovers_a_line_and_its_quality() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    fit = best_fit(x, 2.0 * x - 1.0)
    assert fit is not None
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(-1.0)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.count == 4
    assert fit.equation() == f"y = 2·x {MINUS_SIGN} 1 (R² = 1.000, n = 4)"
    scattered = best_fit(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 2.0, 0.0, 2.0]))
    assert scattered is not None and scattered.r_squared == pytest.approx(0.0)


def test_best_fit_needs_numeric_varying_x() -> None:
    assert best_fit(np.ones(4), np.arange(4.0)) is None
    assert best_fit(np.array([1.0]), np.array([2.0])) is None
    assert best_fit(np.array(["a", "b"]), np.array([1.0, 2.0])) is None
    flat = best_fit(np.arange(3.0), np.full(3, 5.0))
    assert flat is not None and flat.slope == pytest.approx(0.0) and np.isnan(flat.r_squared)


def test_render_matrix_cells_name_their_pair_and_outline_the_selection() -> None:
    html = render_matrix(_matrix(), {}, {}, selected=("y", "c"))
    cells = html.split("<td")[1:]
    assert 'data-output="y" data-input="a"' in cells[0]
    assert 'class="selected"' in cells[1] and 'data-input="c"' in cells[1]
    assert html.count("selected") == 1


def test_matrix_component_reports_clicked_cells() -> None:
    picked: list[tuple[str, str]] = []
    matrix = SensitivityMatrix(on_select=lambda output, dim: picked.append((output, dim)))
    matrix._handle_msg({"output": "y", "input": "a"})
    matrix._handle_msg({"output": "y"})
    matrix._handle_msg("noise")
    assert picked == [("y", "a")]
    assert CORRELATION_STYLES in matrix.stylesheets


def _fit_curves(view: FilterView) -> list[hv.Curve]:
    return view._plot.object.traverse(lambda element: element, specs=[hv.Curve])


def test_clicking_a_cell_plots_the_pair_with_its_line_and_keeps_the_filters(
    dataset: xr.Dataset,
) -> None:
    schema = inspect_dataset(dataset)
    view = FilterView(dataset, schema)
    frame = view._sample_frame()
    widget = view._filter_widgets["fuel_temperature"]
    drag(widget, (widget.start, (widget.start + widget.end) / 2))
    z = view.variable
    ranges = {name: w.value for name, w in view._filter_widgets.items()}
    included = view._included_mask(frame)
    assert not view.correlate_widget.value and not _fit_curves(view)

    view._correlation._handle_msg({"output": "k_eff", "input": "drum_angle"})
    assert (view.x_dim, view.y_dim, view.variable) == ("drum_angle", "k_eff", z)
    assert view.correlate_widget.value
    assert {name: w.value for name, w in view._filter_widgets.items()} == ranges
    assert (view._included_mask(frame) == included).all()

    inside = frame.loc[included]
    fit = best_fit(inside["drum_angle"].to_numpy(), inside["k_eff"].to_numpy())
    assert fit is not None
    (curve,) = _fit_curves(view)
    assert curve.label == f"Best fit: {fit.equation()}"
    assert curve.dimension_values(1)[0] == pytest.approx(
        fit.slope * curve.dimension_values(0)[0] + fit.intercept
    )
    assert "Best fit:" in view._summary.object
    (picked,) = [cell for cell in view._correlation.content.split("<td")[1:] if "selected" in cell]
    assert 'data-output="k_eff" data-input="drum_angle"' in picked

    view.correlate_widget.value = False
    assert not _fit_curves(view)
    assert "Best fit:" not in view._summary.object
    assert "selected" not in view._correlation.content


def test_correlate_ignores_pairs_that_cannot_be_plotted(dataset: xr.Dataset) -> None:
    view = FilterView(dataset, inspect_dataset(dataset))
    axes = (view.x_dim, view.y_dim)
    view.correlate("k_eff", "no_such_input")
    view.correlate("k_eff", "k_eff")
    assert (view.x_dim, view.y_dim) == axes
    assert not view.correlate_widget.value


def test_correlate_button_sits_under_the_nonmatching_selector(dataset: xr.Dataset) -> None:
    view = FilterView(dataset, inspect_dataset(dataset))
    controls = list(view.view[0])
    assert controls.index(view.correlate_widget) == controls.index(view.nonmatching_widget) + 1
    assert view.correlate_widget.name == "Correlate"
