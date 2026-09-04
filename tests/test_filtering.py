from __future__ import annotations

import holoviews as hv
import numpy as np
import panel as pn
import xarray as xr
from conftest import drag

from hyperslice import Explorer
from hyperslice.colors import HIGHLIGHT_COLOR, VIRIDIS
from hyperslice.explorer import TAB_STYLES
from hyperslice.filtering import (
    CONTINUOUS_COLOR,
    CONTINUOUS_MARKER,
    NONMATCHING_ALPHA,
    PLAIN_MARKER,
    TICK_LIMIT,
    FilterView,
    distinct_values,
    most_interesting,
    tick_stylesheet,
)
from hyperslice.schema import inspect_dataset


def test_explorer_exposes_slicer_and_filter_tabs(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    assert isinstance(explorer.view, pn.Tabs)
    assert list(explorer.view._names) == ["Filter", "Slicer"]
    assert explorer.view.active == 0
    assert explorer.view[0] is explorer.filter_view.view
    assert explorer.view[1] is explorer.slicer_view
    assert "position: sticky" in TAB_STYLES
    assert TAB_STYLES in explorer.view.stylesheets


def test_filter_view_has_input_and_output_absolute_ranges(dataset: xr.Dataset) -> None:
    view = FilterView(dataset, inspect_dataset(dataset))
    data = dataset[view.variable]
    expected = set(data.dims) | set(view._compatible_outputs())
    assert set(view._filter_widgets) == expected
    frame = view._sample_frame()
    for name, widget in view._filter_widgets.items():
        finite = frame[name].to_numpy()
        finite = finite[np.isfinite(finite)]
        assert widget.start == float(finite.min())
        assert widget.value[0] == float(finite.min())
        assert widget.value[1] == float(finite.max())
        assert widget.end >= float(finite.max())


def test_range_filter_fades_points_without_removing_them(dataset: xr.Dataset) -> None:
    view = FilterView(dataset, inspect_dataset(dataset))
    frame = view._sample_frame()
    widget = view._filter_widgets["fuel_temperature"]
    midpoint = (widget.start + widget.end) / 2
    drag(widget, (widget.start, midpoint))

    points = view._plot.object.traverse(lambda element: element, specs=[hv.Points])
    assert len(points) == 2
    counts = {point.label: len(point) for point in points}
    assert counts["Inside filters"] > 0
    assert counts["Outside filters"] > 0
    assert sum(counts.values()) == len(frame)
    faded = next(point for point in points if point.label == "Outside filters")
    options = faded.opts.get(backend="bokeh").kwargs
    assert options["alpha"] == NONMATCHING_ALPHA
    assert 0.0 < NONMATCHING_ALPHA < 1.0
    inside = next(point for point in points if point.label == "Inside filters")
    assert inside.opts.get(backend="bokeh").kwargs["alpha"] == 1.0


def test_nonmatching_points_are_faded_by_default(dataset: xr.Dataset) -> None:
    view = FilterView(dataset, inspect_dataset(dataset))
    assert view.nonmatching_widget.options == ["Fade", "Hide"]
    assert view.nonmatching_widget.value == "Fade"


def test_legend_clicks_hide_a_layer_rather_than_fading_it(dataset: xr.Dataset) -> None:
    view = FilterView(dataset, inspect_dataset(dataset))
    figure = hv.render(view._plot.object, backend="bokeh")
    assert [legend.click_policy for legend in figure.legend] == ["hide"]


def test_nonmatching_points_can_be_hidden(dataset: xr.Dataset) -> None:
    view = FilterView(dataset, inspect_dataset(dataset))
    frame = view._sample_frame()
    widget = view._filter_widgets["fuel_temperature"]
    drag(widget, (widget.start, (widget.start + widget.end) / 2))
    view.nonmatching_widget.value = "Hide"

    points = view._plot.object.traverse(lambda element: element, specs=[hv.Points])
    assert len(points) == 1
    assert points[0].label == "Inside filters"
    assert 0 < len(points[0]) < len(frame)
    assert "outside filters" in view._summary.object


def test_filter_axis_checkbox_matrix_updates_projection(dataset: xr.Dataset) -> None:
    view = FilterView(dataset, inspect_dataset(dataset))
    new_x = next(dim for dim in dataset[view.variable].dims if dim not in {view.x_dim, view.y_dim})
    view._axis_x_checks[new_x].value = True
    assert view.x_dim == new_x
    assert sum(check.value for check in view._axis_x_checks.values()) == 1
    assert view._axis_y_checks[new_x].disabled
    assert view._plot.object is not None


def test_filter_allows_output_variables_on_axes(dataset: xr.Dataset) -> None:
    view = FilterView(dataset, inspect_dataset(dataset))
    output_axis = next(name for name in view._compatible_outputs() if name != view.variable)
    assert output_axis in view.x_widget.options.values()
    assert output_axis in view._axis_x_checks
    view._axis_x_checks[output_axis].value = True
    assert view.x_dim == output_axis
    assert view._plot.object is not None
    assert not view._message.visible
    assert view._axis_label(output_axis) == view._output_label(output_axis)


def test_filter_output_change_rebuilds_compatible_controls() -> None:
    dataset = xr.Dataset(
        {
            "first": (("a", "b", "c"), np.ones((2, 3, 4))),
            "second": (("a", "b"), np.arange(6).reshape(2, 3)),
        },
        coords={"a": [0, 1], "b": [1, 2, 3], "c": [4, 5, 6, 7]},
    )
    view = FilterView(dataset, inspect_dataset(dataset))
    view.variable_widget.value = "second"
    assert set(view._axis_x_checks) == {"a", "b", "second"}
    assert set(view._filter_widgets) == {"a", "b", "second"}
    assert view._plot.object is not None


def _gapped_dataset() -> xr.Dataset:
    """Two outputs where 'sparse' has no value at half the samples."""
    sparse = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, 6.0]])
    return xr.Dataset(
        {
            "dense": (("a", "b"), np.arange(6.0).reshape(2, 3)),
            "sparse": (("a", "b"), sparse),
        },
        coords={"a": [0, 1], "b": [1, 2, 3]},
    )


def test_missing_checkbox_appears_only_for_gapped_variables() -> None:
    view = FilterView(_gapped_dataset(), inspect_dataset(_gapped_dataset()))
    assert set(view._missing_widgets) == {"sparse"}
    checkbox = view._missing_widgets["sparse"]
    assert checkbox.value is True
    assert "3 samples with no value" in checkbox.name


def test_missing_checkbox_toggles_point_inclusion() -> None:
    dataset = _gapped_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    view.variable_widget.value = "dense"
    included = int(view._summary.object.split()[0].replace("**", "").replace(",", ""))
    assert included == 6

    view._missing_widgets["sparse"].value = False
    remaining = int(view._summary.object.split()[0].replace("**", "").replace(",", ""))
    assert remaining == 3
    assert "no value to filter on" in view._coverage.object
    assert view._coverage.visible

    view._missing_widgets["sparse"].value = True
    assert view._coverage.visible is False


def test_all_missing_variable_disables_slider_but_keeps_checkbox() -> None:
    dataset = xr.Dataset(
        {
            "dense": (("a", "b"), np.arange(6.0).reshape(2, 3)),
            "empty": (("a", "b"), np.full((2, 3), np.nan)),
        },
        coords={"a": [0, 1], "b": [1, 2, 3]},
    )
    view = FilterView(dataset, inspect_dataset(dataset))
    assert view._filter_widgets["empty"].disabled is True
    assert view._missing_widgets["empty"].value is True
    assert int(view._summary.object.split()[0].replace("**", "")) == 6


def test_coverage_reports_uncolorable_samples() -> None:
    dataset = _gapped_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    view.variable_widget.value = "sparse"
    assert "drawn without color" in view._coverage.object
    assert view._coverage.visible


def test_missing_checkbox_state_survives_rebuild() -> None:
    dataset = _gapped_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    view._missing_widgets["sparse"].value = False
    view.variable_widget.value = "sparse"
    view.variable_widget.value = "dense"
    assert view._missing_widgets["sparse"].value is False


def _constant_field_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "varies": (("a", "b"), np.arange(6.0).reshape(2, 3)),
            "also_varies": (("a", "b"), np.arange(6.0).reshape(2, 3) * 3),
            "fixed": (("a", "b"), np.full((2, 3), 7.0)),
        },
        coords={"a": [0, 1], "b": [1, 2, 3]},
    )


def test_constant_outputs_are_hidden_from_filters_and_dropdowns() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    assert "fixed" not in view._filter_widgets
    assert "fixed" not in view.variable_widget.options
    assert "fixed" not in view.x_widget.options
    assert "fixed" not in view.y_widget.options
    assert "fixed" not in view._axis_x_checks
    assert "varies" in view._filter_widgets


def test_constant_outputs_remain_in_hover_data() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    assert "fixed" in view._sample_frame().columns
    points = view._plot.object.traverse(lambda element: element, specs=[hv.Points])
    assert "fixed" in {dimension.name for dimension in points[0].vdims}


def test_constant_input_stays_in_hover_but_not_in_controls() -> None:
    dataset = xr.Dataset(
        {"varies": (("a", "b"), np.arange(6.0).reshape(2, 3))},
        coords={"a": [0, 1], "b": [1, 2, 3], "pinned": 4.2},
    )
    view = FilterView(dataset, inspect_dataset(dataset))
    assert "pinned" not in view._filter_widgets
    assert "pinned" in view._sample_frame().columns


def test_high_cardinality_outputs_are_styled_green() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    styled = view._filter_widgets["also_varies"].stylesheets
    assert styled and CONTINUOUS_COLOR in styled[0]
    bucketed = xr.Dataset(
        {
            "varies": (("a", "b"), np.arange(6.0).reshape(2, 3)),
            "repeated": (("a", "b"), np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])),
        },
        coords={"a": [0, 1], "b": [1, 2, 3]},
    )
    plain = FilterView(bucketed, inspect_dataset(bucketed))
    assert all(
        CONTINUOUS_COLOR not in sheet for sheet in plain._filter_widgets["repeated"].stylesheets
    )


def test_z_variable_widget_offers_inputs_and_outputs() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    assert view.variable_widget.name == "Z variable"
    offered = set(view.variable_widget.options.values())
    assert {"a", "b"}.issubset(offered)
    assert {"varies", "also_varies"}.issubset(offered)


def test_input_coordinate_can_colour_the_cloud() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    view.variable_widget.value = "b"

    assert view._grid_variable() in dataset.data_vars
    assert view._grid_dimensions() == ("a", "b")
    points = view._plot.object.traverse(lambda element: element, specs=[hv.Points])
    assert points[0].opts.get(backend="bokeh").kwargs["color"] == "b"
    assert sum(len(element) for element in points) == 6
    assert not view._message.visible
    assert set(view._filter_widgets) == {"a", "b", "varies", "also_varies"}


def test_plots_use_the_viridis_ramp() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    points = view._plot.object.traverse(lambda element: element, specs=[hv.Points])
    palette = points[0].opts.get(backend="bokeh").kwargs["cmap"]
    assert palette is VIRIDIS
    assert palette[0].lower() == "#440154"
    assert palette[-1].lower() == "#fde724"


def test_tick_stylesheet_marks_each_distinct_value() -> None:
    css = tick_stylesheet(np.array([0.0, 5.0, 5.0, 10.0]), 0.0, 10.0)
    assert css is not None
    assert "0.000% 0" in css
    assert "50.000% 0" in css
    assert "100.000% 0" in css
    assert css.count("linear-gradient") == 3
    assert "pointer-events: none" in css


def test_tick_stylesheet_skips_degenerate_and_dense_fields() -> None:
    assert tick_stylesheet(np.array([4.0, 4.0]), 4.0, 5.0) is None
    assert tick_stylesheet(np.array([np.nan, np.nan]), 0.0, 1.0) is None
    assert tick_stylesheet(np.arange(TICK_LIMIT + 1, dtype=float), 0.0, float(TICK_LIMIT)) is None
    assert tick_stylesheet(np.array([1.0, 2.0]), 2.0, 2.0) is None


def test_filter_sliders_carry_data_ticklines(dataset: xr.Dataset) -> None:
    view = FilterView(dataset, inspect_dataset(dataset))
    frame = view._sample_frame()
    for name, widget in view._filter_widgets.items():
        if not isinstance(widget, pn.widgets.RangeSlider):
            continue
        values = np.asarray(frame[name].to_numpy(), dtype=float)
        distinct = np.unique(values[np.isfinite(values)])
        tick_sheets = [sheet for sheet in widget.stylesheets if ".noUi-base::after" in sheet]
        if 2 <= distinct.size <= TICK_LIMIT:
            assert len(tick_sheets) == 1
            span = widget.end - widget.start
            first = (distinct[0] - widget.start) / span * 100.0
            last = (distinct[-1] - widget.start) / span * 100.0
            assert f"{first:.3f}% 0" in tick_sheets[0]
            assert f"{last:.3f}% 0" in tick_sheets[0]
        else:
            assert not tick_sheets


def test_inputs_are_always_styled_green() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    for name in ("a", "b"):
        styled = view._filter_widgets[name].stylesheets
        assert styled and CONTINUOUS_COLOR in styled[0]
    assert view._is_continuous("a")


def test_categorical_input_is_styled_green() -> None:
    dataset = xr.Dataset(
        {"varies": (("scope", "b"), np.arange(6.0).reshape(2, 3))},
        coords={"scope": ["full", "fuel-only"], "b": [1, 2, 3]},
    )
    view = FilterView(dataset, inspect_dataset(dataset))
    styled = view._filter_widgets["scope"].stylesheets
    assert styled and CONTINUOUS_COLOR in styled[0]


def test_dropdowns_mark_continuous_fields_green() -> None:
    dataset = xr.Dataset(
        {
            "independent": (("a", "b"), np.arange(6.0).reshape(2, 3)),
            "bucketed": (("a", "b"), np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])),
        },
        coords={"a": [0, 1], "b": [1, 2, 3]},
    )
    view = FilterView(dataset, inspect_dataset(dataset))
    labels = {value: label for label, value in view.variable_widget.options.items()}
    assert labels["independent"].startswith(CONTINUOUS_MARKER)
    assert labels["bucketed"].startswith(PLAIN_MARKER)
    assert labels["a"].startswith(CONTINUOUS_MARKER)


def test_axis_dropdown_keeps_markers_after_rebuild() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    view.variable_widget.value = "also_varies"
    assert all(
        label.startswith((CONTINUOUS_MARKER, PLAIN_MARKER)) for label in view.x_widget.options
    )
    assert set(view.y_widget.options.values()) == set(view.x_widget.options.values()) - {view.x_dim}


def test_point_size_option_resizes_both_layers() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    drag(view._filter_widgets["varies"], (0.0, 2.0))
    drag(view.point_size_widget, 14)

    points = view._plot.object.traverse(lambda element: element, specs=[hv.Points])
    assert len(points) == 2
    assert all(element.opts.get(backend="bokeh").kwargs["size"] == 14 for element in points)


def test_colour_divisions_band_the_palette() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    points = view._plot.object.traverse(lambda element: element, specs=[hv.Points])
    assert points[0].opts.get(backend="bokeh").kwargs["cmap"] is VIRIDIS

    view.continuous_color_widget.value = False
    assert view.color_levels_widget.disabled is False
    drag(view.color_levels_widget, 5)
    points = view._plot.object.traverse(lambda element: element, specs=[hv.Points])
    palette = points[0].opts.get(backend="bokeh").kwargs["cmap"]
    assert len(palette) == 5
    assert palette[0] == VIRIDIS[0]
    assert palette[-1] == VIRIDIS[-1]

    view.continuous_color_widget.value = True
    assert view.color_levels_widget.disabled is True


def test_hamburger_menu_toggles_the_options_panel() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    assert view._menu.visible is False
    view.menu_toggle.value = True
    assert view._menu.visible is True
    assert view.point_size_widget in view._menu
    assert view.color_levels_widget in view._menu


def test_tapping_a_point_reports_its_design_point() -> None:
    dataset = _constant_field_dataset()
    captured: list[tuple[dict[str, object], str]] = []
    view = FilterView(
        dataset,
        inspect_dataset(dataset),
        on_point_selected=lambda coordinates, variable: captured.append((coordinates, variable)),
    )
    rows = view._selected_rows
    assert rows is not None and len(rows) == 6

    view._on_point_tapped([3])
    assert len(captured) == 1
    coordinates, variable = captured[0]
    assert coordinates == {"a": rows.iloc[3]["a"], "b": rows.iloc[3]["b"]}
    assert variable == view.variable

    view._on_point_tapped([])
    assert len(captured) == 1


def test_tapping_a_point_rings_it_in_the_cloud() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    view._on_point_tapped([3])

    rows = view._selected_rows
    assert rows is not None
    assert view._selected_point == {"a": rows.iloc[3]["a"], "b": rows.iloc[3]["b"]}
    ring = next(
        element
        for element in view._plot.object.traverse(lambda item: item, specs=[hv.Points])
        if element.label == "Selected point"
    )
    assert len(ring) == 1
    options = ring.opts.get(backend="bokeh").kwargs
    assert options["line_color"] == HIGHLIGHT_COLOR
    assert options["fill_alpha"] == 0.0
    assert options["size"] > view.point_size_widget.value


def test_selecting_a_filtered_out_point_clears_every_filter(dataset: xr.Dataset) -> None:
    view = FilterView(dataset, inspect_dataset(dataset))
    widget = view._filter_widgets["fuel_temperature"]
    drag(widget, (widget.start, (widget.start + widget.end) / 2))
    assert "0 outside filters" not in view._summary.object

    hidden = {dim: float(dataset.coords[dim].values[-1]) for dim in dataset["k_eff"].dims}
    view.select_point(hidden)

    assert widget.value == (widget.start, widget.end)
    assert all(
        checkbox.value is True for checkbox in view._missing_widgets.values()
    )  # missing samples come back too
    assert "0 outside filters" in view._summary.object
    assert view._selected_point == hidden


def test_selecting_a_visible_point_leaves_the_filters_alone(dataset: xr.Dataset) -> None:
    view = FilterView(dataset, inspect_dataset(dataset))
    widget = view._filter_widgets["fuel_temperature"]
    narrowed = (widget.start, (widget.start + widget.end) / 2)
    drag(widget, narrowed)

    visible = {dim: float(dataset.coords[dim].values[0]) for dim in dataset["k_eff"].dims}
    view.select_point(visible)

    assert widget.value == narrowed
    assert view._selected_point == visible


def test_dragging_a_slider_does_not_redraw_until_release() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    widget = view._filter_widgets["varies"]
    before = view._plot.object

    # Intermediate positions during the gesture must not rebuild the plot.
    for upper in (4.0, 3.0, 2.0):
        widget.value = (0.0, upper)
    assert view._plot.object is before

    drag(widget, (0.0, 2.0))
    assert view._plot.object is not before


def test_plot_pane_has_a_fixed_height() -> None:
    dataset = _constant_field_dataset()
    view = FilterView(dataset, inspect_dataset(dataset))
    assert view._plot.height == 560
    assert view._plot.sizing_mode == "stretch_width"


def test_filter_view_opens_on_the_most_interesting_outputs(dataset: xr.Dataset) -> None:
    schema = inspect_dataset(dataset)
    view = FilterView(dataset, schema)
    by_distinct = sorted(
        (name for name, info in schema.variables.items() if not info.constant),
        key=lambda name: schema.variables[name].distinct_count,
        reverse=True,
    )
    # Z takes the output with the most distinct values, X the next output, and
    # Y — the outputs exhausted — the input swept over the most levels.
    assert view.variable == by_distinct[0]
    assert view.x_dim == by_distinct[1]
    swept = max(
        (dim for dim in dataset[view.variable].dims if not schema.coordinates[dim].constant),
        key=lambda dim: schema.coordinates[dim].size,
    )
    assert view.y_dim == swept
    assert len({view.variable, view.x_dim, view.y_dim}) == 3


def test_most_interesting_prefers_outputs_then_distinct_counts() -> None:
    dataset = xr.Dataset(
        {
            "coarse": (("a", "b"), np.repeat([[0.0, 1.0]], 4, axis=0)),
            "fine": (("a", "b"), np.arange(8.0).reshape(4, 2)),
        },
        coords={"a": [0.0, 1.0, 2.0, 3.0], "b": [0.0, 1.0]},
    )
    schema = inspect_dataset(dataset)
    assert distinct_values(schema, "fine") == 8
    assert distinct_values(schema, "coarse") == 2
    assert distinct_values(schema, "a") == 4
    assert most_interesting(schema, ["b", "a", "coarse", "fine"]) == ["fine", "coarse", "a", "b"]

    view = FilterView(dataset, schema)
    assert (view.variable, view.x_dim, view.y_dim) == ("fine", "coarse", "a")
    assert "coarse" not in view.y_widget.options.values()
