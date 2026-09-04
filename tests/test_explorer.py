from __future__ import annotations

import holoviews as hv
import numpy as np
import pytest
import xarray as xr
from conftest import drag

from hyperslice import Explorer
from hyperslice.colors import HIGHLIGHT_COLOR, VIRIDIS


def test_view_model_axis_variable_and_state(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    assert explorer.x_dim != explorer.y_dim
    expected = set(dataset[explorer.variable].dims) - {explorer.x_dim, explorer.y_dim}
    assert set(explorer.fixed_selections) == expected
    explorer.x_widget.value = "pressure"
    assert explorer.x_dim != explorer.y_dim
    assert set(explorer.fixed_selections) == set(dataset[explorer.variable].dims) - {
        explorer.x_dim,
        explorer.y_dim,
    }
    state = explorer.get_state()
    restored = Explorer(dataset)
    restored.set_state(state)
    assert restored.get_state() == state


def test_plot_updates(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    old = explorer._plot.object
    first = next(iter(explorer._dimension_widgets.values()))
    if len(first.options) > 1:
        drag(first, first.options[1])
    assert explorer._plot.object is not None
    assert explorer._plot.object is not old


@pytest.mark.parametrize("plot_type", ["heatmap", "filled contour", "contour lines", "image"])
def test_every_plot_type_renders_without_error(dataset: xr.Dataset, plot_type: str) -> None:
    explorer = Explorer(dataset)
    explorer.plot_type_widget.value = plot_type
    assert explorer._plot.object is not None
    assert explorer._message.alert_type != "danger"


def test_axis_checkbox_matrix_is_synchronized(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    assert set(explorer._axis_x_checks) == set(dataset[explorer.variable].dims)
    assert set(explorer._axis_y_checks) == set(dataset[explorer.variable].dims)
    assert explorer._axis_x_checks[explorer.x_dim].value
    assert explorer._axis_y_checks[explorer.y_dim].value
    assert explorer._axis_x_checks[explorer.y_dim].disabled
    assert explorer._axis_y_checks[explorer.x_dim].disabled

    new_x = next(
        dim
        for dim in dataset[explorer.variable].dims
        if dim not in {explorer.x_dim, explorer.y_dim}
    )
    explorer._axis_x_checks[new_x].value = True
    assert explorer.x_dim == new_x
    assert sum(check.value for check in explorer._axis_x_checks.values()) == 1
    assert explorer._axis_y_checks[new_x].disabled
    assert set(explorer.fixed_selections) == set(dataset[explorer.variable].dims) - {
        explorer.x_dim,
        explorer.y_dim,
    }


def test_dataset_section_lists_variable_dimensions(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    text = explorer._variable_dimensions_pane().object
    for name, data in dataset.data_vars.items():
        assert f"`{name}`" in text
        assert all(f"{dim}: {data.sizes[dim]}" in text for dim in data.dims)


def test_dimension_labels_include_unique_value_counts(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    for dim, widget in explorer._dimension_widgets.items():
        assert f"({dataset.sizes[dim]} values)" in widget.name
    for dim in dataset[explorer.variable].dims:
        assert f"({dataset.sizes[dim]} values)" in explorer._dimension_label(dim)


def test_pareto_slider_tracks_output_and_draws_supported_contour(
    dataset: xr.Dataset,
) -> None:
    explorer = Explorer(
        dataset,
        default_x="drum_angle",
        default_y="fuel_temperature",
    )
    drag(explorer._dimension_widgets["flow_rate"], 10.0)
    result, _ = explorer.current_slice()
    finite = result.values[np.isfinite(result.values)]
    assert explorer.pareto_value_widget.start == pytest.approx(float(finite.min()))
    assert explorer.pareto_value_widget.end == pytest.approx(float(finite.max()))

    drag(explorer.pareto_value_widget, float(np.median(finite)))
    explorer.pareto_widget.value = True
    assert explorer._plot.object is not None
    assert "contour unavailable" not in str(explorer._message.object).lower()


def test_pareto_contour_reports_incomplete_support(dataset: xr.Dataset) -> None:
    explorer = Explorer(
        dataset,
        default_x="drum_angle",
        default_y="fuel_temperature",
    )
    explorer.pareto_widget.value = True
    assert "semantically invalid cells" in str(explorer._message.object)


def test_pareto_contour_draws_with_isolated_missing_grid_point(
    dataset: xr.Dataset,
) -> None:
    complete = dataset.copy(deep=True)
    complete["status"][:] = 0
    complete["k_eff"][:] = complete["k_eff"].fillna(1.0)
    complete["k_eff"].loc[
        dict(
            fuel_temperature=850,
            drum_angle=50,
            pressure=2.5,
            flow_rate=10,
            burnup=5,
        )
    ] = np.nan
    explorer = Explorer(
        complete,
        default_variable="k_eff",
        default_x="drum_angle",
        default_y="fuel_temperature",
    )
    drag(explorer._dimension_widgets["pressure"], 2.5)
    drag(explorer._dimension_widgets["flow_rate"], 10.0)
    drag(explorer._dimension_widgets["burnup"], 5.0)
    explorer.pareto_widget.value = True
    assert explorer._plot.object is not None
    assert "contour unavailable" not in str(explorer._message.object).lower()


def test_tapping_a_filter_point_pins_and_shows_the_slicer(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    view = explorer.filter_view
    rows = view._selected_rows
    assert rows is not None

    row = rows.iloc[7]
    fixed = [
        dim
        for dim in explorer.schema.variables[explorer.variable].dims
        if dim not in {explorer.x_dim, explorer.y_dim}
    ]
    assert fixed, "dataset needs a non-axis dimension to pin"

    explorer.view.active = 0
    view._on_point_tapped([7])

    assert explorer.view.active == 1
    for dim in fixed:
        assert explorer._dimension_widgets[dim].value == row[dim]
    assert explorer.variable == view.variable


def test_open_design_point_snaps_to_the_nearest_grid_value(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    fixed = next(
        dim
        for dim in explorer.schema.variables[explorer.variable].dims
        if dim not in {explorer.x_dim, explorer.y_dim}
    )
    levels = list(explorer._dimension_widgets[fixed].options)
    target = levels[-1]

    explorer.open_design_point({fixed: float(target) + 1e-9})
    assert explorer._dimension_widgets[fixed].value == target
    assert explorer.view.active == 1


def test_selected_design_point_is_outlined_in_the_slicer(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    assert explorer._selection_outline() is None

    coords = {
        dim: float(explorer.dataset.coords[dim].values[1])
        for dim in explorer.schema.variables[explorer.variable].dims
    }
    explorer.open_design_point(coords)

    outline = explorer._selection_outline()
    assert outline is not None
    left, bottom, right, top = outline.lbrt
    x_values = np.asarray(explorer.dataset.coords[explorer.x_dim].values, dtype=float)
    y_values = np.asarray(explorer.dataset.coords[explorer.y_dim].values, dtype=float)
    assert left < x_values[1] < right
    assert bottom < y_values[1] < top

    bounds = explorer._plot.object.traverse(lambda item: item, specs=[hv.Bounds])
    assert len(bounds) == 1
    assert bounds[0].opts.get(backend="bokeh").kwargs["color"] == HIGHLIGHT_COLOR


def test_tapping_the_slicer_selects_the_point_in_both_views(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    x_value = float(explorer.dataset.coords[explorer.x_dim].values[1])
    y_value = float(explorer.dataset.coords[explorer.y_dim].values[2])

    # Drive the stream the way the browser does: a position inside the cell.
    assert explorer._tap_stream is not None
    explorer._tap_stream.event(x=x_value + 1e-3, y=y_value - 1e-3)

    expected = dict(explorer.fixed_selections) | {explorer.x_dim: x_value, explorer.y_dim: y_value}
    assert explorer._selected_point == expected
    assert explorer._selection_outline() is not None
    assert explorer.filter_view._selected_point == expected
    labels = [
        element.label
        for element in explorer.filter_view._plot.object.traverse(
            lambda item: item, specs=[hv.Points]
        )
    ]
    assert "Selected point" in labels


def test_taps_beyond_the_drawn_cells_select_nothing(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    x_values = np.asarray(explorer.dataset.coords[explorer.x_dim].values, dtype=float)
    y_value = float(explorer.dataset.coords[explorer.y_dim].values[0])

    explorer._on_tap(float(x_values.max()) * 10 + 1, y_value)
    assert explorer._selected_point == {}
    assert explorer._sensitivity_box.visible is False

    explorer._on_tap(None, y_value)
    assert explorer._selected_point == {}


def test_sensitivity_table_reports_slopes_at_the_selected_point(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    assert explorer._sensitivity_box.visible is False
    assert explorer._sensitivity_box in explorer.slicer_view[1]

    explorer._on_tap(
        float(explorer.dataset.coords[explorer.x_dim].values[1]),
        float(explorer.dataset.coords[explorer.y_dim].values[1]),
    )

    assert explorer._sensitivity_box.visible is True
    table = explorer._sensitivity_table.object
    assert list(table.index) == ["Multiplication factor [dimensionless]", "Peak temperature [K]"]
    assert "Control drum angle [degree]" in table.columns
    assert table.loc["Multiplication factor [dimensionless]", "Pressure [MPa]"] == "0.002"
    for dim in explorer.schema.variables[explorer.variable].dims:
        assert f"`{dim}` =" in explorer._sensitivity_title.object


def test_tapping_a_filter_point_shows_its_sensitivities(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    explorer.filter_view._on_point_tapped([11])
    assert explorer.view.active == 1
    assert explorer._sensitivity_box.visible is True
    assert explorer._sensitivity_table.object is not None


def test_highlight_colour_is_outside_viridis() -> None:
    assert HIGHLIGHT_COLOR.lower() not in {color.lower() for color in VIRIDIS}


def test_slicer_heatmap_uses_viridis(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    meshes = explorer._plot.object.traverse(lambda item: item, specs=[hv.QuadMesh])
    assert meshes
    assert meshes[0].opts.get(backend="bokeh").kwargs["cmap"] is VIRIDIS


def test_slicer_plot_pane_has_a_fixed_height(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    assert explorer._plot.height == 590
    assert explorer._plot.sizing_mode == "stretch_width"
