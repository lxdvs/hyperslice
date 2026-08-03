from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hyperslice import Explorer


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
        first.value = first.options[1]
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
    explorer._dimension_widgets["flow_rate"].value = 10.0
    result, _ = explorer.current_slice()
    finite = result.values[np.isfinite(result.values)]
    assert explorer.pareto_value_widget.start == pytest.approx(float(finite.min()))
    assert explorer.pareto_value_widget.end == pytest.approx(float(finite.max()))

    explorer.pareto_value_widget.value = float(np.median(finite))
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
    explorer._dimension_widgets["pressure"].value = 2.5
    explorer._dimension_widgets["flow_rate"].value = 10.0
    explorer._dimension_widgets["burnup"].value = 5.0
    explorer.pareto_widget.value = True
    assert explorer._plot.object is not None
    assert "contour unavailable" not in str(explorer._message.object).lower()
