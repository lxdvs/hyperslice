from __future__ import annotations

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
