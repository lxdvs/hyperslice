from __future__ import annotations

import panel as pn
import xarray as xr
from bokeh.models import Button as BokehButton
from bokeh.models import Column as BokehColumn
from bokeh.models import Plot
from panel.links import Callback

from hyperslice import Explorer
from hyperslice.filtering import FilterView
from hyperslice.schema import inspect_dataset
from hyperslice.widgets import RESET_VIEW_JS, reset_view_button


def test_reset_view_button_targets_its_container_with_a_reset_script() -> None:
    container = pn.Column(pn.pane.Markdown("placeholder"))
    button = reset_view_button(container)
    assert button.name == "Reset plot bounds"
    assert button.button_type == "primary"
    assert button.icon == "zoom-reset"
    (callback,) = Callback.registry[button]
    assert callback.args == {"container": container}
    assert callback.code == {"event:button_click": RESET_VIEW_JS}
    assert "reset.emit()" in RESET_VIEW_JS


def _rendered_reset_callback(view: pn.viewable.Viewable) -> tuple[BokehColumn, Plot]:
    """Render *view* and return the model the reset script receives plus the plot in it."""
    root = view.get_root()
    button = next(
        model for model in root.select({"type": BokehButton}) if model.label == "Reset plot bounds"
    )
    (custom_js,) = button.js_event_callbacks["button_click"]
    container = custom_js.args["container"]
    assert isinstance(container, BokehColumn)
    (plot,) = container.select({"type": Plot})
    return container, plot


def test_slicer_reset_button_sits_below_the_plot(dataset: xr.Dataset) -> None:
    explorer = Explorer(dataset)
    main = explorer.slicer_view[1]
    objects = list(main)
    assert objects.index(explorer._reset_view) == objects.index(explorer._plot_box) + 1
    assert explorer._plot in explorer._plot_box
    container, plot = _rendered_reset_callback(explorer.slicer_view)
    assert plot in container.select({"type": Plot})


def test_filter_reset_button_sits_below_the_plot(dataset: xr.Dataset) -> None:
    view = FilterView(dataset, inspect_dataset(dataset))
    main = view.view[1]
    objects = list(main)
    assert objects.index(view._reset_view) == objects.index(view._plot_box) + 1
    assert view._plot in view._plot_box
    _rendered_reset_callback(view.view)


def test_reset_container_survives_a_redraw(dataset: xr.Dataset) -> None:
    """The script must reach the plot drawn *after* a redraw, not a stale model."""
    view = FilterView(dataset, inspect_dataset(dataset))
    container, before = _rendered_reset_callback(view.view)
    view.point_size_widget.param.trigger("value_throttled")
    view._update()
    (after,) = container.select({"type": Plot})
    assert after is not before
