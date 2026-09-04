"""Interactive HyperSlice explorer and view-model."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import holoviews as hv
import numpy as np
import panel as pn
import xarray as xr

from hyperslice.colors import HIGHLIGHT_COLOR
from hyperslice.exceptions import DatasetSchemaError, SliceError
from hyperslice.export import bytes_io, csv_bytes, netcdf_bytes, save_png
from hyperslice.filtering import FilterView, watch_settled
from hyperslice.loading import DatasetSource, load_dataset
from hyperslice.plotting import build_plot
from hyperslice.schema import DatasetSchema, inspect_dataset
from hyperslice.sensitivity import labelled_frame, sensitivity_frame
from hyperslice.slicing import SliceMethod, make_slice
from hyperslice.status import SliceStatus, apply_strict_validity, classify_slice

LOGGER = logging.getLogger(__name__)


def _cell_edges(values: np.ndarray, index: int) -> tuple[float, float]:
    """Half-way boundaries of the heatmap cell centred on ``values[index]``."""
    center = float(values[index])
    if values.size == 1:
        span = max(abs(center) * 0.05, 0.5)
        return center - span, center + span
    below = (
        float(values[index - 1]) if index > 0 else center - (float(values[1]) - float(values[0]))
    )
    above = (
        float(values[index + 1])
        if index < values.size - 1
        else center + (float(values[-1]) - float(values[-2]))
    )
    return (center + below) / 2, (center + above) / 2


def _coordinate_text(value: Any) -> str:
    """Render a coordinate value compactly, without float noise."""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _nearest_option(options: list[Any], value: Any) -> Any:
    """Pick the option matching *value*, falling back to the closest number."""
    if value in options:
        return value
    try:
        target = float(value)
        return min(options, key=lambda option: abs(float(option) - target))
    except (TypeError, ValueError):
        return options[0] if options else value


pn.extension(sizing_mode="stretch_width")

TAB_STYLES = """
:host {
  --hyperslice-navy: #17324d;
  --hyperslice-blue: #2d6f9f;
  --hyperslice-surface: #f4f7fa;
}
.bk-header {
  position: sticky;
  top: 0;
  z-index: 1000;
  padding: 10px 16px 0 16px;
  background: rgba(244, 247, 250, 0.97);
  border-bottom: 1px solid #cbd6df;
  box-shadow: 0 3px 10px rgba(23, 50, 77, 0.12);
  backdrop-filter: blur(8px);
}
.bk-headers-wrapper {
  gap: 6px;
}
.bk-tab {
  min-width: 120px;
  padding: 10px 20px;
  border: 1px solid transparent;
  border-radius: 8px 8px 0 0;
  color: #536878;
  font-size: 14px;
  font-weight: 650;
  letter-spacing: 0.02em;
  transition: color 120ms ease, background-color 120ms ease,
              border-color 120ms ease;
}
.bk-tab:hover {
  color: var(--hyperslice-navy);
  background: #e5edf4;
}
.bk-tab.bk-active {
  color: white;
  background: var(--hyperslice-blue);
  border-color: var(--hyperslice-blue);
}
"""


class Explorer:
    """Explore any multidimensional numeric variable in an xarray Dataset."""

    def __init__(
        self,
        source: DatasetSource,
        *,
        default_variable: str | None = None,
        default_x: str | None = None,
        default_y: str | None = None,
        status_variable: str | None = None,
        validity_variable: str | None = None,
        initial_method: SliceMethod = "exact",
    ) -> None:
        self.source = source
        self.dataset = load_dataset(source)
        self.schema: DatasetSchema = inspect_dataset(self.dataset)
        self.status_variable = status_variable or self._default_status()
        self.validity_variable = validity_variable
        variables = [
            name for name, info in self.schema.variables.items() if not info.constant
        ] or list(self.schema.variables)
        variable = default_variable if default_variable in variables else variables[0]
        axis_dims = self._axis_dimensions(variable)
        if len(axis_dims) < 2:
            raise DatasetSchemaError(
                f"Variable '{variable}' needs at least two numeric dimensions "
                "for a two-dimensional slice."
            )
        x_dim = default_x if default_x in axis_dims else axis_dims[-1]
        y_dim = (
            default_y
            if default_y in axis_dims and default_y != x_dim
            else next(dim for dim in reversed(axis_dims) if dim != x_dim)
        )
        self._selections: dict[str, Any] = {}
        self.variable_widget = pn.widgets.Select(
            name="Output variable", options=variables, value=variable
        )
        self.x_widget = pn.widgets.Select(name="X axis", options=axis_dims, value=x_dim)
        self.y_widget = pn.widgets.Select(
            name="Y axis",
            options=[dim for dim in axis_dims if dim != x_dim],
            value=y_dim,
        )
        self.method_widget = pn.widgets.Select(
            name="Slicing method", options=["exact", "nearest", "linear"], value=initial_method
        )
        self.plot_type_widget = pn.widgets.Select(
            name="Plot type", options=["heatmap", "filled contour", "contour lines", "image"]
        )
        self.pareto_value_widget = pn.widgets.FloatSlider(
            name="Output contour", start=0.0, end=1.0, value=0.5, step=0.01
        )
        self.pareto_widget = pn.widgets.Checkbox(name="Pareto", value=False, width=75)
        self._syncing_pareto_value = False
        self.show_samples_widget = pn.widgets.Checkbox(
            name="Show original sample locations", value=False
        )
        self.show_invalid_widget = pn.widgets.Checkbox(name="Show invalid regions", value=True)
        self._dimension_box = pn.Column()
        self._dimension_widgets: dict[str, pn.widgets.Widget] = {}
        self._axis_matrix = pn.Column()
        self._axis_x_checks: dict[str, pn.widgets.Checkbox] = {}
        self._axis_y_checks: dict[str, pn.widgets.Checkbox] = {}
        self._syncing_axis_matrix = False
        self._message = pn.pane.Alert("", alert_type="info", visible=False)
        self._slice_info = pn.pane.Markdown()
        # Fixed height, matching the filter view: a pane free to stretch grows
        # with whatever sits below it, pushing the plot past the window and
        # leaving the sensitivity table off screen.
        self._plot = pn.pane.HoloViews(height=590, sizing_mode="stretch_width")
        self._csv = pn.widgets.FileDownload(
            label="Download CSV", callback=self._csv_download, filename="hyperslice.csv"
        )
        self._netcdf = pn.widgets.FileDownload(
            label="Download NetCDF", callback=self._netcdf_download, filename="hyperslice.nc"
        )
        self._png = pn.widgets.FileDownload(
            label="Download PNG", callback=self._png_download, filename="hyperslice.png"
        )
        self._selected_point: dict[str, Any] = {}
        self._sensitivity_title = pn.pane.Markdown(margin=(0, 0, 4, 0))
        self._sensitivity_table = pn.pane.DataFrame(
            index=True, sizing_mode="stretch_width", margin=(0, 0, 8, 0)
        )
        self._sensitivity_box = pn.Column(
            self._sensitivity_title,
            self._sensitivity_table,
            visible=False,
            sizing_mode="stretch_width",
        )
        self._tap_stream: hv.streams.Tap | None = None
        self._current_slice: xr.DataArray | None = None
        self._current_status: SliceStatus | None = None
        self._wire_events()
        self._rebuild_dimension_widgets()
        self._rebuild_axis_matrix()
        self._update()
        self.slicer_view = self._build_view()
        self.filter_view = FilterView(
            self.dataset, self.schema, on_point_selected=self.open_design_point
        )
        self.view = pn.Tabs(
            ("Filter", self.filter_view.view),
            ("Slicer", self.slicer_view),
            active=0,
            stylesheets=[TAB_STYLES],
            sizing_mode="stretch_both",
        )

    def open_design_point(self, coordinates: dict[str, Any], variable: str | None = None) -> None:
        """Pin the slicer to *coordinates* and bring it to the front.

        Dimensions currently on an axis have no fixed-coordinate widget and are
        left alone; the rest are moved to the nearest available grid value.
        """
        self._selected_point = dict(coordinates)
        if variable and variable in self.variable_widget.options:
            self.variable_widget.value = variable
        for dim, value in coordinates.items():
            widget = self._dimension_widgets.get(dim)
            if widget is not None:
                widget.value = _nearest_option(list(widget.options), value)
        self._update()
        self.view.active = 1

    def _watch_taps(self, plot: Any) -> None:
        """Re-link the tap stream to the freshly built plot.

        Every update replaces the figure, so the stream is rebuilt alongside it
        and kept on the explorer — HoloViews would otherwise let it be collected.
        """
        elements = plot.traverse(lambda item: item, specs=[hv.Element])
        if not elements:
            return
        stream = hv.streams.Tap(source=elements[0])
        stream.add_subscriber(self._on_tap)
        self._tap_stream = stream

    def _on_tap(self, x: float | None = None, y: float | None = None) -> None:
        """Select the design point under the tapped cell in both views."""
        if x is None or y is None:
            return
        point = dict(self.fixed_selections)
        for dim, position in ((self.x_dim, x), (self.y_dim, y)):
            values = np.asarray(self.dataset.coords[dim].values, dtype=float)
            index = int(np.argmin(np.abs(values - float(position))))
            low, high = _cell_edges(values, index)
            # A tap beyond the drawn cells names no design point.
            if not low <= float(position) <= high:
                return
            point[dim] = self.dataset.coords[dim].values[index].item()
        self._selected_point = point
        self._update()
        filter_view = getattr(self, "filter_view", None)
        if filter_view is not None:
            filter_view.select_point(point)

    def _update_sensitivities(self) -> None:
        """Tabulate each output's local slope against each input at the selection."""
        if not self._selected_point:
            self._sensitivity_box.visible = False
            return
        frame = sensitivity_frame(self.dataset, self.schema, self._selected_point)
        self._sensitivity_table.object = labelled_frame(frame, self.schema)
        described = ", ".join(
            f"`{dim}` = {_coordinate_text(self._selected_point[dim])}"
            for dim in self.schema.coordinates
            if dim in self._selected_point
        )
        self._sensitivity_title.object = (
            f"### Sensitivities\nAt {described}. Each cell is the partial derivative of the "
            "row output with respect to the column input, differenced across "
            "neighbouring samples on the grid."
        )
        self._sensitivity_box.visible = True

    def _selection_outline(self) -> hv.Bounds | None:
        """Rectangle around the selected design point's cell, if it is on screen."""
        if not self._selected_point:
            return None
        edges: list[float] = []
        for dim in (self.x_dim, self.y_dim):
            if dim not in self._selected_point:
                return None
            values = np.asarray(self.dataset.coords[dim].values, dtype=float)
            try:
                target = float(self._selected_point[dim])
            except (TypeError, ValueError):
                return None
            index = int(np.argmin(np.abs(values - target)))
            edges.extend(_cell_edges(values, index))
        left, right, bottom, top = edges
        return hv.Bounds((left, bottom, right, top))

    def _default_status(self) -> str | None:
        return self.schema.status_candidates[0] if self.schema.status_candidates else None

    def _axis_dimensions(self, variable: str | None = None) -> list[str]:
        """Return dimensions suitable for continuous two-dimensional plots."""
        selected = variable or self.variable
        continuous = [
            dim
            for dim in self.schema.variables[selected].dims
            if not self.schema.coordinates[dim].categorical
        ]
        varying = [dim for dim in continuous if not self.schema.coordinates[dim].constant]
        return varying if len(varying) >= 2 else continuous

    @property
    def variable(self) -> str:
        """Current output variable name."""
        return str(self.variable_widget.value)

    @property
    def x_dim(self) -> str:
        """Current horizontal dimension."""
        return str(self.x_widget.value)

    @property
    def y_dim(self) -> str:
        """Current vertical dimension."""
        return str(self.y_widget.value)

    @property
    def fixed_selections(self) -> dict[str, Any]:
        """Current values for every non-displayed dimension."""
        return {dim: widget.value for dim, widget in self._dimension_widgets.items()}

    def _wire_events(self) -> None:
        self.variable_widget.param.watch(self._on_variable, "value")
        self.x_widget.param.watch(self._on_x, "value")
        self.y_widget.param.watch(self._on_y, "value")
        watch_settled(self.pareto_value_widget, self._on_pareto_value)
        for widget in (
            self.method_widget,
            self.plot_type_widget,
            self.pareto_widget,
            self.show_samples_widget,
            self.show_invalid_widget,
        ):
            widget.param.watch(lambda _event: self._update(), "value")

    def _on_pareto_value(self, _event: Any) -> None:
        if not self._syncing_pareto_value:
            self._update()

    def _on_variable(self, _event: Any) -> None:
        dims = self._axis_dimensions()
        if len(dims) < 2:
            self._message.object = (
                f"Variable '{self.variable}' has fewer than two numeric dimensions."
            )
            self._message.alert_type = "danger"
            self._message.visible = True
            return
        self.x_widget.options = dims
        if self.x_dim not in dims:
            self.x_widget.value = dims[-1]
        self.y_widget.options = [dim for dim in dims if dim != self.x_dim]
        if self.y_dim not in self.y_widget.options:
            self.y_widget.value = self.y_widget.options[-1]
        self._rebuild_dimension_widgets()
        self._rebuild_axis_matrix()
        self._update()

    def _on_x(self, _event: Any) -> None:
        dims = self._axis_dimensions()
        self.y_widget.options = [dim for dim in dims if dim != self.x_dim]
        if self.y_dim == self.x_dim or self.y_dim not in self.y_widget.options:
            self.y_widget.value = self.y_widget.options[-1]
        self._rebuild_dimension_widgets()
        self._sync_axis_matrix()
        self._update()

    def _on_y(self, _event: Any) -> None:
        if self.y_dim == self.x_dim:
            alternatives = [dim for dim in self.x_widget.options if dim != self.x_dim]
            self.y_widget.value = alternatives[-1]
        self._rebuild_dimension_widgets()
        self._sync_axis_matrix()
        self._update()

    def _variable_dimensions_pane(self) -> pn.pane.Markdown:
        lines = ["#### Variables"]
        for name, data in self.dataset.data_vars.items():
            dimensions = ", ".join(f"{dim}: {data.sizes[dim]}" for dim in data.dims)
            lines.append(f"- `{name}` — {dimensions or 'scalar'}")
        return pn.pane.Markdown("\n".join(lines), margin=(0, 0, 8, 0))

    def _dimension_label(self, dim: str, *, include_units: bool = True) -> str:
        """Return a sidebar label including metadata and unique coordinate count."""
        info = self.schema.coordinates[dim]
        units = f" [{info.units}]" if include_units and info.units else ""
        noun = "value" if info.size == 1 else "values"
        return f"{info.long_name}{units} ({info.size} {noun})"

    def _rebuild_axis_matrix(self) -> None:
        """Build the checkbox axis selector for the current output variable."""
        self._axis_x_checks = {}
        self._axis_y_checks = {}
        rows: list[Any] = [
            pn.Row(
                pn.pane.Markdown("**Dimension**", width=155, margin=(5, 5)),
                pn.pane.Markdown("**X Axis**", width=65, margin=(5, 0)),
                pn.pane.Markdown("**Y Axis**", width=65, margin=(5, 0)),
                sizing_mode="fixed",
            )
        ]
        for dim in self.schema.variables[self.variable].dims:
            x_check = pn.widgets.Checkbox(
                name="",
                value=dim == self.x_dim,
                width=65,
                align="center",
            )
            y_check = pn.widgets.Checkbox(
                name="",
                value=dim == self.y_dim,
                width=65,
                align="center",
            )
            x_check.param.watch(
                lambda event, selected_dim=dim: self._on_axis_check(selected_dim, "x", event.new),
                "value",
            )
            y_check.param.watch(
                lambda event, selected_dim=dim: self._on_axis_check(selected_dim, "y", event.new),
                "value",
            )
            self._axis_x_checks[dim] = x_check
            self._axis_y_checks[dim] = y_check
            label = self._dimension_label(dim)
            rows.append(
                pn.Row(
                    pn.pane.Markdown(f"`{label}`", width=155, margin=(5, 5)),
                    x_check,
                    y_check,
                    sizing_mode="fixed",
                )
            )
        self._axis_matrix.objects = rows
        self._sync_axis_matrix()

    def _on_axis_check(self, dim: str, axis: str, selected: bool) -> None:
        if self._syncing_axis_matrix:
            return
        current = self.x_dim if axis == "x" else self.y_dim
        if selected:
            if axis == "x":
                self.x_widget.value = dim
            else:
                self.y_widget.value = dim
        elif dim == current:
            self._sync_axis_matrix()

    def _sync_axis_matrix(self) -> None:
        """Synchronize checkboxes and disable selections that would collide."""
        if not self._axis_x_checks:
            return
        self._syncing_axis_matrix = True
        try:
            for dim, checkbox in self._axis_x_checks.items():
                checkbox.value = dim == self.x_dim
                checkbox.disabled = self.schema.coordinates[dim].categorical or dim == self.y_dim
            for dim, checkbox in self._axis_y_checks.items():
                checkbox.value = dim == self.y_dim
                checkbox.disabled = self.schema.coordinates[dim].categorical or dim == self.x_dim
        finally:
            self._syncing_axis_matrix = False

    def _coordinate_widget(self, dim: str) -> pn.widgets.Widget:
        coord = self.dataset.coords[dim]
        values = list(coord.values)
        info = self.schema.coordinates[dim]
        label = self._dimension_label(dim)
        previous = self._selections.get(dim)
        value = previous if previous in values else values[0]
        if not info.categorical and len(values) <= 50:
            return pn.widgets.DiscreteSlider(name=label, options=values, value=value)
        return pn.widgets.Select(name=label, options=values, value=value)

    def _rebuild_dimension_widgets(self) -> None:
        self._selections.update(
            {dim: widget.value for dim, widget in self._dimension_widgets.items()}
        )
        dims = self.schema.variables[self.variable].dims
        self._dimension_widgets = {
            dim: self._coordinate_widget(dim) for dim in dims if dim not in {self.x_dim, self.y_dim}
        }
        for widget in self._dimension_widgets.values():
            watch_settled(widget, lambda _event: self._update())
        # Constant dimensions still pin the slice, but a one-option slicer is noise.
        self._dimension_box.objects = [
            widget
            for dim, widget in self._dimension_widgets.items()
            if not self.schema.coordinates[dim].constant
        ]

    def _status_data(self) -> xr.DataArray | None:
        return self.dataset[self.status_variable] if self.status_variable else None

    def _validity_data(self) -> xr.DataArray | None:
        return self.dataset[self.validity_variable] if self.validity_variable else None

    def current_slice(self) -> tuple[xr.DataArray, SliceStatus]:
        """Compute the current result and aligned semantic status."""
        data = self.dataset[self.variable]
        method: SliceMethod = self.method_widget.value
        if method == "linear":
            data = apply_strict_validity(data, self._validity_data(), self._status_data())
        result = make_slice(
            data,
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            selections=self.fixed_selections,
            method=method,
        )
        result.name = self.variable
        status = classify_slice(
            result,
            status=self._status_data(),
            validity=self._validity_data(),
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            selections=self.fixed_selections,
            method=method,
        )
        return result, status

    def _title(self) -> str:
        info = self.schema.variables[self.variable]
        fixed = ", ".join(f"{key} = {value}" for key, value in self.fixed_selections.items())
        return f"{info.long_name} — {self.x_dim} by {self.y_dim}" + (f" — {fixed}" if fixed else "")

    def _update(self) -> None:
        try:
            result, status = self.current_slice()
            self._current_slice, self._current_status = result, status
            self._sync_pareto_range(result)
            plot_arguments = {
                "dataset": self.dataset,
                "x_dim": self.x_dim,
                "y_dim": self.y_dim,
                "plot_type": self.plot_type_widget.value,
                "title": self._title(),
                "status": status,
                "show_samples": self.show_samples_widget.value,
                "show_invalid": self.show_invalid_widget.value,
            }
            pareto_error: str | None = None
            try:
                plot = build_plot(
                    result,
                    **plot_arguments,
                    pareto=self.pareto_widget.value,
                    pareto_value=self.pareto_value_widget.value,
                )
            except SliceError as exc:
                pareto_error = str(exc)
                plot = build_plot(result, **plot_arguments)
            self._watch_taps(plot)
            outline = self._selection_outline()
            if outline is not None:
                plot = plot * outline.opts(color=HIGHLIGHT_COLOR, line_width=3, backend="bokeh")
            self._plot.object = plot
            self._update_sensitivities()
            counts = status.counts
            cells = int(np.prod(result.shape))
            self._slice_info.object = (
                f"### Slice information\n"
                f"`{self.variable}` · {result.dtype} · {result.shape} · {cells} cells  \n"
                f"**{counts['valid']} valid** · **{counts['invalid']} invalid** · "
                f"**{counts['missing']} missing**  \n"
                f"Method: **{self.method_widget.value}** "
                f"({'interpolated' if self.method_widget.value == 'linear' else 'grid selection'})"
            )
            affected = counts["invalid"] + counts["missing"]
            self._message.visible = affected > 0 or pareto_error is not None
            messages = []
            if affected:
                messages.append(
                    f"This slice contains {counts['invalid']} invalid and "
                    f"{counts['missing']} missing cells."
                )
            if pareto_error:
                messages.append(pareto_error)
            self._message.object = " ".join(messages)
            self._message.alert_type = "warning"
        except Exception as exc:
            LOGGER.exception("Slice update failed")
            self._message.object = str(exc)
            self._message.alert_type = "danger"
            self._message.visible = True

    def _sync_pareto_range(self, data: xr.DataArray) -> None:
        """Keep the output-contour slider within the finite displayed range."""
        finite = np.asarray(data.values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            self.pareto_value_widget.disabled = True
            return
        low, high = float(finite.min()), float(finite.max())
        if low == high:
            high = low + max(abs(low) * 1e-6, 1e-9)
        previous = float(self.pareto_value_widget.value)
        self._syncing_pareto_value = True
        try:
            self.pareto_value_widget.start = low
            self.pareto_value_widget.end = high
            self.pareto_value_widget.step = (high - low) / 200
            self.pareto_value_widget.value = min(max(previous, low), high)
            self.pareto_value_widget.disabled = False
        finally:
            self._syncing_pareto_value = False

    def _csv_download(self) -> Any:
        result, status = self.current_slice()
        return bytes_io(csv_bytes(result, x_dim=self.x_dim, y_dim=self.y_dim, status=status.status))

    def _netcdf_download(self) -> Any:
        result, status = self.current_slice()
        return bytes_io(
            netcdf_bytes(
                result,
                selections=self.fixed_selections,
                method=self.method_widget.value,
                source_attrs=dict(self.dataset.attrs),
                status=status.status,
            )
        )

    def _png_download(self) -> Any:
        """Render the visible plot through Bokeh's browser-based PNG exporter."""
        with NamedTemporaryFile(suffix=".png", delete=False) as temporary:
            path = Path(temporary.name)
        try:
            save_png(self._plot.object, path)
            return bytes_io(path.read_bytes())
        finally:
            path.unlink(missing_ok=True)

    def _build_view(self) -> pn.viewable.Viewable:
        source_name = (
            str(self.source) if not isinstance(self.source, xr.Dataset) else "In-memory Dataset"
        )
        controls = pn.Column(
            pn.pane.Markdown(f"## HyperSlice\n`{source_name}`"),
            self.variable_widget,
            pn.Row(self.x_widget, self.y_widget),
            pn.Row(self.method_widget, self.plot_type_widget),
            pn.Row(self.pareto_value_widget, self.pareto_widget),
            pn.pane.Markdown("### Fixed coordinates"),
            self._dimension_box,
            self.show_samples_widget,
            self.show_invalid_widget,
            pn.pane.Markdown("### Export"),
            pn.Row(self._csv, self._netcdf, self._png),
            pn.pane.Markdown(
                f"### Dataset\n{len(self.dataset.dims)} dimensions · "
                f"{len(self.schema.variables)} plottable variables"
            ),
            self._variable_dimensions_pane(),
            pn.pane.Markdown("#### Display axes", margin=(0, 0, 4, 0)),
            self._axis_matrix,
            width=360,
            height=720,
            sizing_mode="fixed",
            scroll=True,
            styles={"padding": "12px", "background": "#f5f7f9"},
        )
        main = pn.Column(
            self._message,
            self._plot,
            self._sensitivity_box,
            self._slice_info,
            sizing_mode="stretch_both",
        )
        return pn.Row(controls, main, sizing_mode="stretch_both", min_height=720)

    def show(self, **kwargs: Any) -> Any:
        """Open the explorer in a local Panel server."""
        return self.view.show(**kwargs)

    def servable(self, **kwargs: Any) -> pn.viewable.Viewable:
        """Mark and return the Panel view for ``panel serve``."""
        return self.view.servable(**kwargs)

    def get_state(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the current UI state."""

        def scalar(value: Any) -> Any:
            return value.item() if isinstance(value, np.generic) else value

        return {
            "variable": self.variable,
            "x_dim": self.x_dim,
            "y_dim": self.y_dim,
            "selections": {key: scalar(value) for key, value in self.fixed_selections.items()},
            "method": self.method_widget.value,
            "plot_type": self.plot_type_widget.value,
            "show_samples": self.show_samples_widget.value,
            "show_invalid": self.show_invalid_widget.value,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore a previously serialized state, validating available choices."""
        if state.get("variable") in self.variable_widget.options:
            self.variable_widget.value = state["variable"]
        if state.get("x_dim") in self.x_widget.options:
            self.x_widget.value = state["x_dim"]
        if state.get("y_dim") in self.y_widget.options:
            self.y_widget.value = state["y_dim"]
        self._rebuild_dimension_widgets()
        for dim, value in state.get("selections", {}).items():
            if dim in self._dimension_widgets and value in self._dimension_widgets[dim].options:
                self._dimension_widgets[dim].value = value
        for key, widget in (
            ("method", self.method_widget),
            ("plot_type", self.plot_type_widget),
        ):
            if state.get(key) in widget.options:
                widget.value = state[key]
        self.show_samples_widget.value = bool(state.get("show_samples", False))
        self.show_invalid_widget.value = bool(state.get("show_invalid", True))
        self._update()

    def __panel__(self) -> pn.viewable.Viewable:
        """Expose the explorer as an embeddable Panel object."""
        return self.view
