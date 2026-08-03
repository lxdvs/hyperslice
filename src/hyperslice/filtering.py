"""Point-cloud filtering view for complete dataset exploration."""

from __future__ import annotations

from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr

from hyperslice.colors import BLUE_PURPLE_RED
from hyperslice.schema import DatasetSchema, axis_label

#: Label styling for high-cardinality outputs — every sample an independent value.
CONTINUOUS_COLOR = "#0b8a3e"
CONTINUOUS_STYLESHEET = f"""
label, .bk-slider-title, .bk-input-group label {{
  color: {CONTINUOUS_COLOR};
  font-weight: 600;
}}
"""

#: Most distinct values a slider will mark with data ticklines; anything denser
#: would render as a solid bar rather than readable marks.
TICK_LIMIT = 160
TICK_COLOR = "#8b9aa9"


def tick_stylesheet(values: Any, start: float, end: float) -> str | None:
    """CSS drawing a tickline under a slider track at each distinct data value.

    Positions are percentages of the slider's ``start``-``end`` span, painted
    as stacked one-pixel background gradients on a pseudo-element so they never
    intercept pointer events or get hidden behind the selected-range bar.
    """
    finite = np.asarray(values, dtype=float)
    distinct = np.unique(finite[np.isfinite(finite)])
    if distinct.size < 2 or distinct.size > TICK_LIMIT or end <= start:
        return None
    positions = (distinct - start) / (end - start) * 100.0
    layers = ", ".join(f"linear-gradient({TICK_COLOR}, {TICK_COLOR})" for _ in positions)
    offsets = ", ".join(f"{position:.3f}% 0" for position in positions)
    sizes = ", ".join("1px 100%" for _ in positions)
    return f"""
.noUi-base::after {{
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  bottom: -6px;
  height: 5px;
  background-image: {layers};
  background-position: {offsets};
  background-size: {sizes};
  background-repeat: no-repeat;
  pointer-events: none;
}}
.noUi-target {{
  margin-bottom: 10px;
}}
"""


class FilterView:
    """Project every sample onto two axes and outline samples outside active ranges."""

    def __init__(self, dataset: xr.Dataset, schema: DatasetSchema) -> None:
        self.dataset = dataset
        self.schema = schema
        outputs = [name for name, info in schema.variables.items() if not info.constant] or list(
            schema.variables
        )
        variable = outputs[0]
        dims = list(schema.variables[variable].dims)
        # Any field can colour the cloud, inputs included.
        variables = outputs + [
            dim for dim in dims if not schema.coordinates[dim].constant and dim not in outputs
        ]
        varying_dims = [dim for dim in dims if not schema.coordinates[dim].constant] or dims
        compatible_outputs = [
            name
            for name, info in schema.variables.items()
            if set(info.dims).issubset(set(dims)) and not info.constant
        ]
        axis_options = varying_dims + [
            name for name in compatible_outputs if name not in varying_dims
        ]
        self.variable_widget = pn.widgets.Select(
            name="Z variable", options=variables, value=variable
        )
        self.x_widget = pn.widgets.Select(
            name="X axis", options=axis_options, value=varying_dims[-1]
        )
        self.y_widget = pn.widgets.Select(
            name="Y axis",
            options=[name for name in axis_options if name != varying_dims[-1]],
            value=varying_dims[-2] if len(varying_dims) > 1 else axis_options[0],
        )
        self._axis_matrix = pn.Column()
        self._axis_x_checks: dict[str, pn.widgets.Checkbox] = {}
        self._axis_y_checks: dict[str, pn.widgets.Checkbox] = {}
        self._syncing_axes = False
        self._filter_grid = pn.GridBox(ncols=2, sizing_mode="stretch_width")
        self._filter_widgets: dict[str, pn.widgets.Widget] = {}
        self._missing_widgets: dict[str, pn.widgets.Checkbox] = {}
        self._saved_ranges: dict[str, Any] = {}
        self._saved_missing: dict[str, bool] = {}
        self.nonmatching_widget = pn.widgets.RadioButtonGroup(
            name="Non-matching points",
            options=["Outline", "Hide"],
            value="Outline",
            button_type="default",
        )
        self._plot = pn.pane.HoloViews(min_height=540, sizing_mode="stretch_both")
        self._summary = pn.pane.Markdown()
        self._coverage = pn.pane.Alert("", alert_type="warning", visible=False)
        self._message = pn.pane.Alert("", alert_type="danger", visible=False)
        self.variable_widget.param.watch(self._on_variable, "value")
        self.x_widget.param.watch(self._on_x, "value")
        self.y_widget.param.watch(self._on_y, "value")
        self.nonmatching_widget.param.watch(lambda _event: self._update(), "value")
        self._rebuild_axis_matrix()
        self._rebuild_filters()
        self._update()
        self.view = self._build_view()

    @property
    def variable(self) -> str:
        """Selected response used to color the point cloud."""
        return str(self.variable_widget.value)

    @property
    def x_dim(self) -> str:
        """Selected horizontal coordinate."""
        return str(self.x_widget.value)

    @property
    def y_dim(self) -> str:
        """Selected vertical coordinate."""
        return str(self.y_widget.value)

    def _dimension_label(self, dim: str) -> str:
        info = self.schema.coordinates[dim]
        units = f" [{info.units}]" if info.units else ""
        noun = "value" if info.size == 1 else "values"
        return f"{info.long_name}{units} ({info.size} {noun})"

    def _output_label(self, name: str) -> str:
        info = self.schema.variables[name]
        return info.long_name + (f" [{info.units}]" if info.units else "")

    def _axis_label(self, name: str) -> str:
        if name in self.schema.coordinates:
            return axis_label(self.dataset, name)
        return self._output_label(name)

    def _varying_dimensions(self) -> list[str]:
        """Input dimensions that take more than one value."""
        dimensions = list(self._grid_dimensions())
        return [
            dim for dim in dimensions if not self.schema.coordinates[dim].constant
        ] or dimensions

    def _varying_outputs(self) -> list[str]:
        """Compatible outputs that are not fixed at a single value everywhere."""
        return [
            name for name in self._compatible_outputs() if not self.schema.variables[name].constant
        ]

    def _is_continuous(self, name: str) -> bool:
        """True when an output's values are essentially all independent."""
        info = self.schema.variables.get(name)
        return bool(info and info.high_cardinality)

    def _axis_candidates(self) -> list[str]:
        dimensions = self._varying_dimensions()
        return dimensions + [name for name in self._varying_outputs() if name not in dimensions]

    def _on_variable(self, _event: Any) -> None:
        candidates = self._axis_candidates()
        self.x_widget.options = candidates
        if self.x_dim not in candidates:
            self.x_widget.value = candidates[-1]
        self.y_widget.options = [candidate for candidate in candidates if candidate != self.x_dim]
        if self.y_dim not in self.y_widget.options:
            self.y_widget.value = self.y_widget.options[-1]
        self._rebuild_axis_matrix()
        self._rebuild_filters()
        self._update()

    def _on_x(self, _event: Any) -> None:
        self.y_widget.options = [
            candidate for candidate in self._axis_candidates() if candidate != self.x_dim
        ]
        if self.y_dim not in self.y_widget.options:
            self.y_widget.value = self.y_widget.options[-1]
        self._sync_axis_matrix()
        self._update()

    def _on_y(self, _event: Any) -> None:
        self._sync_axis_matrix()
        self._update()

    def _rebuild_axis_matrix(self) -> None:
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
        for dim in self._axis_candidates():
            x_check = pn.widgets.Checkbox(name="", width=65, align="center")
            y_check = pn.widgets.Checkbox(name="", width=65, align="center")
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
            label = (
                self._dimension_label(dim)
                if dim in self.schema.coordinates
                else f"Output · {self._output_label(dim)}"
            )
            markup = (
                f"<code style='color:{CONTINUOUS_COLOR};font-weight:600'>{label}</code>"
                if self._is_continuous(dim)
                else f"`{label}`"
            )
            rows.append(
                pn.Row(
                    pn.pane.Markdown(markup, width=155, margin=(5, 5)),
                    x_check,
                    y_check,
                    sizing_mode="fixed",
                )
            )
        self._axis_matrix.objects = rows
        self._sync_axis_matrix()

    def _on_axis_check(self, dim: str, axis: str, selected: bool) -> None:
        if self._syncing_axes:
            return
        current = self.x_dim if axis == "x" else self.y_dim
        if selected:
            if axis == "x":
                self.x_widget.value = dim
            else:
                self.y_widget.value = dim
        elif current == dim:
            self._sync_axis_matrix()

    def _sync_axis_matrix(self) -> None:
        if not self._axis_x_checks:
            return
        self._syncing_axes = True
        try:
            for dim, checkbox in self._axis_x_checks.items():
                checkbox.value = dim == self.x_dim
                checkbox.disabled = dim == self.y_dim
            for dim, checkbox in self._axis_y_checks.items():
                checkbox.value = dim == self.y_dim
                checkbox.disabled = dim == self.x_dim
        finally:
            self._syncing_axes = False

    def _grid_variable(self) -> str:
        """Variable whose grid defines the sample set.

        The z variable may be an input coordinate, which spans no grid of its
        own, so fall back to the first output defined over it.
        """
        if self.variable in self.schema.variables:
            return self.variable
        spanning = next(
            (name for name, info in self.schema.variables.items() if self.variable in info.dims),
            None,
        )
        return spanning or next(iter(self.schema.variables))

    def _grid_dimensions(self) -> tuple[str, ...]:
        """Dimensions of the grid the current z variable is drawn from."""
        return self.schema.variables[self._grid_variable()].dims

    def _compatible_outputs(self) -> list[str]:
        dimensions = set(self._grid_dimensions())
        return [
            name
            for name, info in self.schema.variables.items()
            if set(info.dims).issubset(dimensions)
        ]

    def _sample_frame(self) -> pd.DataFrame:
        grid = self._grid_variable()
        data = self.dataset[grid]
        frame = data.to_dataframe(name=grid).reset_index()
        for name in self._compatible_outputs():
            if name == grid:
                continue
            broadcast = self.dataset[name].broadcast_like(data).transpose(*data.dims)
            frame[name] = np.asarray(broadcast.values).reshape(-1)
        return frame

    def _range_spec(self, values: np.ndarray) -> tuple[float, float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return 0.0, 1.0, 0.01
        low, high = float(finite.min()), float(finite.max())
        width = high - low
        step = width / 200 if width else max(abs(low) * 1e-6, 1e-9)
        return low, high, step

    def _rebuild_filters(self) -> None:
        self._saved_ranges.update(
            {
                name: (
                    list(widget.value)
                    if isinstance(widget, pn.widgets.MultiChoice)
                    else tuple(widget.value)
                )
                for name, widget in self._filter_widgets.items()
            }
        )
        self._saved_missing.update(
            {name: bool(widget.value) for name, widget in self._missing_widgets.items()}
        )
        frame = self._sample_frame()
        widgets: dict[str, pn.widgets.Widget] = {}
        missing_widgets: dict[str, pn.widgets.Checkbox] = {}
        cells: list[Any] = []
        inputs = self._varying_dimensions()
        outputs = self._varying_outputs()
        for name in inputs + outputs:
            label = (
                f"Input · {self._dimension_label(name)}"
                if name in inputs
                else f"Output · {self.schema.variables[name].long_name}"
            )
            styling = {"stylesheets": [CONTINUOUS_STYLESHEET]} if self._is_continuous(name) else {}
            if name in self.schema.coordinates and self.schema.coordinates[name].categorical:
                options = frame[name].drop_duplicates().tolist()
                saved_categories = self._saved_ranges.get(name, options)
                selected_categories = [value for value in saved_categories if value in options]
                widget = pn.widgets.MultiChoice(
                    name=label,
                    options=options,
                    value=selected_categories or options,
                    sizing_mode="stretch_width",
                    **styling,
                )
                widget.param.watch(lambda _event: self._update(), "value")
                widgets[name] = widget
                cells.append(widget)
                continue
            values = frame[name].to_numpy()
            missing_count = int((~np.isfinite(np.asarray(values, dtype=float))).sum())
            low, high, step = self._range_spec(values)
            saved = self._saved_ranges.get(name, (low, high))
            selected = (max(low, saved[0]), min(high, saved[1]))
            if selected[0] > selected[1]:
                selected = (low, high)
            slider_end = high if high > low else low + step
            slider_styles = [CONTINUOUS_STYLESHEET] if self._is_continuous(name) else []
            ticks = tick_stylesheet(values, low, slider_end)
            if ticks:
                slider_styles.append(ticks)
            widget = pn.widgets.RangeSlider(
                name=label,
                start=low,
                end=slider_end,
                value=selected,
                step=step,
                sizing_mode="stretch_width",
                disabled=missing_count == len(values),
                stylesheets=slider_styles,
            )
            widget.param.watch(lambda _event: self._update(), "value")
            widgets[name] = widget
            if not missing_count:
                cells.append(widget)
                continue
            noun = "sample" if missing_count == 1 else "samples"
            checkbox = pn.widgets.Checkbox(
                name=f"Include {missing_count:,} {noun} with no value",
                value=self._saved_missing.get(name, True),
                margin=(0, 10, 8, 10),
                **styling,
            )
            checkbox.param.watch(lambda _event: self._update(), "value")
            missing_widgets[name] = checkbox
            cells.append(pn.Column(widget, checkbox, sizing_mode="stretch_width"))
        self._filter_widgets = widgets
        self._missing_widgets = missing_widgets
        self._filter_grid.objects = cells

    def _update(self) -> None:
        try:
            frame = self._sample_frame()
            included = np.ones(len(frame), dtype=bool)
            for name, widget in self._filter_widgets.items():
                values = frame[name].to_numpy()
                if isinstance(widget, pn.widgets.MultiChoice):
                    included &= np.isin(values, widget.value)
                else:
                    lower, upper = widget.value
                    finite = np.isfinite(np.asarray(values, dtype=float))
                    matched = finite & (values >= lower) & (values <= upper)
                    keep_missing = self._missing_widgets.get(name)
                    if keep_missing is not None and keep_missing.value:
                        matched |= ~finite
                    included &= matched
            value_label = self._axis_label(self.variable)
            color_low, color_high, _ = self._range_spec(frame[self.variable].to_numpy())
            color_limits = (color_low, color_high)
            vdims = [column for column in frame.columns if column not in {self.x_dim, self.y_dim}]
            selected = hv.Points(
                frame.loc[included],
                kdims=[self.x_dim, self.y_dim],
                vdims=vdims,
                label="Inside filters",
            ).opts(
                color=self.variable,
                cmap=BLUE_PURPLE_RED,
                clim=color_limits,
                alpha=1.0,
                size=6,
                colorbar=True,
                colorbar_opts={"title": value_label},
                tools=["hover"],
            )
            plot = selected
            if self.nonmatching_widget.value == "Outline":
                outlined = hv.Points(
                    frame.loc[~included],
                    kdims=[self.x_dim, self.y_dim],
                    vdims=vdims,
                    label="Outside filters",
                ).opts(
                    color=self.variable,
                    cmap=BLUE_PURPLE_RED,
                    clim=color_limits,
                    fill_alpha=0.0,
                    line_alpha=1.0,
                    line_width=1.5,
                    size=6,
                    tools=["hover"],
                )
                plot = outlined * selected
            self._plot.object = plot.opts(
                responsive=True,
                height=540,
                xlabel=self._axis_label(self.x_dim),
                ylabel=self._axis_label(self.y_dim),
                title=f"{value_label} — all samples",
                show_legend=True,
                legend_position="right",
            )
            self._summary.object = (
                f"**{int(included.sum()):,} inside filters** · "
                f"{int((~included).sum()):,} outside filters · "
                f"{len(frame):,} total samples"
            )
            self._update_coverage(frame, included)
            self._message.visible = False
        except Exception as exc:
            self._message.object = str(exc)
            self._message.visible = True

    def _update_coverage(self, frame: pd.DataFrame, included: np.ndarray) -> None:
        """Report samples the plot cannot color and filters the user cannot apply."""
        messages: list[str] = []
        values = np.asarray(frame[self.variable].to_numpy(), dtype=float)
        undrawable = int((~np.isfinite(values) & included).sum())
        if undrawable:
            label = self._axis_label(self.variable)
            messages.append(
                f"{undrawable:,} of {int(included.sum()):,} plotted samples have no value "
                f"for `{label}` and are drawn without color."
            )
        excluded = [
            (name, count)
            for name, checkbox in self._missing_widgets.items()
            if not checkbox.value
            and (
                count := int((~np.isfinite(np.asarray(frame[name].to_numpy(), dtype=float))).sum())
            )
        ]
        for name, count in excluded:
            label = (
                self._dimension_label(name)
                if name in self.schema.coordinates
                else self.schema.variables[name].long_name
            )
            messages.append(
                f"{count:,} samples hidden by the `{label}` filter — no value to filter on."
            )
        self._coverage.object = "  \n".join(messages)
        self._coverage.visible = bool(messages)

    def _build_view(self) -> pn.viewable.Viewable:
        controls = pn.Column(
            pn.pane.Markdown("## Filter"),
            self.variable_widget,
            pn.Row(self.x_widget, self.y_widget),
            pn.pane.Markdown("### Non-matching points"),
            self.nonmatching_widget,
            pn.pane.Markdown("### Display axes"),
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
            self._coverage,
            self._summary,
            pn.pane.Markdown("### Input and output filters"),
            self._filter_grid,
            sizing_mode="stretch_both",
        )
        return pn.Row(controls, main, sizing_mode="stretch_both", min_height=720)
