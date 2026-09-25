"""Point-cloud filtering view for complete dataset exploration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr
from bokeh.models import ColorBar

from hyperslice.colors import HIGHLIGHT_COLOR, VIRIDIS, banded
from hyperslice.schema import DatasetSchema, axis_label
from hyperslice.widgets import reset_view_button

#: Opacity of samples outside the active filters. Low enough to read as
#: background against the matching cloud, high enough to keep the shape of the
#: sampled space visible — filtering narrows attention, it does not delete data.
NONMATCHING_ALPHA = 0.15

#: Label styling for inputs and high-cardinality outputs — every sample an
#: independent value rather than one of a few shared levels.
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

#: Native dropdowns ignore per-option CSS, so continuity is marked with a glyph
#: that carries its own colour. Both markers are the same width so names align.
CONTINUOUS_MARKER = "🟢 "
PLAIN_MARKER = "◦ "


def watch_settled(widget: pn.widgets.Widget, handler: Callable[[Any], None]) -> None:
    """Call *handler* once a widget settles rather than on every intermediate value.

    Sliders emit ``value`` continuously while dragged; redrawing on each tick
    replaces the plot mid-gesture and costs the browser its scroll position.
    Widgets without a throttled trait (dropdowns, checkboxes) settle instantly.
    """
    trait = "value_throttled" if "value_throttled" in widget.param else "value"
    widget.param.watch(handler, trait)


#: Quiet time after the last keystroke before the filter search is applied.
SEARCH_SETTLE_MS = 500

#: Margin around the points inside the filters, as a fraction of their span on
#: each side, matching HoloViews' own default framing.
FRAME_PADDING = 0.1


def axis_limits(values: Any) -> tuple[float, float] | None:
    """Padded axis range framing the finite numeric *values*, or None if there are none.

    A single distinct value gets a margin proportional to its magnitude, or of
    one unit at zero, so the axis never collapses to zero width.
    """
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return None
    low, high = float(finite.min()), float(finite.max())
    span = high - low
    pad = span * FRAME_PADDING if span > 0 else (abs(low) * FRAME_PADDING or 1.0)
    return (low - pad, high + pad)


def matches_search(query: str, *texts: str) -> bool:
    """Whether the (case-insensitive) *query* occurs in any of *texts*.

    A blank query matches everything.
    """
    needle = query.strip().casefold()
    return not needle or any(needle in text.casefold() for text in texts)


def text_fontsize(points: int) -> dict[str, str]:
    """HoloViews ``fontsize`` mapping putting every plot label at *points* pt.

    The title is a step larger so it still reads as the heading; the colorbar
    title and ticks follow the axes so the whole figure scales together.
    """
    size = f"{points}pt"
    return {
        "title": f"{points + 2}pt",
        "labels": size,
        "ticks": size,
        "legend": size,
        "legend_title": size,
        "clabel": size,
        "cticks": size,
    }


def bold_text(plot: Any, element: Any) -> None:
    """Bokeh hook setting every label bold and upright.

    Bokeh's defaults italicise axis and colorbar titles; a plot with sizeable,
    mixed-style text reads unevenly, so all text shares one weight and style.
    """
    figure = plot.state
    if figure.title is not None:
        figure.title.text_font_style = "bold"
    for axis in (*figure.xaxis, *figure.yaxis):
        axis.axis_label_text_font_style = "bold"
        axis.major_label_text_font_style = "bold"
    for legend in figure.legend:
        legend.label_text_font_style = "bold"
        legend.title_text_font_style = "bold"
    for colorbar in figure.select(type=ColorBar):
        colorbar.title_text_font_style = "bold"
        colorbar.major_label_text_font_style = "bold"


def option_map(names: list[str], continuous: Callable[[str], bool]) -> dict[str, str]:
    """Label each option, marking the continuous ones green."""
    return {
        f"{CONTINUOUS_MARKER if continuous(name) else PLAIN_MARKER}{name}": name for name in names
    }


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


def distinct_values(schema: DatasetSchema, name: str) -> int:
    """Number of distinct values *name* takes: sampled levels for an input."""
    if name in schema.coordinates:
        return int(schema.coordinates[name].size)
    return int(schema.variables[name].distinct_count)


def most_interesting(schema: DatasetSchema, names: list[str]) -> list[str]:
    """Order *names* from most to least interesting.

    Outputs come before inputs — they are what a sweep was run to learn — and
    within each class a field taking more distinct values ranks higher, since
    it has more structure to show. Ties keep the dataset's own order.
    """
    return sorted(
        names,
        key=lambda name: (name in schema.variables, distinct_values(schema, name)),
        reverse=True,
    )


class FilterView:
    """Project every sample onto two axes and outline samples outside active ranges."""

    def __init__(
        self,
        dataset: xr.Dataset,
        schema: DatasetSchema,
        *,
        on_point_selected: Callable[[dict[str, Any], str], None] | None = None,
    ) -> None:
        self.dataset = dataset
        self.schema = schema
        self.on_point_selected = on_point_selected
        outputs = [name for name, info in schema.variables.items() if not info.constant] or list(
            schema.variables
        )
        # Open on the most interesting fields: outputs before inputs, and
        # within each class the one taking the most distinct values.
        variable = most_interesting(schema, outputs)[0]
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
        x_default, y_default = most_interesting(
            schema, [name for name in axis_options if name != variable] or axis_options
        )[:2]
        self.variable_widget = pn.widgets.Select(
            name="Z variable", options=self._options(variables), value=variable
        )
        self.x_widget = pn.widgets.Select(
            name="X axis", options=self._options(axis_options), value=x_default
        )
        self.y_widget = pn.widgets.Select(
            name="Y axis",
            options=self._options([name for name in axis_options if name != x_default]),
            value=y_default,
        )
        self.point_size_widget = pn.widgets.IntSlider(
            name="Point size", start=3, end=24, step=1, value=6
        )
        self.continuous_color_widget = pn.widgets.Checkbox(name="Continuous colour", value=True)
        self.color_levels_widget = pn.widgets.IntSlider(
            name="Colour divisions", start=2, end=50, step=1, value=8, disabled=True
        )
        self.text_size_widget = pn.widgets.IntSlider(
            name="Text size (pt)", start=8, end=24, step=1, value=11
        )
        self.menu_toggle = pn.widgets.Toggle(name="☰", width=45, align="end")
        self._axis_matrix = pn.Column()
        self._axis_x_checks: dict[str, pn.widgets.Checkbox] = {}
        self._axis_y_checks: dict[str, pn.widgets.Checkbox] = {}
        self._syncing_axes = False
        self._filter_grid = pn.GridBox(ncols=2, sizing_mode="stretch_width")
        self.search_widget = pn.widgets.TextInput(
            placeholder="search...", width=220, align="center", margin=(0, 4, 0, 10)
        )
        self.clear_search_widget = pn.widgets.Button(
            name="Clear", width=70, align="center", margin=(0, 10, 0, 4)
        )
        self._pending_search: Any = None
        self._filter_cells: dict[str, tuple[Any, tuple[str, ...]]] = {}
        self._filter_widgets: dict[str, pn.widgets.Widget] = {}
        self._missing_widgets: dict[str, pn.widgets.Checkbox] = {}
        self._saved_ranges: dict[str, Any] = {}
        self._saved_missing: dict[str, bool] = {}
        self.nonmatching_widget = pn.widgets.RadioButtonGroup(
            name="Non-matching points",
            options=["Fade", "Hide"],
            value="Hide",
            button_type="default",
        )
        # Fixed height: a re-rendered plot that changes size would shift the
        # filter controls below it and cost the browser its scroll anchor.
        self._plot = pn.pane.HoloViews(height=560, sizing_mode="stretch_width")
        self._plot_box = pn.Column(self._plot, sizing_mode="stretch_width")
        self._reset_view = reset_view_button(self._plot_box)
        self._summary = pn.pane.Markdown()
        self._coverage = pn.pane.Alert("", alert_type="warning", visible=False)
        self._message = pn.pane.Alert("", alert_type="danger", visible=False)
        self._menu = pn.Column(
            self.point_size_widget,
            self.text_size_widget,
            self.continuous_color_widget,
            self.color_levels_widget,
            visible=False,
            styles={"padding": "8px", "background": "#eef2f6", "border-radius": "6px"},
        )
        self._selected_rows: pd.DataFrame | None = None
        self._selected_point: dict[str, Any] = {}
        self._resetting = False
        self.variable_widget.param.watch(self._on_variable, "value")
        self.x_widget.param.watch(self._on_x, "value")
        self.y_widget.param.watch(self._on_y, "value")
        self.nonmatching_widget.param.watch(lambda _event: self._update(), "value")
        watch_settled(self.point_size_widget, lambda _event: self._update())
        watch_settled(self.color_levels_widget, lambda _event: self._update())
        watch_settled(self.text_size_widget, lambda _event: self._update())
        self.continuous_color_widget.param.watch(self._on_continuous_color, "value")
        self.menu_toggle.param.watch(
            lambda event: setattr(self._menu, "visible", event.new), "value"
        )
        self.search_widget.param.watch(self._on_search_typed, "value_input")
        self.search_widget.param.watch(self._on_search_entered, "value")
        self.clear_search_widget.on_click(lambda _event: self.clear_search())
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
        """True when a field's values are essentially all independent.

        Every input qualifies: a swept coordinate's levels are distinct by
        construction, since the schema rejects duplicate coordinate values.
        """
        if name in self.schema.coordinates:
            return True
        info = self.schema.variables.get(name)
        return bool(info and info.high_cardinality)

    def _options(self, names: list[str]) -> dict[str, str]:
        """Dropdown options labelled so continuous fields read green."""
        return option_map(names, self._is_continuous)

    def _palette(self) -> list[str]:
        """Colour ramp for the z axis, banded when divisions are requested."""
        if self.continuous_color_widget.value:
            return VIRIDIS
        return banded(int(self.color_levels_widget.value))

    def _on_continuous_color(self, event: Any) -> None:
        self.color_levels_widget.disabled = bool(event.new)
        self._update()

    def _axis_candidates(self) -> list[str]:
        dimensions = self._varying_dimensions()
        return dimensions + [name for name in self._varying_outputs() if name not in dimensions]

    def _on_variable(self, _event: Any) -> None:
        candidates = self._axis_candidates()
        self.x_widget.options = self._options(candidates)
        if self.x_dim not in candidates:
            self.x_widget.value = candidates[-1]
        remaining = [candidate for candidate in candidates if candidate != self.x_dim]
        self.y_widget.options = self._options(remaining)
        if self.y_dim not in remaining:
            self.y_widget.value = remaining[-1]
        self._rebuild_axis_matrix()
        self._rebuild_filters()
        self._update()

    def _on_x(self, _event: Any) -> None:
        remaining = [candidate for candidate in self._axis_candidates() if candidate != self.x_dim]
        self.y_widget.options = self._options(remaining)
        if self.y_dim not in remaining:
            self.y_widget.value = remaining[-1]
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
        cells: dict[str, tuple[Any, tuple[str, ...]]] = {}
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
                cells[name] = (widget, (name, label))
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
            watch_settled(widget, lambda _event: self._update())
            widgets[name] = widget
            if not missing_count:
                cells[name] = (widget, (name, label))
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
            cell = pn.Column(widget, checkbox, sizing_mode="stretch_width")
            cells[name] = (cell, (name, label))
        self._filter_widgets = widgets
        self._missing_widgets = missing_widgets
        self._filter_cells = cells
        self._apply_search()

    def _on_search_typed(self, _event: Any) -> None:
        """Apply the search once typing has paused, not on every keystroke.

        Rebuilding the grid on each character would churn the layout while the
        query is still being written. Served documents wait out the settle
        time on the event loop; without a document there is nothing to wait on,
        so the search applies at once.
        """
        document = pn.state.curdoc
        if document is None:
            self._apply_search()
            return
        if self._pending_search is not None:
            document.remove_timeout_callback(self._pending_search)
        self._pending_search = document.add_timeout_callback(self._apply_search, SEARCH_SETTLE_MS)

    def _on_search_entered(self, event: Any) -> None:
        """Enter (or a programmatic ``value``) applies the search without waiting."""
        self.search_widget.value_input = event.new
        self._apply_search()

    def clear_search(self) -> None:
        """Empty the search box and bring every filter back on screen."""
        self.search_widget.value = ""
        # Resetting the typed text re-arms the settle wait; cancel it and apply now.
        self.search_widget.value_input = ""
        document = pn.state.curdoc
        if self._pending_search is not None and document is not None:
            document.remove_timeout_callback(self._pending_search)
        self._pending_search = None
        self._apply_search()

    def _apply_search(self) -> None:
        """Show only the filters whose name or label contains the search text.

        Hidden filters stay active: the search narrows what is on screen, not
        which samples are included.
        """
        self._pending_search = None
        query = self.search_widget.value_input or ""
        shown = [
            cell for cell, texts in self._filter_cells.values() if matches_search(query, *texts)
        ]
        if not shown and self._filter_cells:
            shown = [
                pn.pane.Markdown(
                    f"No filters match `{query.strip()}`.", sizing_mode="stretch_width"
                )
            ]
        self._filter_grid.objects = shown

    def _included_mask(self, frame: pd.DataFrame) -> np.ndarray:
        """Samples matching every active filter."""
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
        return included

    def _update(self) -> None:
        if self._resetting:
            return
        try:
            frame = self._sample_frame()
            included = self._included_mask(frame)
            value_label = self._axis_label(self.variable)
            color_low, color_high, _ = self._range_spec(frame[self.variable].to_numpy())
            color_limits = (color_low, color_high)
            palette = self._palette()
            size = int(self.point_size_widget.value)
            vdims = [column for column in frame.columns if column not in {self.x_dim, self.y_dim}]
            inside = frame.loc[included].reset_index(drop=True)
            self._selected_rows = inside
            selected = hv.Points(
                inside,
                kdims=[self.x_dim, self.y_dim],
                vdims=vdims,
                label="Inside filters",
            ).opts(
                color=self.variable,
                cmap=palette,
                clim=color_limits,
                alpha=1.0,
                size=size,
                colorbar=True,
                colorbar_opts={"title": value_label},
                tools=["hover", "tap"],
                nonselection_alpha=1.0,
            )
            self._tap_stream = hv.streams.Selection1D(source=selected)
            self._tap_stream.add_subscriber(self._on_point_tapped)
            plot = selected
            if self.nonmatching_widget.value == "Fade":
                faded = hv.Points(
                    frame.loc[~included],
                    kdims=[self.x_dim, self.y_dim],
                    vdims=vdims,
                    label="Outside filters",
                ).opts(
                    color=self.variable,
                    cmap=palette,
                    clim=color_limits,
                    alpha=NONMATCHING_ALPHA,
                    size=size,
                    tools=["hover"],
                )
                plot = faded * selected
            highlight = self._selection_overlay(frame, size)
            if highlight is not None:
                plot = plot * highlight
            # The default ranges, which "Reset plot bounds" restores, frame only
            # the points inside the filters; faded points may lie off screen.
            limits = {
                key: limit
                for key, limit in (
                    ("xlim", axis_limits(inside[self.x_dim])),
                    ("ylim", axis_limits(inside[self.y_dim])),
                )
                if limit is not None
            }
            self._plot.object = plot.opts(
                responsive=True,
                **limits,
                height=540,
                xlabel=self._axis_label(self.x_dim),
                ylabel=self._axis_label(self.y_dim),
                title=f"{value_label} — all samples",
                fontsize=text_fontsize(int(self.text_size_widget.value)),
                hooks=[bold_text],
                show_legend=True,
                legend_position="right",
                # Clicking a legend entry removes that layer outright. Bokeh's
                # default is to mute it, which fades it to the same treatment
                # non-matching points already carry — two states, one look.
                legend_opts={"click_policy": "hide"},
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

    def _match_mask(self, frame: pd.DataFrame) -> np.ndarray | None:
        """Rows of *frame* at the selected design point, or None if nothing is selected."""
        if not self._selected_point:
            return None
        mask = np.ones(len(frame), dtype=bool)
        matched = False
        for dim in self._grid_dimensions():
            if dim not in self._selected_point or dim not in frame.columns:
                continue
            values = frame[dim].to_numpy()
            target = self._selected_point[dim]
            if values.dtype.kind in "iufc":
                mask &= np.isclose(values.astype(float), float(target))
            else:
                mask &= values == target
            matched = True
        return mask if matched else None

    def _selection_overlay(self, frame: pd.DataFrame, size: int) -> hv.Points | None:
        """Ring around the selected sample wherever it lands in the projection."""
        mask = self._match_mask(frame)
        if mask is None or not mask.any():
            return None
        return hv.Points(
            frame.loc[mask, [self.x_dim, self.y_dim]],
            kdims=[self.x_dim, self.y_dim],
            label="Selected point",
        ).opts(
            fill_alpha=0.0,
            line_color=HIGHLIGHT_COLOR,
            line_width=3,
            size=size + 8,
        )

    def select_point(self, coordinates: dict[str, Any]) -> None:
        """Mark *coordinates* as the selected sample, clearing filters that hide it.

        A selection the filters exclude would be invisible, so rather than
        silently drop it the filters give way: the point the user asked for wins.
        """
        self._selected_point = dict(coordinates)
        frame = self._sample_frame()
        mask = self._match_mask(frame)
        if mask is not None and mask.any() and not (mask & self._included_mask(frame)).any():
            self.reset_filters()
            return
        self._update()

    def reset_filters(self) -> None:
        """Return every filter to its full range and include missing samples."""
        self._resetting = True
        try:
            for widget in self._filter_widgets.values():
                if isinstance(widget, pn.widgets.MultiChoice):
                    widget.value = list(widget.options)
                else:
                    widget.value = (widget.start, widget.end)
            for checkbox in self._missing_widgets.values():
                checkbox.value = True
        finally:
            self._resetting = False
        self._update()

    def _on_point_tapped(self, index: list[int]) -> None:
        """Select the tapped sample here and hand its design point to the slicer."""
        if not index or self._selected_rows is None:
            return
        row = self._selected_rows.iloc[index[0]]
        coordinates = {dim: row[dim] for dim in self._grid_dimensions() if dim in row}
        self.select_point(coordinates)
        if self.on_point_selected is not None:
            self.on_point_selected(coordinates, self.variable)

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
            pn.Row(
                pn.pane.Markdown("## Filter", sizing_mode="stretch_width"),
                self.menu_toggle,
                sizing_mode="stretch_width",
            ),
            self._menu,
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
            self._plot_box,
            self._reset_view,
            self._coverage,
            self._summary,
            pn.Row(
                pn.pane.Markdown(
                    "### Input and output filters", sizing_mode="fixed", align="center"
                ),
                self.search_widget,
                self.clear_search_widget,
                sizing_mode="stretch_width",
            ),
            self._filter_grid,
            sizing_mode="stretch_both",
        )
        return pn.Row(controls, main, sizing_mode="stretch_both", min_height=720)
