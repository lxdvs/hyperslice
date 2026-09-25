"""Tornado-style display of the relative sensitivities at a design point.

Each output gets a strip of cells, one per input, showing the coefficient and
a bar about a centre line: red and upward for a positive sensitivity, blue and
downward for a negative one. Bar length is the coefficient's magnitude
relative to the largest in that output's strip, so every strip has one
full-length bar and the rest scale from it. A toggle beside each output label
orders the inputs by that output's magnitudes; the order applies to every
strip, so the cells stay column-aligned across outputs.
"""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np
import pandas as pd
import panel as pn

from hyperslice.schema import DatasetSchema
from hyperslice.sensitivity import UNDEFINED, format_sensitivity

#: Fill for a positive coefficient (output rises with the input).
POSITIVE_COLOR = "#c0392b"
#: Fill for a negative coefficient (output falls as the input rises).
NEGATIVE_COLOR = "#2d6f9f"

SENSITIVITY_STYLES = f"""
.hs-sens-strip {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: stretch;
}}
.hs-sens-cell {{
  display: flex;
  flex-direction: column;
  gap: 2px;
  width: 118px;
  flex: none;
  padding: 6px 8px;
  border: 1px solid #d5dde5;
  border-radius: 6px;
  background: #fbfcfd;
}}
.hs-sens-cell.undefined {{
  color: #8a98a6;
  background: #f3f5f7;
}}
.hs-sens-input {{
  font-size: 11px;
  font-weight: 650;
  color: #536878;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.hs-sens-reading {{
  display: flex;
  align-items: center;
  gap: 8px;
}}
.hs-sens-value {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  min-width: 56px;
}}
.hs-sens-bar {{
  position: relative;
  width: 14px;
  height: 44px;
  flex: none;
}}
.hs-sens-axis {{
  position: absolute;
  left: -3px;
  right: -3px;
  top: 50%;
  height: 1px;
  background: #536878;
}}
.hs-sens-fill {{
  position: absolute;
  left: 0;
  width: 100%;
  border-radius: 2px;
}}
.hs-sens-fill.up {{
  bottom: 50%;
  background: {POSITIVE_COLOR};
}}
.hs-sens-fill.down {{
  top: 50%;
  background: {NEGATIVE_COLOR};
}}
"""


def bar_fractions(row: pd.Series) -> pd.Series:
    """Magnitude of each coefficient relative to the largest finite one in *row*.

    Undefined coefficients, and every coefficient when none is nonzero, get 0.
    """
    magnitudes = row.astype(float).abs().where(np.isfinite(row.astype(float)), 0.0)
    largest = float(magnitudes.max()) if len(magnitudes) else 0.0
    if largest <= 0:
        return magnitudes * 0.0
    return magnitudes / largest


def ordered_inputs(row: pd.Series, by_magnitude: bool) -> list[str]:
    """Input names in grid order, or by descending magnitude with undefined last."""
    names = [str(name) for name in row.index]
    if not by_magnitude:
        return names
    values = row.astype(float)

    def key(name: str) -> tuple[int, float]:
        value = float(values[name])
        return (0, -abs(value)) if np.isfinite(value) else (1, 0.0)

    return sorted(names, key=key)


def render_strip(row: pd.Series, labels: dict[str, str], order: list[str] | None = None) -> str:
    """HTML for one output's cells, in *order* (grid order when None)."""
    fractions = bar_fractions(row)
    cells = []
    for name in order if order is not None else (str(name) for name in row.index):
        value = float(row[name])
        text = format_sensitivity(value)
        label = escape(labels.get(name, name))
        if text == UNDEFINED:
            bar = '<div class="hs-sens-bar"><div class="hs-sens-axis"></div></div>'
            cells.append(
                f'<div class="hs-sens-cell undefined" title="{label}: undefined">'
                f'<div class="hs-sens-input">{label}</div>'
                f'<div class="hs-sens-reading"><span class="hs-sens-value">{text}</span>{bar}</div>'
                "</div>"
            )
            continue
        direction = "up" if value > 0 else "down"
        height = float(fractions[name]) * 50.0
        bar = (
            '<div class="hs-sens-bar">'
            f'<div class="hs-sens-fill {direction}" style="height: {height:.1f}%"></div>'
            '<div class="hs-sens-axis"></div>'
            "</div>"
        )
        cells.append(
            f'<div class="hs-sens-cell" title="{label}: {text}">'
            f'<div class="hs-sens-input">{label}</div>'
            f'<div class="hs-sens-reading"><span class="hs-sens-value">{text}</span>{bar}</div>'
            "</div>"
        )
    return f'<div class="hs-sens-strip">{"".join(cells)}</div>'


class SensitivityPanel:
    """Panel column showing a strip of bars per output, with a sort toggle each.

    At most one toggle is on: it names the output whose magnitudes order the
    inputs, and that order is shared by every strip.
    """

    def __init__(self, schema: DatasetSchema) -> None:
        self.schema = schema
        self._frame: pd.DataFrame | None = None
        self._sort_by: str | None = None
        self._syncing = False
        self._title = pn.pane.Markdown(margin=(0, 0, 4, 0))
        self._blocks = pn.Column(sizing_mode="stretch_width")
        self._strips: dict[str, pn.pane.HTML] = {}
        self.sort_toggles: dict[str, pn.widgets.Toggle] = {}
        self.view = pn.Column(
            self._title,
            self._blocks,
            visible=False,
            sizing_mode="stretch_width",
        )

    @property
    def frame(self) -> pd.DataFrame | None:
        """Coefficients on display: outputs as rows, inputs as columns."""
        return self._frame

    @property
    def sort_by(self) -> str | None:
        """Output whose magnitudes order the inputs, or None for grid order."""
        return self._sort_by

    def input_order(self) -> list[str] | None:
        """Shared column order for every strip, or None for grid order."""
        if self._frame is None or self._sort_by not in self._frame.index:
            return None
        return ordered_inputs(self._frame.loc[self._sort_by], by_magnitude=True)

    def _input_labels(self) -> dict[str, str]:
        return {name: info.long_name for name, info in self.schema.coordinates.items()}

    def _output_label(self, name: str) -> str:
        return self.schema.variables[name].long_name

    def _toggle_for(self, output: str) -> pn.widgets.Toggle:
        toggle = self.sort_toggles.get(output)
        if toggle is None:
            toggle = pn.widgets.Toggle(
                name="Sort", icon="sort-descending", width=80, align="center", margin=(0, 0, 0, 8)
            )
            toggle.param.watch(lambda event, output=output: self._on_sort(output, event), "value")
            self.sort_toggles[output] = toggle
        return toggle

    def _on_sort(self, output: str, event: Any) -> None:
        if self._syncing:
            return
        if event.new:
            self._sort_by = output
        elif self._sort_by == output:
            self._sort_by = None
        self._sync_toggles()
        self._render_all()

    def _sync_toggles(self) -> None:
        """Leave only the ordering output's toggle on."""
        self._syncing = True
        try:
            for name, toggle in self.sort_toggles.items():
                toggle.value = name == self._sort_by
        finally:
            self._syncing = False

    def _render_all(self) -> None:
        if self._frame is None:
            return
        order = self.input_order()
        labels = self._input_labels()
        for output, strip in self._strips.items():
            strip.object = render_strip(self._frame.loc[output], labels, order)

    def update(self, frame: pd.DataFrame, title: str) -> None:
        """Show *frame* (outputs by inputs) under *title*, keeping sort choices."""
        self._frame = frame
        self._title.object = title
        if self._sort_by is not None and self._sort_by not in frame.index:
            self._sort_by = None
            self._sync_toggles()
        blocks: list[Any] = []
        self._strips = {}
        for output in (str(name) for name in frame.index):
            strip = pn.pane.HTML(
                stylesheets=[SENSITIVITY_STYLES], sizing_mode="stretch_width", margin=(0, 0, 6, 0)
            )
            self._strips[output] = strip
            header = pn.Row(
                pn.pane.Markdown(
                    f"**{self._output_label(output)}**", sizing_mode="fixed", align="center"
                ),
                self._toggle_for(output),
                sizing_mode="stretch_width",
            )
            blocks.append(pn.Column(header, strip, sizing_mode="stretch_width"))
        self._render_all()
        self._blocks.objects = blocks
        self.view.visible = True

    def clear(self) -> None:
        """Hide the panel; nothing is selected."""
        self._frame = None
        self.view.visible = False
