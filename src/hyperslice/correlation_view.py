"""Clickable display of the filtered-sample sensitivity matrix."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import param
from panel.custom import JSComponent

from hyperslice.correlation import CORRELATION_STYLES


class SensitivityMatrix(JSComponent):
    """The rendered matrix, reporting which cell the user clicks.

    A click on a cell carrying ``data-output`` and ``data-input`` attributes
    calls *on_select* with that output and input. Plain HTML panes cannot send
    events back to Python, hence the small component.
    """

    content = param.String(default="", doc="Matrix HTML from render_matrix.")

    _esm = """
export function render({ model, el }) {
  const draw = () => { el.innerHTML = model.content }
  el.addEventListener("click", (event) => {
    const cell = event.target.closest("td[data-output]")
    if (cell) {
      model.send_msg({ output: cell.dataset.output, input: cell.dataset.input })
    }
  })
  model.on("content", draw)
  draw()
}
"""

    def __init__(self, on_select: Callable[[str, str], None] | None = None, **params: Any) -> None:
        params.setdefault("stylesheets", [CORRELATION_STYLES])
        super().__init__(**params)  # type: ignore[no-untyped-call]
        self._on_select = on_select

    def _handle_msg(self, data: Any) -> None:
        if self._on_select is None or not isinstance(data, dict):
            return
        output, dim = data.get("output"), data.get("input")
        if isinstance(output, str) and isinstance(dim, str):
            self._on_select(output, dim)
