"""Small Panel controls shared by the slicer and filter views."""

from __future__ import annotations

import panel as pn

#: Browser-side reset of every Bokeh plot inside ``container``.
#:
#: The HoloViews pane swaps its Bokeh model on every redraw, so the callback is
#: handed the pane's enclosing column — a model that survives redraws — and
#: walks down to whatever plots it holds at click time. Emitting a plot's
#: ``reset`` signal is exactly what the toolbar's reset tool does: axis ranges
#: return to their defaults without a round trip to the server.
RESET_VIEW_JS = """
const stack = [container];
const seen = new Set();
while (stack.length > 0) {
  const model = stack.pop();
  if (model == null || typeof model !== "object" || seen.has(model)) continue;
  seen.add(model);
  if (model.reset != null && typeof model.reset.emit === "function") {
    model.reset.emit();
    continue;
  }
  if (model.child != null) stack.push(model.child);
  for (const child of model.children ?? []) {
    stack.push(Array.isArray(child) ? child[0] : child);
  }
}
"""


def reset_view_button(container: pn.Column) -> pn.widgets.Button:
    """Button that restores the default axis ranges of the plots in *container*.

    *container* must be the column the plot pane lives in, not the pane itself:
    the pane's model is replaced whenever the plot is redrawn, while the column
    keeps its identity for the life of the page.
    """
    # Filled primary styling and an icon: a plain grey label under a plot reads
    # as a caption, not as something to click.
    button = pn.widgets.Button(
        name="Reset plot bounds",
        button_type="primary",
        icon="zoom-reset",
        width=190,
        margin=(0, 0, 8, 0),
        description="Undo zooming and panning: return the axes to their default ranges.",
    )
    button.js_on_click(args={"container": container}, code=RESET_VIEW_JS)
    return button
