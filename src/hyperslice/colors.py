"""Shared colour ramps for every rendering surface."""

from __future__ import annotations

from bokeh.palettes import Viridis256

#: Continuous palette used by the plots, the point cloud, and PNG export.
VIRIDIS: list[str] = list(Viridis256)

#: Outline for the design point selected in the filter view. Deliberately a hue
#: the Viridis path never reaches — it runs dark purple to blue to green to
#: yellow, with no magenta — and distinct from the red used for Pareto contours.
HIGHLIGHT_COLOR = "#ff00ff"


def banded(steps: int, palette: list[str] | None = None) -> list[str]:
    """Sample *palette* down to *steps* evenly spaced colours.

    Fewer bands make the z axis read as discrete levels rather than a gradient;
    the endpoints are always kept so the range still spans the full ramp.
    """
    colors = palette if palette is not None else VIRIDIS
    if steps < 2 or steps >= len(colors):
        return list(colors)
    last = len(colors) - 1
    return [colors[round(index * last / (steps - 1))] for index in range(steps)]
