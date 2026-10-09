"""HoloViews plot construction."""

from __future__ import annotations

from typing import Any

import holoviews as hv
import numpy as np
import xarray as xr

from hyperslice.colors import VIRIDIS
from hyperslice.interpolation import linear_contour_surface
from hyperslice.schema import axis_label
from hyperslice.status import SliceStatus

hv.extension("bokeh")


def _inset_label_boxes(plot: Any, _element: Any) -> None:
    """Pad the background box around each label; not an option HoloViews exposes."""
    plot.handles["glyph"].padding = 2


#: Where along its line each successive height is written, as a fraction of
#: the vertices, so that neighbouring labels on parallel lines do not collide.
LABEL_POSITIONS = (0.5, 0.3, 0.7)


def contour_labels(isolines: Any, *, x_dim: str, y_dim: str) -> Any:
    """One text label per contour height, on the longest segment of its line.

    Bokeh draws no inline contour labels, so each height is written once on
    the segment that is longest in axis-normalised length, which keeps the
    label on the line and away from fragments clipped at the plot edge.
    Successive heights are staggered along their lines.
    """
    value_dim = isolines.vdims[0].name
    x_lo, x_hi = isolines.range(x_dim)
    y_lo, y_hi = isolines.range(y_dim)
    x_span = (x_hi - x_lo) or 1.0
    y_span = (y_hi - y_lo) or 1.0
    best: dict[float, tuple[float, np.ndarray, np.ndarray]] = {}
    for path in isolines.data:
        level = float(path[value_dim])
        xs = np.asarray(path[x_dim], dtype=float)
        ys = np.asarray(path[y_dim], dtype=float)
        # NaN rows separate the sub-paths of one height.
        breaks = np.flatnonzero(~np.isfinite(xs) | ~np.isfinite(ys))
        for start, stop in zip(
            np.concatenate(([0], breaks + 1)), np.concatenate((breaks, [len(xs)])), strict=True
        ):
            if stop - start < 2:
                continue
            seg_x, seg_y = xs[start:stop], ys[start:stop]
            length = float(np.sum(np.hypot(np.diff(seg_x) / x_span, np.diff(seg_y) / y_span)))
            if length > best.get(level, (-1.0, None, None))[0]:
                best[level] = (length, seg_x, seg_y)
    rows = []
    for index, (level, (_, seg_x, seg_y)) in enumerate(sorted(best.items())):
        at = round(LABEL_POSITIONS[index % len(LABEL_POSITIONS)] * (len(seg_x) - 1))
        rows.append((float(seg_x[at]), float(seg_y[at]), f"{level:g}"))
    return hv.Labels(rows, kdims=[x_dim, y_dim], vdims=["text"]).opts(
        text_color="white",
        text_font_size="9pt",
        text_align="center",
        text_baseline="middle",
        background_fill_color="black",
        background_fill_alpha=0.45,
        border_radius=3,
        hooks=[_inset_label_boxes],
    )


def build_plot(
    data: xr.DataArray,
    *,
    dataset: xr.Dataset,
    x_dim: str,
    y_dim: str,
    plot_type: str,
    title: str,
    status: SliceStatus,
    show_samples: bool,
    show_invalid: bool,
    contours: bool = False,
    contour_levels: int = 10,
) -> Any:
    """Build the selected plot plus truthful sample/status overlays.

    ``contours`` overlays isolines at automatically chosen heights of the
    response. They are traced on the linearly interpolated surface rather
    than on the coarse sample grid, and raise :class:`SliceError` when that
    surface cannot be built truthfully.
    """
    value_label = str(data.attrs.get("long_name", data.name or "Value"))
    units = data.attrs.get("units")
    vdims = [
        hv.Dimension(
            data.name or "value", label=f"{value_label} [{units}]" if units else value_label
        )
    ]
    # QuadMesh preserves irregular coordinate spacing; Image assumes uniform sampling.
    image = hv.QuadMesh(data, kdims=[x_dim, y_dim], vdims=vdims)
    common = dict(
        colorbar=True,
        cmap=VIRIDIS,
        responsive=True,
        height=570,
        title=title,
        xlabel=axis_label(dataset, x_dim),
        ylabel=axis_label(dataset, y_dim),
        tools=["hover"],
    )
    if plot_type in {"heatmap", "image"}:
        base = image.opts(**common)
    elif plot_type == "filled contour":
        base = hv.operation.contours(image, filled=True).opts(**common)
    else:
        line_options = common | {"colorbar": False}
        base = hv.operation.contours(image, filled=False).opts(**line_options)
    overlay = base
    xx, yy = np.meshgrid(data.coords[x_dim].values, data.coords[y_dim].values)
    if show_samples:
        valid = np.asarray(status.mask_valid.values)
        overlay *= hv.Points(
            (xx[valid], yy[valid]), kdims=[x_dim, y_dim], label="Valid samples"
        ).opts(size=4, color="white", line_color="black", alpha=0.75)
        missing = np.asarray(status.mask_missing.values)
        if missing.any():
            overlay *= hv.Points(
                (xx[missing], yy[missing]), kdims=[x_dim, y_dim], label="Missing samples"
            ).opts(marker="x", size=8, color="#ffbf00")
    if show_invalid:
        invalid = np.asarray(status.mask_invalid.values)
        if invalid.any():
            overlay *= hv.Points(
                (xx[invalid], yy[invalid]), kdims=[x_dim, y_dim], label="Invalid samples"
            ).opts(marker="x", size=9, color="#d62728", line_width=3)
    if contours and contour_levels < 1:
        raise ValueError("At least one contour level is required.")
    if contours:
        surface = linear_contour_surface(
            data,
            x_dim=x_dim,
            y_dim=y_dim,
            invalid_mask=status.mask_invalid,
        )
        interpolated_mesh = hv.QuadMesh(surface, kdims=[x_dim, y_dim], vdims=vdims)
        isolines = hv.operation.contours(interpolated_mesh, levels=contour_levels).relabel(
            "Interpolated contours"
        )
        overlay *= isolines.opts(
            color="white", line_width=1.5, alpha=0.85, tools=["hover"], show_legend=False
        )
        overlay *= contour_labels(isolines, x_dim=x_dim, y_dim=y_dim)
    return overlay
