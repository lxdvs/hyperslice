"""HoloViews plot construction."""

from __future__ import annotations

from typing import Any

import holoviews as hv
import numpy as np
import xarray as xr

from hyperslice.interpolation import linear_contour_surface
from hyperslice.schema import axis_label
from hyperslice.status import SliceStatus

hv.extension("bokeh")


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
    pareto: bool = False,
    pareto_value: float | None = None,
) -> Any:
    """Build the selected plot plus truthful sample/status overlays."""
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
        cmap="Viridis",
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
        base = hv.operation.contours(image, filled=False).opts(**common, colorbar=False)
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
    if pareto:
        if pareto_value is None:
            raise ValueError("A Pareto contour value is required.")
        surface = linear_contour_surface(
            data,
            x_dim=x_dim,
            y_dim=y_dim,
            invalid_mask=status.mask_invalid,
        )
        interpolated_mesh = hv.QuadMesh(
            surface,
            kdims=[x_dim, y_dim],
            vdims=vdims,
        )
        contour = hv.operation.contours(interpolated_mesh, levels=[pareto_value]).relabel(
            f"Pareto: {pareto_value:g}"
        )
        overlay *= contour.opts(color="#ff3b30", line_width=3, tools=["hover"])
    return overlay
