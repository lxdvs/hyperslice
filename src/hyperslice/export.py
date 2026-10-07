"""Slice and plot export helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
import xarray as xr

from hyperslice._version import __version__


def slice_dataframe(
    data: xr.DataArray,
    *,
    x_dim: str,
    y_dim: str,
    status: xr.DataArray | None = None,
) -> pd.DataFrame:
    """Convert a two-dimensional slice into useful long-form tabular data."""
    name = data.name or "value"
    frame = data.to_dataframe(name=name).reset_index()
    if status is not None:
        status_frame = status.to_dataframe(name="status").reset_index()
        frame = frame.merge(status_frame[[x_dim, y_dim, "status"]], on=[x_dim, y_dim], how="left")
    return frame[[x_dim, y_dim, name] + (["status"] if status is not None else [])]


def csv_bytes(
    data: xr.DataArray, *, x_dim: str, y_dim: str, status: xr.DataArray | None = None
) -> bytes:
    """Serialize the current slice as UTF-8 CSV."""
    return (
        slice_dataframe(data, x_dim=x_dim, y_dim=y_dim, status=status).to_csv(index=False).encode()
    )


def netcdf_bytes(
    data: xr.DataArray,
    *,
    selections: dict[str, Any],
    method: str,
    source_attrs: dict[str, Any] | None = None,
    status: xr.DataArray | None = None,
) -> bytes:
    """Serialize a self-describing slice as NetCDF."""
    dataset = data.to_dataset(name=data.name or "value")
    if status is not None:
        dataset["status"] = status
    for dim, value in selections.items():
        dataset = dataset.assign_coords({dim: value})
    dataset.attrs.update(source_attrs or {})
    dataset.attrs.update(hyperslice_version=__version__, hyperslice_interpolation=method)
    raw = dataset.to_netcdf()
    return raw if isinstance(raw, bytes) else bytes(raw)


def save_png(plot: Any, path: str | Path) -> Path:
    """Export a plot to PNG, with a headless Matplotlib fallback."""
    import holoviews as hv

    target = Path(path)
    try:
        hv.save(plot, target, fmt="png", backend="bokeh")
    except RuntimeError as exc:
        if "webdriver" not in str(exc) and "chromium" not in str(exc):
            raise
        _save_png_headless(plot, target)
    return target


def png_bytes(plot: Any) -> BytesIO:
    """*plot* rendered to PNG, as a stream ready for a Panel download."""
    with NamedTemporaryFile(suffix=".png", delete=False) as temporary:
        path = Path(temporary.name)
    try:
        save_png(plot, path)
        return bytes_io(path.read_bytes())
    finally:
        path.unlink(missing_ok=True)


def _bokeh_opts(element: Any) -> dict[str, Any]:
    return dict(element.opts.get(backend="bokeh", defaults=False).kwargs)


def _save_png_headless(plot: Any, target: Path) -> None:
    """Redraw *plot* with Matplotlib when no browser driver is available.

    Covers the two plots HyperSlice makes: the slicer's heatmap (a QuadMesh)
    and the filter view's point cloud (Points, with any best-fit Curve).
    """
    import holoviews as hv

    if plot.traverse(lambda item: item, specs=[hv.QuadMesh]):
        _save_mesh_headless(plot, target)
    elif plot.traverse(lambda item: item, specs=[hv.Points]):
        _save_points_headless(plot, target)
    else:
        raise RuntimeError("PNG fallback supports heatmaps and point clouds only.")


def _save_points_headless(plot: Any, target: Path) -> None:
    """Scatter each Points layer as styled, plus any Curve, under the plot's title."""
    import holoviews as hv
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    overlay = _bokeh_opts(plot)
    figure, axes = plt.subplots(figsize=(10, 7), constrained_layout=True)
    colorbar: tuple[Any, str] | None = None
    for element in plot.traverse(lambda item: item, specs=[hv.Points, hv.Curve]):
        opts = _bokeh_opts(element)
        x = element.dimension_values(0)
        y = element.dimension_values(1)
        label = element.label or None
        if isinstance(element, hv.Curve):
            axes.plot(
                x,
                y,
                color=opts.get("color", "black"),
                linewidth=opts.get("line_width", 2),
                linestyle="--" if opts.get("line_dash") == "dashed" else "-",
                label=label,
            )
            continue
        # Bokeh sizes are diameters in screen pixels; Matplotlib wants area.
        area = (0.75 * float(opts.get("size", 6))) ** 2
        alpha = float(opts.get("alpha", 1.0))
        if opts.get("fill_alpha") == 0.0:
            axes.scatter(
                x,
                y,
                s=area,
                facecolors="none",
                edgecolors=opts.get("line_color", "black"),
                linewidths=opts.get("line_width", 2),
                label=label,
            )
            continue
        color = opts.get("color")
        if isinstance(color, str) and color in element:
            values = element.dimension_values(color).astype(float)
            cmap = ListedColormap(opts["cmap"]) if "cmap" in opts else "viridis"
            low, high = opts.get("clim", (None, None))
            artist = axes.scatter(
                x, y, c=values, cmap=cmap, vmin=low, vmax=high, s=area, alpha=alpha, label=label
            )
            # Samples with no value are drawn grey, as in the browser.
            artist.cmap.set_bad("#b0b0b0")
            if colorbar is None and alpha == 1.0:
                title = opts.get("colorbar_opts", {}).get("title")
                colorbar = (artist, title or element.get_dimension(color).pprint_label)
        else:
            axes.scatter(x, y, s=area, alpha=alpha, label=label)
    if colorbar is not None:
        figure.colorbar(colorbar[0], ax=axes, label=colorbar[1])
    for key, setter in (("xlim", axes.set_xlim), ("ylim", axes.set_ylim)):
        if key in overlay:
            setter(*overlay[key])
    first = plot.traverse(lambda item: item, specs=[hv.Points])[0]
    axes.set_xlabel(overlay.get("xlabel", first.kdims[0].pprint_label))
    axes.set_ylabel(overlay.get("ylabel", first.kdims[1].pprint_label))
    axes.set_title(str(overlay.get("title", "")), fontweight="bold")
    if axes.get_legend_handles_labels()[0]:
        axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    figure.savefig(target, dpi=150)
    plt.close(figure)


def _save_mesh_headless(plot: Any, target: Path) -> None:
    """Render the primary QuadMesh."""
    import holoviews as hv
    import matplotlib.pyplot as plt

    meshes = plot.traverse(lambda item: item, specs=[hv.QuadMesh])
    mesh = meshes[0]
    x_dim, y_dim = mesh.kdims
    value_dim = mesh.vdims[0]
    x, y, values = (
        mesh.dimension_values(x_dim, expanded=False),
        mesh.dimension_values(y_dim, expanded=False),
        mesh.dimension_values(value_dim, flat=False),
    )
    figure, axes = plt.subplots(figsize=(10, 7), constrained_layout=True)
    artist = axes.pcolormesh(x, y, values, shading="auto", cmap="viridis")
    axes.set_xlabel(x_dim.label)
    axes.set_ylabel(y_dim.label)
    axes.set_title(str(mesh.opts.get(backend="bokeh", defaults=False).kwargs.get("title", "")))
    figure.colorbar(artist, ax=axes, label=value_dim.label)
    figure.savefig(target, dpi=150)
    plt.close(figure)


def bytes_io(data: bytes) -> BytesIO:
    """Return a rewound binary stream suitable for Panel downloads."""
    stream = BytesIO(data)
    stream.seek(0)
    return stream
