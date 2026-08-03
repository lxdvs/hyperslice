"""Slice and plot export helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from hyperslice._version import __version__
from hyperslice.colors import BLUE_PURPLE_RED_ANCHORS


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


def _save_png_headless(plot: Any, target: Path) -> None:
    """Render the primary QuadMesh when no browser driver is available."""
    import holoviews as hv
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    meshes = plot.traverse(lambda item: item, specs=[hv.QuadMesh])
    if not meshes:
        raise RuntimeError("PNG fallback requires a two-dimensional QuadMesh plot.")
    mesh = meshes[0]
    x_dim, y_dim = mesh.kdims
    value_dim = mesh.vdims[0]
    x, y, values = (
        mesh.dimension_values(x_dim, expanded=False),
        mesh.dimension_values(y_dim, expanded=False),
        mesh.dimension_values(value_dim, flat=False),
    )
    figure, axes = plt.subplots(figsize=(10, 7), constrained_layout=True)
    colors = LinearSegmentedColormap.from_list("blue_purple_red", BLUE_PURPLE_RED_ANCHORS)
    artist = axes.pcolormesh(x, y, values, shading="auto", cmap=colors)
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
