"""Lazy dataset loading."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from hyperslice.exceptions import DatasetLoadError

type DatasetSource = xr.Dataset | str | Path


def load_dataset(source: DatasetSource) -> xr.Dataset:
    """Return an xarray Dataset, opening supported paths lazily."""
    if isinstance(source, xr.Dataset):
        return source
    path = Path(source).expanduser()
    if not path.exists():
        raise DatasetLoadError(f"Dataset not found: {path}")
    try:
        if path.is_dir() or path.suffix.lower() == ".zarr":
            return xr.open_zarr(path)
        if path.suffix.lower() in {".nc", ".nc4", ".cdf", ".netcdf"}:
            return xr.open_dataset(path)
    except Exception as exc:
        raise DatasetLoadError(f"Could not open dataset '{path}': {exc}") from exc
    raise DatasetLoadError(
        f"Unsupported dataset format for '{path}'. Use NetCDF (.nc/.nc4) or Zarr (.zarr)."
    )
