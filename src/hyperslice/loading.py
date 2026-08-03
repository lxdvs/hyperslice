"""Lazy dataset loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from hyperslice.builder import DatasetBuilder, DatasetBuilderError
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
        if path.suffix.lower() == ".json":
            return load_points_json(path)
    except DatasetLoadError:
        raise
    except Exception as exc:
        raise DatasetLoadError(f"Could not open dataset '{path}': {exc}") from exc
    raise DatasetLoadError(
        f"Unsupported dataset format for '{path}'. "
        "Use NetCDF (.nc/.nc4), Zarr (.zarr), or design-point JSON (.json)."
    )


def load_points_json(path: str | Path) -> xr.Dataset:
    """Build a Dataset from a JSON list of ``{"inputs": ..., "outputs": ...}`` points.

    Outputs may be grouped into nested objects; each group name becomes a
    dot-separated prefix on the variable name, so ``{"geometry": {"hex_pitch": 18}}``
    yields the variable ``geometry.hex_pitch``. Flat outputs are kept verbatim.
    """
    source = Path(path).expanduser()
    try:
        records = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetLoadError(f"Could not read design-point JSON '{source}': {exc}") from exc
    if not isinstance(records, list) or not records:
        raise DatasetLoadError(
            f"Design-point JSON '{source}' must be a nonempty list of design points."
        )

    builder = DatasetBuilder(attrs={"title": source.stem, "source_file": source.name})
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or "inputs" not in record or "outputs" not in record:
            raise DatasetLoadError(
                f"Design point {index} in '{source}' must be an object with "
                "'inputs' and 'outputs' keys."
            )
        try:
            builder.add_point(record["inputs"], _flatten(record["outputs"]))
        except DatasetBuilderError as exc:
            raise DatasetLoadError(f"Design point {index} in '{source}' is invalid: {exc}") from exc
    return builder.to_dataset()


def _flatten(outputs: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested output groups into dot-separated variable names.

    A ``null`` output becomes ``NaN``, matching how the builder represents an
    unsupplied grid combination, so points that lack a variable stay in the
    dataset instead of rejecting the whole file.
    """
    if not isinstance(outputs, Mapping):
        raise DatasetLoadError(f"Outputs must be an object; received {type(outputs).__name__}.")
    flattened: dict[str, Any] = {}
    for name, value in outputs.items():
        qualified = f"{prefix}{name}"
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, prefix=f"{qualified}."))
        else:
            flattened[qualified] = np.nan if value is None else value
    return flattened
