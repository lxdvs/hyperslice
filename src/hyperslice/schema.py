"""Dataset inspection and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from hyperslice.exceptions import DatasetSchemaError


@dataclass(frozen=True)
class CoordinateInfo:
    """Metadata for an independent one-dimensional coordinate."""

    name: str
    dtype: str
    size: int
    units: str | None
    long_name: str
    monotonic: bool
    unique: bool
    categorical: bool


@dataclass(frozen=True)
class VariableInfo:
    """Metadata for a plottable response variable."""

    name: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    units: str | None
    long_name: str
    fill_value: Any | None


@dataclass(frozen=True)
class DatasetSchema:
    """Inspected dataset structure used by the explorer."""

    coordinates: dict[str, CoordinateInfo]
    variables: dict[str, VariableInfo]
    status_candidates: tuple[str, ...]
    attrs: dict[str, Any]


def _monotonic(values: np.ndarray) -> bool:
    if len(values) < 2:
        return True
    try:
        delta = np.diff(values)
        return bool(np.all(delta > 0) or np.all(delta < 0))
    except (TypeError, ValueError):
        return False


def inspect_dataset(dataset: xr.Dataset) -> DatasetSchema:
    """Inspect and validate the rectilinear parts of *dataset*."""
    if not dataset.dims:
        raise DatasetSchemaError("Dataset has no usable dimensions.")
    coordinates: dict[str, CoordinateInfo] = {}
    for dim, size in dataset.sizes.items():
        if dim not in dataset.coords:
            raise DatasetSchemaError(f"Dimension '{dim}' has no coordinate array.")
        coord = dataset.coords[dim]
        if coord.dims != (dim,):
            raise DatasetSchemaError(f"Coordinate '{dim}' must be one-dimensional and independent.")
        values = np.asarray(coord.values)
        unique = len(np.unique(values)) == size
        if not unique:
            raise DatasetSchemaError(f"Coordinate '{dim}' contains duplicate values.")
        categorical = coord.dtype.kind in "OUSb"
        coordinates[dim] = CoordinateInfo(
            name=dim,
            dtype=str(coord.dtype),
            size=size,
            units=coord.attrs.get("units"),
            long_name=str(coord.attrs.get("long_name", dim.replace("_", " ").title())),
            monotonic=_monotonic(values) if not categorical else False,
            unique=unique,
            categorical=categorical,
        )
    variables: dict[str, VariableInfo] = {}
    statuses: list[str] = []
    for name, data in dataset.data_vars.items():
        attrs = data.attrs
        is_flag = (
            "flag_values" in attrs
            or "flag_meanings" in attrs
            or data.dtype.kind == "b"
            or name.lower() in {"status", "valid", "validity", "mask"}
        )
        if is_flag:
            statuses.append(name)
        if data.dtype.kind not in "iufc" or data.ndim < 2 or is_flag:
            continue
        variables[name] = VariableInfo(
            name=name,
            dims=tuple(data.dims),
            shape=tuple(data.shape),
            dtype=str(data.dtype),
            units=attrs.get("units"),
            long_name=str(attrs.get("long_name", name.replace("_", " ").title())),
            fill_value=attrs.get("_FillValue", attrs.get("missing_value")),
        )
    if not variables:
        raise DatasetSchemaError(
            "Dataset has no plottable numeric variables with at least two dimensions."
        )
    return DatasetSchema(coordinates, variables, tuple(statuses), dict(dataset.attrs))


def axis_label(dataset: xr.Dataset, name: str) -> str:
    """Build a human-readable coordinate label with units."""
    coord = dataset.coords[name]
    label = str(coord.attrs.get("long_name", name.replace("_", " ").title()))
    units = coord.attrs.get("units")
    return f"{label} [{units}]" if units else label
