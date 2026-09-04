"""Local sensitivities of every output to every input at one design point."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from hyperslice.schema import DatasetSchema

#: Cell text for a derivative the grid cannot support — a categorical or
#: single-valued axis, a dimension the output does not span, or a neighbourhood
#: with missing values. Never rendered as a number, which would read as zero.
UNDEFINED = "—"


def input_dimensions(schema: DatasetSchema) -> list[str]:
    """Coordinates a derivative can be taken along: numeric and swept.

    A categorical axis has no spacing to divide by, and a constant one has no
    neighbour to difference against.
    """
    return [
        name
        for name, info in schema.coordinates.items()
        if not info.categorical and not info.constant
    ]


def _index_of(values: np.ndarray, target: Any) -> int | None:
    """Position of *target* in *values*: nearest for numbers, exact otherwise."""
    if values.dtype.kind in "iufc":
        try:
            return int(np.argmin(np.abs(values.astype(float) - float(target))))
        except (TypeError, ValueError):
            return None
    matches = np.flatnonzero(values == target)
    return int(matches[0]) if matches.size else None


def partial_derivative(data: xr.DataArray, dim: str, point: Mapping[str, Any]) -> float:
    """Slope of *data* along *dim* at *point*, from the sampled grid alone.

    Every other dimension is pinned to its value in *point*, leaving a line of
    samples that ``numpy.gradient`` differences — centrally in the interior and
    one-sidedly at the ends. Returns NaN whenever the grid cannot answer:
    ``dim`` is not one of the output's dimensions, *point* leaves a dimension
    unpinned, or a neighbouring sample is missing.
    """
    if dim not in data.dims:
        return float("nan")
    indexers: dict[str, int] = {}
    for other in (str(name) for name in data.dims):
        if other == dim:
            continue
        if other not in point:
            return float("nan")
        index = _index_of(np.asarray(data.coords[other].values), point[other])
        if index is None:
            return float("nan")
        indexers[other] = index
    positions = np.asarray(data.coords[dim].values, dtype=float)
    if positions.size < 2 or dim not in point:
        return float("nan")
    index = _index_of(positions, point[dim])
    if index is None:
        return float("nan")
    line = np.asarray(data.isel(indexers).values, dtype=float)
    # numpy.gradient assumes increasing positions; coordinates need not be sorted.
    order = np.argsort(positions)
    slope = np.gradient(line[order], positions[order])
    return float(slope[int(np.flatnonzero(order == index)[0])])


def sensitivity_frame(
    dataset: xr.Dataset, schema: DatasetSchema, point: Mapping[str, Any]
) -> pd.DataFrame:
    """Partial derivative of every output (rows) by every input (columns)."""
    inputs = input_dimensions(schema)
    outputs = list(schema.variables)
    values = [[partial_derivative(dataset[name], dim, point) for dim in inputs] for name in outputs]
    return pd.DataFrame(values, index=outputs, columns=inputs, dtype=float)


def format_derivative(value: float) -> str:
    """Render one cell, marking derivatives the grid cannot support."""
    return UNDEFINED if not np.isfinite(value) else f"{value:.4g}"


def _units(units: str | None) -> str:
    return units or "1"


def labelled_frame(frame: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Relabel *frame* with long names and the units of each derivative."""
    rows = {
        name: f"{schema.variables[name].long_name} [{_units(schema.variables[name].units)}]"
        for name in frame.index
    }
    columns = {
        name: f"{schema.coordinates[name].long_name} [{_units(schema.coordinates[name].units)}]"
        for name in frame.columns
    }
    return frame.rename(index=rows, columns=columns).map(format_derivative)
