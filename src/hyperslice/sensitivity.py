"""Local relative sensitivities of every output to every input at one design point.

A relative (normalised) sensitivity is the elasticity ``(x / y) * dy/dx``: the
fractional change in an output per fractional change in an input. Being
dimensionless, the cells of one table compare directly across inputs and
outputs with different units and magnitudes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from hyperslice.schema import DatasetSchema

#: Cell text for a sensitivity the grid cannot support — a categorical or
#: single-valued axis, a dimension the output does not span, a neighbourhood
#: with missing values, or an output of zero that no fraction can be taken of.
#: Never rendered as a number, which would read as zero.
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


def value_at(data: xr.DataArray, point: Mapping[str, Any]) -> float:
    """Sample of *data* at *point*, or NaN if any of its dimensions is unpinned."""
    indexers: dict[str, int] = {}
    for dim in (str(name) for name in data.dims):
        if dim not in point:
            return float("nan")
        index = _index_of(np.asarray(data.coords[dim].values), point[dim])
        if index is None:
            return float("nan")
        indexers[dim] = index
    return float(data.isel(indexers).values)


def relative_sensitivity(data: xr.DataArray, dim: str, point: Mapping[str, Any]) -> float:
    """Elasticity of *data* to *dim* at *point*: ``(x / y) * dy/dx``.

    The partial derivative is scaled by the input and output values at the
    point, so the result is the fractional change in the output per fractional
    change in the input, independent of either quantity's units. An input at
    exactly zero therefore has zero relative sensitivity, whatever its slope.
    Returns NaN whenever the derivative is undefined or the output is zero,
    where no fraction of it exists to compare against.
    """
    slope = partial_derivative(data, dim, point)
    if not np.isfinite(slope):
        return float("nan")
    output = value_at(data, point)
    if not np.isfinite(output) or output == 0:
        return float("nan")
    try:
        position = float(point[dim])
    except (TypeError, ValueError):
        return float("nan")
    return slope * position / output


def sensitivity_frame(
    dataset: xr.Dataset, schema: DatasetSchema, point: Mapping[str, Any]
) -> pd.DataFrame:
    """Relative sensitivity of every output (rows) to every input (columns)."""
    inputs = input_dimensions(schema)
    outputs = list(schema.variables)
    values = [
        [relative_sensitivity(dataset[name], dim, point) for dim in inputs] for name in outputs
    ]
    return pd.DataFrame(values, index=outputs, columns=inputs, dtype=float)


def format_sensitivity(value: float) -> str:
    """Render one cell, marking sensitivities the grid cannot support."""
    return UNDEFINED if not np.isfinite(value) else f"{value:.4g}"


def labelled_frame(frame: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Relabel *frame* with long names; relative sensitivities carry no units."""
    rows = {name: schema.variables[name].long_name for name in frame.index}
    columns = {name: schema.coordinates[name].long_name for name in frame.columns}
    return frame.rename(index=rows, columns=columns).map(format_sensitivity)
