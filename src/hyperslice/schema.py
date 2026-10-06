"""Dataset inspection and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from hyperslice.exceptions import DatasetSchemaError

#: Dataset attribute naming how the samples are arranged.
LAYOUT_ATTR = "hyperslice_layout"
#: :data:`LAYOUT_ATTR` value for scattered samples: one row per design point
#: along :data:`SAMPLE_DIM`, with inputs as coordinates over that dimension
#: rather than as grid axes.
POINTS_LAYOUT = "points"
SAMPLE_DIM = "sample"


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
    constant: bool
    constant_value: Any | None


@dataclass(frozen=True)
class VariableInfo:
    """Metadata for a plottable response variable.

    ``cardinality`` is the number of distinct values the output takes across
    the dataset, ignoring missing ones; ``distinct_ratio`` is that count as a
    share of the values present.
    """

    name: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    units: str | None
    long_name: str
    fill_value: Any | None
    constant: bool
    constant_value: Any | None
    cardinality: int
    distinct_ratio: float
    high_cardinality: bool


@dataclass(frozen=True)
class DatasetSchema:
    """Inspected dataset structure used by the explorer.

    For scattered samples every output is listed as spanning every input, as a
    grid output would, but its ``shape`` is the sample count alone: there is
    no grid to measure, and the slicer, which needs one, is not offered.
    """

    coordinates: dict[str, CoordinateInfo]
    variables: dict[str, VariableInfo]
    status_candidates: tuple[str, ...]
    attrs: dict[str, Any]
    scattered: bool = False


def _monotonic(values: np.ndarray) -> bool:
    if len(values) < 2:
        return True
    try:
        delta = np.diff(values)
        return bool(np.all(delta > 0) or np.all(delta < 0))
    except (TypeError, ValueError):
        return False


#: Distinct-to-present ratio at or above which a variable counts as high-cardinality:
#: its values are essentially all independent rather than falling into shared levels.
HIGH_CARDINALITY_RATIO = 0.9


def _cardinality(values: np.ndarray) -> tuple[int, float, bool]:
    """Report distinct value count, distinct ratio, and the high-cardinality flag."""
    flat = np.asarray(values).reshape(-1)
    present = flat[np.isfinite(flat)] if flat.dtype.kind in "fc" else flat
    if present.size == 0:
        return 0, 0.0, False
    distinct = int(np.unique(present).size)
    ratio = distinct / present.size
    return distinct, ratio, distinct > 1 and ratio >= HIGH_CARDINALITY_RATIO


def _constant_summary(values: np.ndarray) -> tuple[bool, Any]:
    """Report whether *values* never vary, and the single value if so.

    A field with gaps is not constant even when its present values agree: the
    presence pattern itself is something the user can filter on. A field that is
    entirely missing is not constant either, since it has no value to report.
    """
    flat = np.asarray(values).reshape(-1)
    present = flat[np.isfinite(flat)] if flat.dtype.kind in "fc" else flat
    if present.size == 0 or present.size != flat.size:
        return False, None
    distinct = np.unique(present)
    if distinct.size != 1:
        return False, None
    single = distinct[0]
    return True, single.item() if hasattr(single, "item") else single


def _is_flag(name: str, data: xr.DataArray) -> bool:
    attrs = data.attrs
    return (
        "flag_values" in attrs
        or "flag_meanings" in attrs
        or data.dtype.kind == "b"
        or name.lower() in {"status", "valid", "validity", "mask"}
    )


def _variable_info(
    name: str, data: xr.DataArray, dims: tuple[str, ...], shape: tuple[int, ...]
) -> VariableInfo:
    attrs = data.attrs
    values = np.asarray(data.values)
    constant, constant_value = _constant_summary(values)
    cardinality, distinct_ratio, high_cardinality = _cardinality(values)
    return VariableInfo(
        name=name,
        dims=dims,
        shape=shape,
        dtype=str(data.dtype),
        units=attrs.get("units"),
        long_name=str(attrs.get("long_name", name.replace("_", " ").title())),
        fill_value=attrs.get("_FillValue", attrs.get("missing_value")),
        constant=constant,
        constant_value=constant_value,
        cardinality=cardinality,
        distinct_ratio=distinct_ratio,
        high_cardinality=high_cardinality,
    )


def inspect_points(dataset: xr.Dataset) -> DatasetSchema:
    """Inspect scattered samples laid out along :data:`SAMPLE_DIM`.

    Every coordinate over the samples other than the sample index is an input;
    its ``size`` is the number of distinct values it takes. Every numeric
    variable over the samples that is not a status flag is an output.
    """
    if dataset.sizes.get(SAMPLE_DIM, 0) == 0:
        raise DatasetSchemaError(f"Scattered dataset has no '{SAMPLE_DIM}' samples.")
    samples = int(dataset.sizes[SAMPLE_DIM])
    coordinates: dict[str, CoordinateInfo] = {}
    for name, coord in dataset.coords.items():
        name = str(name)
        if name == SAMPLE_DIM or coord.dims != (SAMPLE_DIM,):
            continue
        values = np.asarray(coord.values)
        categorical = coord.dtype.kind in "OUSb"
        present = values if categorical else values[np.isfinite(values.astype(float))]
        distinct = np.unique(present)
        coordinates[name] = CoordinateInfo(
            name=name,
            dtype=str(coord.dtype),
            size=int(distinct.size),
            units=coord.attrs.get("units"),
            long_name=str(coord.attrs.get("long_name", name.replace("_", " ").title())),
            monotonic=False,
            unique=bool(distinct.size == samples),
            categorical=categorical,
            constant=distinct.size == 1,
            constant_value=distinct[0].item() if distinct.size == 1 else None,
        )
    if not coordinates:
        raise DatasetSchemaError("Scattered dataset has no input coordinates.")
    inputs = tuple(coordinates)
    variables: dict[str, VariableInfo] = {}
    statuses: list[str] = []
    for name, data in dataset.data_vars.items():
        name = str(name)
        if data.dims != (SAMPLE_DIM,):
            continue
        if _is_flag(name, data):
            statuses.append(name)
        if data.dtype.kind not in "iufc" or _is_flag(name, data):
            continue
        variables[name] = _variable_info(name, data, inputs, (samples,))
    if not variables:
        raise DatasetSchemaError("Scattered dataset has no numeric outputs.")
    return DatasetSchema(
        coordinates, variables, tuple(statuses), dict(dataset.attrs), scattered=True
    )


def inspect_dataset(dataset: xr.Dataset) -> DatasetSchema:
    """Inspect and validate the rectilinear parts of *dataset*.

    A dataset marked with the points layout is inspected as scattered samples
    instead, see :func:`inspect_points`.
    """
    if dataset.attrs.get(LAYOUT_ATTR) == POINTS_LAYOUT:
        return inspect_points(dataset)
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
            constant=size == 1,
            constant_value=values[0].item() if size == 1 and hasattr(values[0], "item") else None,
        )
    variables: dict[str, VariableInfo] = {}
    statuses: list[str] = []
    for name, data in dataset.data_vars.items():
        is_flag = _is_flag(str(name), data)
        if is_flag:
            statuses.append(str(name))
        if data.dtype.kind not in "iufc" or data.ndim < 2 or is_flag:
            continue
        variables[str(name)] = _variable_info(
            str(name), data, tuple(str(dim) for dim in data.dims), tuple(data.shape)
        )
    if not variables:
        raise DatasetSchemaError(
            "Dataset has no plottable numeric variables with at least two dimensions."
        )
    return DatasetSchema(coordinates, variables, tuple(statuses), dict(dataset.attrs))


def cardinality_table(schema: DatasetSchema) -> str:
    """Plain-text table of every output's cardinality, highest first.

    Ties keep the dataset's order, matching how the views rank outputs.
    """
    outputs = sorted(schema.variables.values(), key=lambda info: -info.cardinality)
    header = (
        "Output",
        "Cardinality",
        "Distinct share",
        "Samples" if schema.scattered else "Dimensions",
    )
    rows = [
        (
            info.name,
            f"{info.cardinality:,}",
            f"{info.distinct_ratio:.1%}",
            f"{info.shape[0]:,} samples"
            if schema.scattered
            else " x ".join(
                f"{dim}[{size}]" for dim, size in zip(info.dims, info.shape, strict=True)
            ),
        )
        for info in outputs
    ]
    widths = [max(len(row[column]) for row in [header, *rows]) for column in range(len(header))]

    def line(cells: tuple[str, ...]) -> str:
        # Name left, numbers right, dimensions left and unpadded at the end.
        return "  ".join(
            (
                cell.ljust(width)
                if column == 0
                else cell
                if column == len(cells) - 1
                else cell.rjust(width)
            )
            for column, (cell, width) in enumerate(zip(cells, widths, strict=True))
        )

    rule = "  ".join("-" * width for width in widths)
    return "\n".join([line(header), rule, *(line(row) for row in rows)])


def axis_label(dataset: xr.Dataset, name: str) -> str:
    """Build a human-readable coordinate label with units."""
    coord = dataset.coords[name]
    label = str(coord.attrs.get("long_name", name.replace("_", " ").title()))
    units = coord.attrs.get("units")
    return f"{label} [{units}]" if units else label
