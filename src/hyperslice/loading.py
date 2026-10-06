"""Lazy dataset loading."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from hyperslice.builder import DatasetBuilder, DatasetBuilderError
from hyperslice.exceptions import DatasetLoadError
from hyperslice.schema import LAYOUT_ATTR, POINTS_LAYOUT, SAMPLE_DIM

type DatasetSource = xr.Dataset | str | Path
#: How design-point JSON is arranged: ``"grid"`` builds the Cartesian product
#: of the input values, ``"points"`` keeps one row per sample, and ``"auto"``
#: picks points whenever the grid would be mostly empty.
type PointsLayout = Literal["auto", "grid", "points"]

#: Grid cells per supplied point beyond which ``"auto"`` loads samples rather
#: than a grid. A factorial sweep with holes fills most of its grid; a
#: space-filling design (Sobol, Latin hypercube) gives nearly every point its
#: own value of each input, so its grid would be almost all gaps and can run
#: to billions of cells.
SPARSE_CELLS_PER_POINT = 100


def load_dataset(source: DatasetSource, layout: PointsLayout = "auto") -> xr.Dataset:
    """Return an xarray Dataset, opening supported paths lazily.

    *layout* applies to design-point JSON only; see :func:`load_points_json`.
    """
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
            return load_points_json(path, layout=layout)
    except DatasetLoadError:
        raise
    except Exception as exc:
        raise DatasetLoadError(f"Could not open dataset '{path}': {exc}") from exc
    raise DatasetLoadError(
        f"Unsupported dataset format for '{path}'. "
        "Use NetCDF (.nc/.nc4), Zarr (.zarr), or design-point JSON (.json)."
    )


def load_points_json(path: str | Path, layout: PointsLayout = "auto") -> xr.Dataset:
    """Build a Dataset from a JSON list of ``{"inputs": ..., "outputs": ...}`` points.

    Outputs may be grouped into nested objects; each group name becomes a
    dot-separated prefix on the variable name, so ``{"geometry": {"hex_pitch": 18}}``
    yields the variable ``geometry.hex_pitch``. Flat outputs are kept verbatim.

    With the ``"grid"`` layout the points fill a rectilinear grid over every
    distinct input value. With ``"points"`` they stay a table of samples, see
    :func:`points_dataset`. ``"auto"`` chooses points when the grid would hold
    more than :data:`SPARSE_CELLS_PER_POINT` cells per point.
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

    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or "inputs" not in record or "outputs" not in record:
            raise DatasetLoadError(
                f"Design point {index} in '{source}' must be an object with "
                "'inputs' and 'outputs' keys."
            )
    attrs = {"title": source.stem, "source_file": source.name}
    if layout == "points" or (layout == "auto" and _grid_is_sparse(records)):
        return points_dataset(records, attrs, source)
    builder = DatasetBuilder(attrs=attrs)
    for index, record in enumerate(records):
        try:
            builder.add_point(record["inputs"], _flatten(record["outputs"]))
        except DatasetBuilderError as exc:
            raise DatasetLoadError(f"Design point {index} in '{source}' is invalid: {exc}") from exc
    return builder.to_dataset()


def _grid_is_sparse(records: list[Any]) -> bool:
    """Whether the points would fill only a sliver of their rectilinear grid."""
    names = {name for record in records for name in record["inputs"]}
    cells = math.prod(
        len({_hashable(record["inputs"].get(name)) for record in records}) for name in names
    )
    return cells > SPARSE_CELLS_PER_POINT * len(records)


def _hashable(value: Any) -> Any:
    return json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value


def points_dataset(records: list[Any], attrs: dict[str, Any], source: Path) -> xr.Dataset:
    """One row per design point along :data:`SAMPLE_DIM`, with no grid at all.

    Inputs become coordinates and outputs variables, each a one-dimensional
    array over the samples, so memory grows with the number of points rather
    than with the product of every input's distinct values. Outputs a point
    does not report are NaN.
    """
    names = list(records[0]["inputs"])
    for index, record in enumerate(records):
        if set(record["inputs"]) != set(names):
            raise DatasetLoadError(
                f"Design point {index} in '{source}' has inputs {sorted(record['inputs'])}; "
                f"expected {sorted(names)}."
            )
    inputs = pd.DataFrame([record["inputs"] for record in records], columns=names)
    outputs = pd.DataFrame([_flatten(record["outputs"]) for record in records])
    clashes = sorted(set(names) & set(outputs.columns))
    if clashes:
        raise DatasetLoadError(f"Names used as both input and output in '{source}': {clashes}.")
    samples = np.arange(len(records))
    return xr.Dataset(
        {name: (SAMPLE_DIM, outputs[name].to_numpy()) for name in outputs.columns},
        coords={
            SAMPLE_DIM: samples,
            **{name: (SAMPLE_DIM, inputs[name].to_numpy()) for name in names},
        },
        attrs={**attrs, LAYOUT_ATTR: POINTS_LAYOUT},
    )


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
