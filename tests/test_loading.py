from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from hyperslice.exceptions import DatasetLoadError
from hyperslice.loading import load_dataset

POINTS: list[dict[str, Any]] = [
    {
        "inputs": {"heat_pipe.inner_diameter": 2.99, "fuel_hex.hex_pitch": 18.0},
        "outputs": {"run": {"status": 1}, "geometry": {"hex_pitch": 18.0}},
    },
    {
        "inputs": {"heat_pipe.inner_diameter": 2.99, "fuel_hex.hex_pitch": 19.0},
        "outputs": {"run": {"status": 1}, "geometry": {"hex_pitch": 19.0}},
    },
]


def _write_points(path: Path, records: Any) -> Path:
    path.write_text(json.dumps(records))
    return path


def test_dataset_passed_directly(dataset: xr.Dataset) -> None:
    assert load_dataset(dataset) is dataset


def test_netcdf_and_zarr_loading(dataset: xr.Dataset, tmp_path: Path) -> None:
    netcdf = tmp_path / "sweep.nc"
    zarr = tmp_path / "sweep.zarr"
    dataset.to_netcdf(netcdf)
    dataset.to_zarr(zarr)
    assert load_dataset(netcdf).sizes == dataset.sizes
    assert load_dataset(zarr).sizes == dataset.sizes


def test_invalid_and_unsupported_path(tmp_path: Path) -> None:
    with pytest.raises(DatasetLoadError, match="not found"):
        load_dataset(tmp_path / "missing.nc")
    text = tmp_path / "data.txt"
    text.write_text("no")
    with pytest.raises(DatasetLoadError, match="Unsupported"):
        load_dataset(text)


def test_points_json_namespaces_output_groups(tmp_path: Path) -> None:
    dataset = load_dataset(_write_points(tmp_path / "points.json", POINTS))
    assert dataset.sizes == {"fuel_hex.hex_pitch": 2}
    assert set(dataset.data_vars) == {"run.status", "geometry.hex_pitch"}
    assert dataset["geometry.hex_pitch"].values.tolist() == [18.0, 19.0]
    assert dataset.coords["heat_pipe.inner_diameter"].item() == 2.99


def test_points_json_keeps_flat_outputs_unprefixed(tmp_path: Path) -> None:
    records = [
        {"inputs": {"x": 1.0}, "outputs": {"mass": 3.0}},
        {"inputs": {"x": 2.0}, "outputs": {"mass": 4.0}},
    ]
    dataset = load_dataset(_write_points(tmp_path / "flat.json", records))
    assert set(dataset.data_vars) == {"mass"}


def test_points_json_treats_null_outputs_as_missing(tmp_path: Path) -> None:
    records = [
        {"inputs": {"x": 1.0}, "outputs": {"lifetime": {"crossing": 120.0}}},
        {"inputs": {"x": 2.0}, "outputs": {"lifetime": {"crossing": None}}},
    ]
    dataset = load_dataset(_write_points(tmp_path / "nulls.json", records))
    values = dataset["lifetime.crossing"].values
    assert values[0] == 120.0
    assert np.isnan(values[1])


def test_points_json_rejects_malformed_input(tmp_path: Path) -> None:
    with pytest.raises(DatasetLoadError, match="nonempty list"):
        load_dataset(_write_points(tmp_path / "empty.json", []))
    with pytest.raises(DatasetLoadError, match="'inputs' and 'outputs'"):
        load_dataset(_write_points(tmp_path / "bad.json", [{"inputs": {"x": 1.0}}]))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(DatasetLoadError, match="Could not read design-point JSON"):
        load_dataset(broken)
    mismatched = [
        {"inputs": {"x": 1.0}, "outputs": {"mass": 3.0}},
        {"inputs": {"y": 2.0}, "outputs": {"mass": 4.0}},
    ]
    with pytest.raises(DatasetLoadError, match="Design point 1"):
        load_dataset(_write_points(tmp_path / "mismatch.json", mismatched))
