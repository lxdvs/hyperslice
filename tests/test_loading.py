from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from hyperslice.exceptions import DatasetLoadError
from hyperslice.loading import load_dataset


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
