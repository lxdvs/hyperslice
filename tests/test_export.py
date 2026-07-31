from __future__ import annotations

import io

import pandas as pd
import xarray as xr

from hyperslice.export import csv_bytes, netcdf_bytes


def test_csv_headers(dataset: xr.Dataset) -> None:
    data = dataset.k_eff.sel(pressure=2.5, flow_rate=5, burnup=5)
    frame = pd.read_csv(io.BytesIO(csv_bytes(data, x_dim="drum_angle", y_dim="fuel_temperature")))
    assert list(frame) == ["drum_angle", "fuel_temperature", "k_eff"]


def test_netcdf_metadata(dataset: xr.Dataset) -> None:
    data = dataset.k_eff.sel(pressure=2.5, flow_rate=5, burnup=5)
    raw = netcdf_bytes(data, selections={"pressure": 2.5}, method="exact")
    restored = xr.open_dataset(io.BytesIO(raw))
    assert restored.attrs["hyperslice_interpolation"] == "exact"
    assert restored.pressure.item() == 2.5
