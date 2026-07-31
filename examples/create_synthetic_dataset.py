"""Create deterministic irregular five-dimensional HyperSlice example data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def create_dataset() -> xr.Dataset:
    """Build the canonical synthetic full-factorial sweep."""
    coords = {
        "fuel_temperature": np.array([600.0, 700.0, 850.0, 1050.0, 1300.0]),
        "drum_angle": np.array([0.0, 15.0, 30.0, 50.0, 75.0, 90.0]),
        "pressure": np.array([1.0, 2.5, 5.0, 8.0]),
        "flow_rate": np.array([2.0, 5.0, 10.0]),
        "burnup": np.array([0.0, 5.0, 20.0]),
    }
    ft, angle, pressure, flow, burnup = xr.broadcast(
        *(xr.DataArray(values, dims=name, coords={name: values}) for name, values in coords.items())
    )
    k_eff = (
        0.92
        + 0.0018 * angle
        - 0.00004 * (ft - 800)
        + 0.002 * pressure
        - 0.0005 * burnup
        + 0.000002 * angle * (ft - 800)
    )
    peak_temperature = (
        ft
        + 190 / np.sqrt(flow)
        + 5.5 * pressure
        + 0.015 * angle**2
        + 0.8 * burnup
        + 0.00015 * (ft - 800) * angle
    )
    status = xr.zeros_like(k_eff, dtype=np.int16)
    status = xr.where((ft >= 1050) & (flow == 2), 3, status)
    status = xr.where((pressure == 8) & (burnup == 20), 2, status)
    status.loc[dict(fuel_temperature=850, drum_angle=50, pressure=5, flow_rate=5, burnup=5)] = 4
    status.loc[dict(fuel_temperature=600, drum_angle=0, pressure=1, flow_rate=2, burnup=0)] = 1
    invalid = status != 0
    dataset = xr.Dataset(
        {
            "k_eff": k_eff.where(~invalid),
            "peak_temperature": peak_temperature.where(~invalid),
            "status": status,
        },
        attrs={
            "title": "HyperSlice synthetic irregular rectilinear sweep",
            "description": "Deterministic development dataset with semantic invalid regions.",
        },
    )
    coordinate_metadata = {
        "fuel_temperature": ("Fuel temperature", "K"),
        "drum_angle": ("Control drum angle", "degree"),
        "pressure": ("Pressure", "MPa"),
        "flow_rate": ("Flow rate", "kg/s"),
        "burnup": ("Burnup", "MWd/kg"),
    }
    for name, (long_name, units) in coordinate_metadata.items():
        dataset[name].attrs.update(long_name=long_name, units=units)
    dataset["k_eff"].attrs.update(long_name="Multiplication factor", units="dimensionless")
    dataset["peak_temperature"].attrs.update(long_name="Peak temperature", units="K")
    dataset["status"].attrs.update(
        long_name="Simulation status",
        flag_values=np.arange(6, dtype=np.int16),
        flag_meanings="valid not_evaluated nonconverged physically_invalid solver_error excluded",
        valid_values=np.array([0], dtype=np.int16),
    )
    return dataset


def main() -> None:
    """Write NetCDF and Zarr forms under examples/data."""
    target = Path(__file__).parent / "data"
    target.mkdir(parents=True, exist_ok=True)
    dataset = create_dataset()
    dataset.to_netcdf(target / "synthetic_sweep.nc")
    dataset.to_zarr(target / "synthetic_sweep.zarr", mode="w")
    print(f"Wrote example datasets to {target}")


if __name__ == "__main__":
    main()
