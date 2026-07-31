"""Create a complete six-dimensional irregular rectilinear example dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def create_dataset() -> xr.Dataset:
    """Build a deterministic 12 x 9 x 10 x 4 x 3 x 7 dataset without missing data."""
    coordinates = {
        "temperature": np.array(
            [280.0, 315.0, 360.0, 420.0, 490.0, 575.0, 670.0, 780.0, 905.0, 1050.0, 1215.0, 1400.0]
        ),
        "pressure": np.array([0.8, 1.2, 1.9, 2.8, 4.0, 5.5, 7.3, 9.6, 12.5]),
        "flow_rate": np.array([0.5, 0.9, 1.5, 2.4, 3.6, 5.1, 7.2, 9.8, 13.0, 17.0]),
        "composition": np.array([0.05, 0.18, 0.42, 0.75]),
        "aspect_ratio": np.array([0.6, 1.0, 1.8]),
        "control_setting": np.array([0.0, 0.08, 0.2, 0.38, 0.58, 0.79, 1.0]),
    }
    temperature, pressure, flow, composition, aspect, control = xr.broadcast(
        *(
            xr.DataArray(values, dims=name, coords={name: values})
            for name, values in coordinates.items()
        )
    )
    efficiency = (
        0.58
        + 0.16 * np.tanh((temperature - 500.0) / 420.0)
        - 0.012 * (pressure - 5.0) ** 2 / 25.0
        + 0.035 * np.log1p(flow)
        + 0.08 * composition * (1.0 - composition)
        - 0.025 * (aspect - 1.1) ** 2
        + 0.09 * control
        + 0.018 * control * composition
    )
    response = (
        12.0
        + 0.018 * temperature
        + 1.7 * np.sqrt(pressure)
        - 0.42 * flow
        + 7.5 * composition
        + 2.2 / aspect
        + 5.0 * np.sin(np.pi * control)
        + 0.0025 * temperature * composition
        - 0.08 * pressure * control
    )
    stability_margin = (
        4.5
        + 0.004 * (1000.0 - temperature)
        + 0.3 * flow
        - 0.22 * pressure
        + 1.8 * (1.0 - composition)
        + 0.7 * aspect
        - 2.4 * control**2
    )
    status = xr.zeros_like(response, dtype=np.int16)
    dataset = xr.Dataset(
        {
            "efficiency": efficiency,
            "response": response,
            "stability_margin": stability_margin,
            "status": status,
        },
        attrs={
            "title": "HyperSlice complete six-dimensional sweep",
            "description": (
                "Deterministic full-factorial irregular rectilinear dataset "
                "with no missing or invalid samples."
            ),
        },
    )
    metadata = {
        "temperature": ("Temperature", "K"),
        "pressure": ("Pressure", "MPa"),
        "flow_rate": ("Flow rate", "kg/s"),
        "composition": ("Composition fraction", "1"),
        "aspect_ratio": ("Aspect ratio", "1"),
        "control_setting": ("Control setting", "1"),
    }
    for name, (long_name, units) in metadata.items():
        dataset[name].attrs.update(long_name=long_name, units=units)
    dataset["efficiency"].attrs.update(long_name="Efficiency", units="1")
    dataset["response"].attrs.update(long_name="Aggregate response", units="a.u.")
    dataset["stability_margin"].attrs.update(long_name="Stability margin", units="a.u.")
    dataset["status"].attrs.update(
        long_name="Simulation status",
        flag_values=np.array([0], dtype=np.int16),
        flag_meanings="valid",
        valid_values=np.array([0], dtype=np.int16),
    )
    return dataset


def main() -> None:
    """Write NetCDF and Zarr forms under examples/data."""
    target = Path(__file__).parent / "data"
    target.mkdir(parents=True, exist_ok=True)
    dataset = create_dataset()
    dataset.to_netcdf(target / "complete_sweep.nc")
    dataset.to_zarr(target / "complete_sweep.zarr", mode="w")
    print(f"Wrote complete example datasets to {target}")


if __name__ == "__main__":
    main()
