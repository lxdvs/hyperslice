"""Create a complete rectilinear dataset with no uniformly spaced inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def create_dataset() -> xr.Dataset:
    """Build a deterministic complete sweep with irregular numeric coordinates."""
    coordinates = {
        "inlet_temperature": np.array([285.0, 310.0, 355.0, 430.0, 560.0, 735.0, 980.0]),
        "pressure": np.array([0.7, 1.1, 2.0, 3.8, 7.1, 12.0]),
        "mass_flow": np.array([0.3, 0.8, 1.7, 3.5, 7.0]),
        "blend_fraction": np.array([0.02, 0.14, 0.39, 0.81]),
        "aspect_ratio": np.array([0.55, 0.9, 1.4, 2.2]),
    }
    temperature, pressure, flow, blend, aspect = xr.broadcast(
        *(
            xr.DataArray(values, dims=name, coords={name: values})
            for name, values in coordinates.items()
        )
    )
    conversion = (
        0.32
        + 0.48 * (1.0 - np.exp(-(temperature - 250.0) / 380.0))
        + 0.055 * np.log1p(pressure)
        + 0.04 * np.sqrt(flow)
        + 0.12 * blend * (1.0 - blend)
        - 0.035 * (aspect - 1.25) ** 2
    )
    peak_response = (
        18.0
        + 0.027 * temperature
        + 2.1 * pressure**0.65
        - 0.75 * np.log1p(flow)
        + 9.0 * blend
        + 3.2 / aspect
        + 0.003 * temperature * blend
    )
    stability_index = (
        7.5
        - 0.0035 * (temperature - 500.0)
        - 0.18 * pressure
        + 0.62 * np.sqrt(flow)
        + 1.4 * (1.0 - blend)
        + 0.85 * np.log1p(aspect)
    )
    dataset = xr.Dataset(
        {
            "conversion": conversion,
            "peak_response": peak_response,
            "stability_index": stability_index,
            "status": xr.zeros_like(conversion, dtype=np.int16),
        },
        attrs={
            "title": "HyperSlice fully irregular rectilinear sweep",
            "description": (
                "Complete full-factorial dataset whose independent numeric "
                "coordinates all have unequal adjacent spacing."
            ),
        },
    )
    coordinate_metadata = {
        "inlet_temperature": ("Inlet temperature", "K"),
        "pressure": ("Pressure", "MPa"),
        "mass_flow": ("Mass flow", "kg/s"),
        "blend_fraction": ("Blend fraction", "1"),
        "aspect_ratio": ("Aspect ratio", "1"),
    }
    for name, (long_name, units) in coordinate_metadata.items():
        dataset[name].attrs.update(long_name=long_name, units=units)
    dataset["conversion"].attrs.update(long_name="Conversion", units="1")
    dataset["peak_response"].attrs.update(long_name="Peak response", units="a.u.")
    dataset["stability_index"].attrs.update(long_name="Stability index", units="1")
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
    dataset.to_netcdf(target / "fully_irregular_sweep.nc")
    dataset.to_zarr(target / "fully_irregular_sweep.zarr", mode="w")
    print(f"Wrote fully irregular example datasets to {target}")


if __name__ == "__main__":
    main()
