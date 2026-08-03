"""Create a rectilinear dataset whose inputs cluster densely mid-range.

Two of the three numeric input dimensions concentrate their sample points
around the middle of their span, where the modelled resonance makes the
outputs change fastest; the third dimension stays uniformly spaced for
contrast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def center_dense(low: float, high: float, count: int, sharpness: float = 3.5) -> np.ndarray:
    """Return *count* points on [low, high] clustered around the interval middle.

    A sinh warp of a uniform grid keeps the endpoints fixed while shrinking
    adjacent spacing near the center by a factor of ``cosh(sharpness)``
    (about 16 for the default sharpness) relative to the edges.
    """
    uniform = np.linspace(-1.0, 1.0, count)
    warped = np.sinh(sharpness * uniform) / np.sinh(sharpness)
    center, half_width = (high + low) / 2.0, (high - low) / 2.0
    return center + half_width * warped


def create_dataset() -> xr.Dataset:
    """Build a deterministic resonance sweep with center-dense coordinates."""
    coordinates = {
        "drive_frequency": center_dense(40.0, 160.0, 41),
        "bias_voltage": center_dense(-12.0, 12.0, 33),
        "coolant_flow": np.linspace(0.5, 4.5, 7),
    }
    frequency, voltage, flow = xr.broadcast(
        *(
            xr.DataArray(values, dims=name, coords={name: values})
            for name, values in coordinates.items()
        )
    )
    detuning = (frequency - 100.0) / (6.0 + 1.5 * flow)
    amplitude = 0.4 + 9.0 / (1.0 + detuning**2 + 0.35 * voltage**2) + 0.05 * flow
    phase_shift = np.arctan(detuning) * (180.0 / np.pi) + 4.0 * np.tanh(voltage / 5.0) - 1.2 * flow
    quality_factor = (
        55.0 + 30.0 / (1.0 + 0.5 * detuning**2) - 0.9 * np.abs(voltage) + 3.5 * np.sqrt(flow)
    )
    dataset = xr.Dataset(
        {
            "amplitude": amplitude,
            "phase_shift": phase_shift,
            "quality_factor": quality_factor,
            "status": xr.zeros_like(amplitude, dtype=np.int16),
        },
        attrs={
            "title": "HyperSlice center-dense rectilinear sweep",
            "description": (
                "Complete full-factorial resonance scan whose drive-frequency "
                "and bias-voltage coordinates cluster densely around the "
                "middle of their ranges."
            ),
        },
    )
    coordinate_metadata = {
        "drive_frequency": ("Drive frequency", "kHz"),
        "bias_voltage": ("Bias voltage", "V"),
        "coolant_flow": ("Coolant flow", "L/min"),
    }
    for name, (long_name, units) in coordinate_metadata.items():
        dataset[name].attrs.update(long_name=long_name, units=units)
    dataset["amplitude"].attrs.update(long_name="Response amplitude", units="mm")
    dataset["phase_shift"].attrs.update(long_name="Phase shift", units="deg")
    dataset["quality_factor"].attrs.update(long_name="Quality factor", units="1")
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
    dataset.to_netcdf(target / "center_dense_sweep.nc")
    dataset.to_zarr(target / "center_dense_sweep.zarr", mode="w")
    print(f"Wrote center-dense example datasets to {target}")


if __name__ == "__main__":
    main()
