"""Create a complete dataset with numeric and categorical input coordinates."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr


def _load_complete_generator() -> object:
    path = Path(__file__).with_name("create_complete_dataset.py")
    spec = importlib.util.spec_from_file_location("complete_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load complete dataset generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_dataset() -> xr.Dataset:
    """Build a complete dataset with four categorical configuration options."""
    base = _load_complete_generator().create_dataset()
    configurations = np.array(
        ["baseline", "high_conductivity", "lightweight", "reinforced"],
        dtype=str,
    )
    efficiency_factor = xr.DataArray(
        [1.0, 1.045, 0.975, 1.015],
        dims="configuration",
        coords={"configuration": configurations},
    )
    response_offset = xr.DataArray(
        [0.0, -1.8, 2.4, 1.1],
        dims="configuration",
        coords={"configuration": configurations},
    )
    margin_offset = xr.DataArray(
        [0.0, 0.65, -0.4, 1.25],
        dims="configuration",
        coords={"configuration": configurations},
    )
    dataset = xr.Dataset(
        {
            "efficiency": base["efficiency"] * efficiency_factor,
            "response": base["response"] + response_offset,
            "stability_margin": base["stability_margin"] + margin_offset,
        },
        attrs={
            "title": "HyperSlice mixed numeric and categorical sweep",
            "description": (
                "Complete deterministic sweep with six numeric inputs and one "
                "four-option string input."
            ),
        },
    )
    dataset["status"] = xr.zeros_like(dataset["response"], dtype=np.int16)
    for name in base.coords:
        dataset[name].attrs = dict(base[name].attrs)
    dataset["configuration"].attrs.update(
        long_name="Design configuration",
        description="Categorical engineering configuration option",
    )
    for name in ("efficiency", "response", "stability_margin"):
        dataset[name].attrs = dict(base[name].attrs)
    dataset["status"].attrs = dict(base["status"].attrs)
    return dataset


def main() -> None:
    """Write NetCDF and Zarr forms under examples/data."""
    target = Path(__file__).parent / "data"
    target.mkdir(parents=True, exist_ok=True)
    dataset = create_dataset()
    dataset.to_netcdf(target / "mixed_input_sweep.nc")
    dataset.to_zarr(target / "mixed_input_sweep.zarr", mode="w")
    print(f"Wrote mixed-input example datasets to {target}")


if __name__ == "__main__":
    main()
