"""Create an example with missing Cartesian combinations."""

from __future__ import annotations

from create_synthetic_dataset import create_dataset

if __name__ == "__main__":
    dataset = create_dataset()
    dataset["k_eff"].loc[dict(fuel_temperature=700, drum_angle=30)] = float("nan")
    dataset.to_netcdf("examples/data/incomplete_sweep.nc")
