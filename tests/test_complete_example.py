from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr

from hyperslice import Explorer


def test_complete_example_has_requested_shape_and_no_missing_data() -> None:
    path = Path(__file__).parents[1] / "examples" / "create_complete_dataset.py"
    spec = importlib.util.spec_from_file_location("complete_example", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = module.create_dataset()

    assert tuple(dataset.sizes.values()) == (12, 9, 10, 4, 3, 7)
    assert int(dataset.status.max()) == 0
    for name in ("efficiency", "response", "stability_margin"):
        assert np.isfinite(dataset[name].values).all()


def test_complete_example_opens_in_explorer_after_netcdf_round_trip(
    tmp_path: Path,
) -> None:
    path = Path(__file__).parents[1] / "examples" / "create_complete_dataset.py"
    spec = importlib.util.spec_from_file_location("complete_example_roundtrip", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    netcdf = tmp_path / "complete.nc"
    module.create_dataset().to_netcdf(netcdf)

    reopened = xr.open_dataset(netcdf)
    assert np.asarray(reopened.status.attrs["flag_values"]).ndim == 0
    explorer = Explorer(netcdf)
    assert explorer._plot.object is not None
    assert explorer._message.alert_type != "danger"
