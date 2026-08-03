from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import panel as pn
import xarray as xr

from hyperslice import Explorer


def _create_dataset() -> xr.Dataset:
    path = Path(__file__).parents[1] / "examples" / "create_mixed_input_dataset.py"
    spec = importlib.util.spec_from_file_location("mixed_example", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_dataset()


def test_mixed_example_has_four_string_options_and_finite_outputs() -> None:
    dataset = _create_dataset()
    assert dataset.sizes["configuration"] == 4
    assert dataset.configuration.dtype.kind in "OU"
    assert dataset.configuration.values.tolist() == [
        "baseline",
        "high_conductivity",
        "lightweight",
        "reinforced",
    ]
    for name in ("efficiency", "response", "stability_margin"):
        assert np.isfinite(dataset[name].values).all()
        assert "configuration" in dataset[name].dims


def test_string_coordinate_survives_netcdf_and_zarr_round_trip(
    tmp_path: Path,
) -> None:
    dataset = _create_dataset()
    netcdf = tmp_path / "mixed.nc"
    zarr = tmp_path / "mixed.zarr"
    dataset.to_netcdf(netcdf)
    dataset.to_zarr(zarr)
    for reopened in (xr.open_dataset(netcdf), xr.open_zarr(zarr)):
        assert reopened.configuration.values.tolist() == dataset.configuration.values.tolist()
        assert not bool(reopened["response"].isnull().any().compute())


def test_mixed_example_opens_with_categorical_slicer_and_filter_controls() -> None:
    explorer = Explorer(_create_dataset())
    assert explorer.x_dim != "configuration"
    assert explorer.y_dim != "configuration"
    assert explorer._plot.object is not None
    assert explorer._message.alert_type != "danger"
    assert isinstance(explorer._dimension_widgets["configuration"], pn.widgets.Select)
    assert explorer._axis_x_checks["configuration"].disabled
    assert explorer._axis_y_checks["configuration"].disabled
    categorical_filter = explorer.filter_view._filter_widgets["configuration"]
    assert isinstance(categorical_filter, pn.widgets.MultiChoice)
    assert categorical_filter.options == [
        "baseline",
        "high_conductivity",
        "lightweight",
        "reinforced",
    ]
    categorical_filter.value = ["baseline", "reinforced"]
    assert explorer.filter_view._plot.object is not None
    assert not explorer.filter_view._message.visible
