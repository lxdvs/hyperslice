from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr

from hyperslice import Explorer


def _create_dataset() -> xr.Dataset:
    path = Path(__file__).parents[1] / "examples" / "create_fully_irregular_dataset.py"
    spec = importlib.util.spec_from_file_location("fully_irregular_example", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_dataset()


def test_every_input_is_independent_numeric_and_irregularly_spaced() -> None:
    dataset = _create_dataset()
    for dim in dataset.dims:
        coordinate = dataset.coords[dim]
        assert coordinate.dims == (dim,)
        assert coordinate.dtype.kind in "iuf"
        spacing = np.diff(coordinate.values)
        assert spacing.size >= 2
        assert not np.allclose(spacing, spacing[0])


def test_irregular_example_is_complete_and_finite() -> None:
    dataset = _create_dataset()
    expected_cells = int(np.prod(list(dataset.sizes.values())))
    assert expected_cells == 3_360
    for name in ("conversion", "peak_response", "stability_index"):
        assert dataset[name].size == expected_cells
        assert np.isfinite(dataset[name].values).all()
    assert int(dataset.status.max()) == 0


def test_irregular_example_opens_in_explorer() -> None:
    explorer = Explorer(_create_dataset())
    assert explorer._plot.object is not None
    assert explorer.filter_view._plot.object is not None
    assert explorer._message.alert_type != "danger"
