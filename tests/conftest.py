"""Shared deterministic fixtures."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import param
import pytest
import xarray as xr


@pytest.fixture
def dataset() -> xr.Dataset:
    path = Path(__file__).parents[1] / "examples" / "create_synthetic_dataset.py"
    spec = importlib.util.spec_from_file_location("synthetic", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_dataset()


def drag(widget: Any, value: Any) -> None:
    """Move a slider the way the browser does: live value, then a settled one.

    Views redraw on ``value_throttled`` so a drag never rebuilds the plot
    mid-gesture; setting ``value`` alone therefore updates nothing.
    """
    widget.value = value
    if "value_throttled" in widget.param:
        with param.edit_constant(widget):
            widget.value_throttled = value
