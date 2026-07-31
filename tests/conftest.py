"""Shared deterministic fixtures."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
