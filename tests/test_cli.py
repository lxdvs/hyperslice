from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import xarray as xr
from typer.testing import CliRunner

from hyperslice import cli
from hyperslice.schema import cardinality_table, inspect_dataset


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cli.pn, "serve", lambda *args, **kwargs: calls.append(kwargs))
    return calls


def test_show_cardinality_prints_the_table_then_serves(
    dataset: xr.Dataset, tmp_path: Path, served: list[dict[str, Any]]
) -> None:
    path = tmp_path / "sweep.nc"
    dataset.to_netcdf(path)
    result = CliRunner().invoke(cli.app, [str(path), "--show-cardinality", "--no-browser"])
    assert result.exit_code == 0, result.output
    assert cardinality_table(inspect_dataset(dataset)) in result.output
    assert len(served) == 1


def test_cardinality_is_not_printed_by_default(
    dataset: xr.Dataset, tmp_path: Path, served: list[dict[str, Any]]
) -> None:
    path = tmp_path / "sweep.nc"
    dataset.to_netcdf(path)
    result = CliRunner().invoke(cli.app, [str(path), "--no-browser"])
    assert result.exit_code == 0, result.output
    assert "Cardinality" not in result.output
    assert len(served) == 1
