from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import panel as pn
import pytest
from typer.testing import CliRunner

from hyperslice import Explorer, cli
from hyperslice.exceptions import DatasetLoadError
from hyperslice.explorer import scattered_notice
from hyperslice.loading import SPARSE_CELLS_PER_POINT, load_dataset
from hyperslice.schema import (
    LAYOUT_ATTR,
    POINTS_LAYOUT,
    SAMPLE_DIM,
    cardinality_table,
    inspect_dataset,
)


def _sobol_like(count: int = 40) -> list[dict[str, Any]]:
    """Points with three continuous inputs, every value distinct, and one discrete."""
    rng = np.random.default_rng(7)
    records = []
    for index in range(count):
        a, b, c = (float(value) for value in rng.uniform(1.0, 2.0, size=3))
        rings = 4 + index % 2
        outputs: dict[str, Any] = {
            "run": {"status": index % 3},
            "geometry": {"mass": 10.0 * a + b**2 - c, "rings": rings},
        }
        if index % 5 == 0:
            outputs["geometry"].pop("mass")
        records.append({"inputs": {"a": a, "b": b, "c": c, "n_rings": rings}, "outputs": outputs})
    return records


def _factorial() -> list[dict[str, Any]]:
    return [
        {"inputs": {"a": a, "b": b}, "outputs": {"y": a + b}}
        for a in (1.0, 2.0, 3.0)
        for b in (10.0, 20.0)
    ]


def _write(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps(records))
    return path


def test_sparse_points_load_as_samples_without_a_grid(tmp_path: Path) -> None:
    records = _sobol_like()
    # 40 x 40 x 40 x 2 grid cells for 40 points: far past the sparse threshold.
    assert SPARSE_CELLS_PER_POINT * len(records) < 40**3 * 2
    dataset = load_dataset(_write(tmp_path / "sobol.json", records))
    assert dataset.attrs[LAYOUT_ATTR] == POINTS_LAYOUT
    assert dict(dataset.sizes) == {SAMPLE_DIM: 40}
    assert {"a", "b", "c", "n_rings"} <= set(dataset.coords)
    assert dataset.coords["a"].dims == (SAMPLE_DIM,)
    assert dataset["geometry.mass"].dims == (SAMPLE_DIM,)
    # A point that does not report an output leaves it missing, not absent.
    assert np.isnan(dataset["geometry.mass"].values[0])
    assert np.isfinite(dataset["geometry.mass"].values[1])


def test_a_filled_factorial_still_loads_as_a_grid(tmp_path: Path) -> None:
    path = _write(tmp_path / "grid.json", _factorial())
    dataset = load_dataset(path)
    assert LAYOUT_ATTR not in dataset.attrs
    assert dict(dataset.sizes) == {"a": 3, "b": 2}
    forced = load_dataset(path, layout="points")
    assert forced.attrs[LAYOUT_ATTR] == POINTS_LAYOUT
    assert dict(forced.sizes) == {SAMPLE_DIM: 6}


def test_points_need_the_same_inputs_and_distinct_names(tmp_path: Path) -> None:
    records = _factorial()
    records[2] = {"inputs": {"a": 2.0}, "outputs": {"y": 1.0}}
    with pytest.raises(DatasetLoadError, match="has inputs"):
        load_dataset(_write(tmp_path / "ragged.json", records), layout="points")
    clash = [{"inputs": {"a": 1.0}, "outputs": {"a": 2.0}}]
    with pytest.raises(DatasetLoadError, match="both input and output"):
        load_dataset(_write(tmp_path / "clash.json", clash), layout="points")


def test_schema_lists_inputs_and_outputs_of_scattered_samples(tmp_path: Path) -> None:
    schema = inspect_dataset(load_dataset(_write(tmp_path / "sobol.json", _sobol_like())))
    assert schema.scattered
    assert set(schema.coordinates) == {"a", "b", "c", "n_rings"}
    assert schema.coordinates["a"].size == 40
    assert schema.coordinates["n_rings"].size == 2
    assert not schema.coordinates["n_rings"].constant
    assert set(schema.variables) == {"geometry.mass", "geometry.rings", "run.status"}
    mass = schema.variables["geometry.mass"]
    assert mass.dims == ("a", "b", "c", "n_rings")
    assert mass.shape == (40,)
    assert mass.cardinality == 32
    table = cardinality_table(schema)
    assert table.splitlines()[0].split()[-1] == "Samples"
    assert "40 samples" in table


def test_scattered_layout_survives_netcdf(tmp_path: Path) -> None:
    dataset = load_dataset(_write(tmp_path / "sobol.json", _sobol_like()))
    dataset.to_netcdf(tmp_path / "sobol.nc")
    assert inspect_dataset(load_dataset(tmp_path / "sobol.nc")).scattered


def test_explorer_offers_only_the_filter_tab_behind_a_notice(tmp_path: Path) -> None:
    explorer = Explorer(_write(tmp_path / "sobol.json", _sobol_like()))
    assert explorer.schema.scattered
    assert list(explorer.tabs._names) == ["Filter"]
    assert explorer.tabs[0] is explorer.filter_view.view
    assert not hasattr(explorer, "slicer_view")
    assert isinstance(explorer.notice, pn.Modal)
    assert explorer.notice.open
    assert not explorer.notice.show_close_button
    assert not explorer.notice.background_close
    assert explorer.notice in explorer.view and explorer.tabs in explorer.view
    explorer.dismiss_notice.clicks += 1
    assert not explorer.notice.open

    view = explorer.filter_view
    assert len(view._sample_frame()) == 40
    assert set(view._filter_widgets) >= {"a", "b", "c", "n_rings", "geometry.mass"}
    assert "40 samples" in view._correlation_title.object
    assert "<table" in view._correlation.object


def test_notice_explains_the_mode(tmp_path: Path) -> None:
    schema = inspect_dataset(load_dataset(_write(tmp_path / "sobol.json", _sobol_like())))
    text = scattered_notice(schema, 40)
    assert "non-rectilinear mode" in text
    assert "40 samples" in text and "128,000 grid" in text
    assert "Filter" in text and "Slicer" in text


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cli.pn, "serve", lambda *args, **kwargs: calls.append(kwargs))
    return calls


def test_cli_warns_about_scattered_mode(tmp_path: Path, served: list[dict[str, Any]]) -> None:
    path = _write(tmp_path / "sobol.json", _sobol_like())
    result = CliRunner().invoke(cli.app, [str(path), "--no-browser", "--show-cardinality"])
    assert result.exit_code == 0, result.output
    assert "Warning: Running in non-rectilinear mode" in result.output
    assert "40 samples" in result.output
    assert len(served) == 1


def test_cli_points_flag_forces_scattered_mode(
    tmp_path: Path, served: list[dict[str, Any]]
) -> None:
    path = _write(tmp_path / "grid.json", _factorial())
    plain = CliRunner().invoke(cli.app, [str(path), "--no-browser"])
    assert "non-rectilinear" not in plain.output
    forced = CliRunner().invoke(cli.app, [str(path), "--no-browser", "--points"])
    assert forced.exit_code == 0, forced.output
    assert "non-rectilinear" in forced.output
