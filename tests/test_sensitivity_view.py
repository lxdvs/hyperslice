from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hyperslice.schema import inspect_dataset
from hyperslice.sensitivity import UNDEFINED, sensitivity_frame
from hyperslice.sensitivity_view import (
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    SENSITIVITY_STYLES,
    SensitivityPanel,
    bar_fractions,
    ordered_inputs,
    render_strip,
)


def _row() -> pd.Series:
    return pd.Series({"a": 0.5, "b": -2.0, "c": np.nan, "d": 1.0})


def test_bar_fractions_scale_to_the_largest_magnitude() -> None:
    fractions = bar_fractions(_row())
    assert fractions["b"] == pytest.approx(1.0)
    assert fractions["d"] == pytest.approx(0.5)
    assert fractions["a"] == pytest.approx(0.25)
    assert fractions["c"] == 0.0
    assert (bar_fractions(pd.Series({"a": 0.0, "b": np.nan})) == 0.0).all()


def test_ordered_inputs_by_magnitude_puts_undefined_last() -> None:
    assert ordered_inputs(_row(), by_magnitude=False) == ["a", "b", "c", "d"]
    assert ordered_inputs(_row(), by_magnitude=True) == ["b", "d", "a", "c"]


def test_render_strip_draws_signed_bars_about_a_centre_line() -> None:
    labels = {"a": "Alpha", "b": "Beta", "d": "Delta"}
    html = render_strip(_row(), labels)
    cells = html.split('<div class="hs-sens-cell')[1:]
    assert len(cells) == 4
    assert "Alpha" in cells[0] and 'class="hs-sens-fill up" style="height: 12.5%"' in cells[0]
    assert "Beta" in cells[1] and 'class="hs-sens-fill down" style="height: 50.0%"' in cells[1]
    assert cells[2].startswith(" undefined") and UNDEFINED in cells[2]
    assert "hs-sens-fill" not in cells[2]
    assert "Delta" in cells[3] and 'class="hs-sens-fill up" style="height: 25.0%"' in cells[3]
    assert all('<div class="hs-sens-axis"></div>' in cell for cell in cells)
    assert f"background: {POSITIVE_COLOR}" in SENSITIVITY_STYLES
    assert f"background: {NEGATIVE_COLOR}" in SENSITIVITY_STYLES


def test_render_strip_follows_the_given_order() -> None:
    html = render_strip(_row(), {}, ordered_inputs(_row(), by_magnitude=True))
    order = [
        cell.split('<div class="hs-sens-input">')[1].split("<")[0]
        for cell in html.split('<div class="hs-sens-cell')[1:]
    ]
    assert order == ["b", "d", "a", "c"]


def test_render_strip_escapes_labels() -> None:
    html = render_strip(pd.Series({"a": 1.0}), {"a": "<b>&"})
    assert "&lt;b&gt;&amp;" in html
    assert "<b>&" not in html


def _valid_point() -> dict[str, float]:
    return {
        "fuel_temperature": 850.0,
        "drum_angle": 30.0,
        "pressure": 2.5,
        "flow_rate": 5.0,
        "burnup": 5.0,
    }


def _shown_order(block: object) -> list[str]:
    html = block[1].object  # type: ignore[index]
    return [
        cell.split('<div class="hs-sens-input">')[1].split("<")[0]
        for cell in html.split('<div class="hs-sens-cell')[1:]
    ]


def test_panel_shows_one_strip_per_output(dataset: xr.Dataset) -> None:
    schema = inspect_dataset(dataset)
    panel = SensitivityPanel(schema)
    assert panel.view.visible is False

    frame = sensitivity_frame(dataset, schema, _valid_point())
    panel.update(frame, "### Title")
    assert panel.view.visible is True
    assert panel.frame is frame
    blocks = panel.view[1]
    # Outputs run from the most distinct values to the fewest.
    assert [block[0][0].object for block in blocks] == [
        "**Peak temperature**",
        "**Multiplication factor**",
    ]
    assert set(panel.sort_toggles) == {"k_eff", "peak_temperature"}
    assert blocks[0][0][1] is panel.sort_toggles["peak_temperature"]
    assert blocks[1][0][1] is panel.sort_toggles["k_eff"]
    labels = {name: schema.coordinates[name].long_name for name in frame.columns}
    grid = [labels[name] for name in frame.columns]
    assert [_shown_order(block) for block in blocks] == [grid, grid]

    panel.clear()
    assert panel.view.visible is False
    assert panel.frame is None


def test_sorting_by_one_output_orders_every_strip_the_same_way(dataset: xr.Dataset) -> None:
    schema = inspect_dataset(dataset)
    panel = SensitivityPanel(schema)
    frame = sensitivity_frame(dataset, schema, _valid_point())
    panel.update(frame, "### Title")
    blocks = panel.view[1]
    labels = {name: schema.coordinates[name].long_name for name in frame.columns}
    grid = [labels[name] for name in frame.columns]

    panel.sort_toggles["k_eff"].value = True
    by_k_eff = [labels[name] for name in ordered_inputs(frame.loc["k_eff"], by_magnitude=True)]
    assert by_k_eff != grid
    assert panel.sort_by == "k_eff"
    assert [_shown_order(block) for block in blocks] == [by_k_eff, by_k_eff]

    # Choosing another output's order switches the first toggle off.
    panel.sort_toggles["peak_temperature"].value = True
    by_peak = [
        labels[name] for name in ordered_inputs(frame.loc["peak_temperature"], by_magnitude=True)
    ]
    assert by_peak != by_k_eff
    assert panel.sort_by == "peak_temperature"
    assert panel.sort_toggles["k_eff"].value is False
    assert [_shown_order(block) for block in blocks] == [by_peak, by_peak]

    # A new selection keeps the ordering output.
    panel.update(sensitivity_frame(dataset, schema, _valid_point() | {"burnup": 20.0}), "### T")
    assert panel.sort_by == "peak_temperature"
    assert panel.sort_toggles["peak_temperature"].value is True
    assert panel.view[1][0][0][1] is panel.sort_toggles["peak_temperature"]

    # Switching the active toggle off returns every strip to grid order.
    panel.sort_toggles["peak_temperature"].value = False
    assert panel.sort_by is None
    assert [_shown_order(block) for block in panel.view[1]] == [grid, grid]
