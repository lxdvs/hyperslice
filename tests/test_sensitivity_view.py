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
    html = render_strip(_row(), labels, by_magnitude=False)
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


def test_render_strip_sorted_leads_with_the_largest_magnitude() -> None:
    html = render_strip(_row(), {}, by_magnitude=True)
    order = [
        cell.split('<div class="hs-sens-input">')[1].split("<")[0]
        for cell in html.split('<div class="hs-sens-cell')[1:]
    ]
    assert order == ["b", "d", "a", "c"]


def test_render_strip_escapes_labels() -> None:
    html = render_strip(pd.Series({"a": 1.0}), {"a": "<b>&"}, by_magnitude=False)
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


def test_panel_shows_one_sortable_strip_per_output(dataset: xr.Dataset) -> None:
    schema = inspect_dataset(dataset)
    panel = SensitivityPanel(schema)
    assert panel.view.visible is False

    frame = sensitivity_frame(dataset, schema, _valid_point())
    panel.update(frame, "### Title")
    assert panel.view.visible is True
    assert panel.frame is frame
    blocks = panel.view[1]
    assert [block[0][0].object for block in blocks] == [
        "**Multiplication factor**",
        "**Peak temperature**",
    ]
    assert set(panel.sort_toggles) == {"k_eff", "peak_temperature"}
    assert blocks[0][0][1] is panel.sort_toggles["k_eff"]

    def shown_order(block: object) -> list[str]:
        html = block[1].object  # type: ignore[index]
        return [
            cell.split('<div class="hs-sens-input">')[1].split("<")[0]
            for cell in html.split('<div class="hs-sens-cell')[1:]
        ]

    labels = {name: schema.coordinates[name].long_name for name in frame.columns}
    assert shown_order(blocks[0]) == [labels[name] for name in frame.columns]

    panel.sort_toggles["k_eff"].value = True
    row = frame.loc["k_eff"]
    expected = [labels[name] for name in ordered_inputs(row, by_magnitude=True)]
    assert shown_order(blocks[0]) == expected
    assert expected != [labels[name] for name in frame.columns]
    # The other output keeps grid order: sorting is per output.
    assert shown_order(blocks[1]) == [labels[name] for name in frame.columns]

    # A new selection keeps the sort choice for that output.
    panel.update(sensitivity_frame(dataset, schema, _valid_point() | {"burnup": 20.0}), "### T")
    assert panel.sort_toggles["k_eff"].value is True
    assert panel.view[1][0][0][1] is panel.sort_toggles["k_eff"]

    panel.clear()
    assert panel.view.visible is False
    assert panel.frame is None
