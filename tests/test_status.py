from __future__ import annotations

import xarray as xr

from hyperslice.slicing import make_slice
from hyperslice.status import classify_slice, parse_status_definition


def test_cf_status_and_counts(dataset: xr.Dataset) -> None:
    selections = {"pressure": 1.0, "flow_rate": 2.0, "burnup": 0.0}
    output = make_slice(
        dataset.k_eff, x_dim="drum_angle", y_dim="fuel_temperature", selections=selections
    )
    result = classify_slice(
        output,
        status=dataset.status,
        x_dim="drum_angle",
        y_dim="fuel_temperature",
        selections=selections,
        method="exact",
    )
    assert result.counts["missing"] > 0
    assert result.labels[3] == "physically invalid"
    assert parse_status_definition(dataset.status).valid_values == frozenset({0})


def test_boolean_validity(dataset: xr.Dataset) -> None:
    output = dataset.k_eff.sel(pressure=2.5, flow_rate=5, burnup=5)
    validity = dataset.status == 0
    result = classify_slice(
        output,
        validity=validity,
        x_dim="drum_angle",
        y_dim="fuel_temperature",
        selections={"pressure": 2.5, "flow_rate": 5, "burnup": 5},
        method="exact",
    )
    assert isinstance(result.mask_valid, xr.DataArray)
