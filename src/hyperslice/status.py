"""Semantic status and validity handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from hyperslice.exceptions import SliceError
from hyperslice.slicing import SliceMethod, make_slice


@dataclass(frozen=True)
class StatusDefinition:
    """Meaning and validity of integer flag values."""

    valid_values: frozenset[int]
    labels: dict[int, str]


@dataclass(frozen=True)
class SliceStatus:
    """Aligned semantic masks for one displayed slice."""

    mask_valid: xr.DataArray
    mask_invalid: xr.DataArray
    mask_missing: xr.DataArray
    labels: dict[int, str]
    status: xr.DataArray | None = None

    @property
    def counts(self) -> dict[str, int]:
        """Compute valid, invalid, and missing cell counts."""
        return {
            "valid": int(self.mask_valid.sum().compute().item()),
            "invalid": int(self.mask_invalid.sum().compute().item()),
            "missing": int(self.mask_missing.sum().compute().item()),
        }


def parse_status_definition(status: xr.DataArray) -> StatusDefinition:
    """Parse CF flag metadata, accepting an explicit ``valid_values`` attribute."""
    raw_values = status.attrs.get("flag_values", [])
    raw_meanings = str(status.attrs.get("flag_meanings", "")).split()
    labels = {
        int(value): meaning.replace("_", " ")
        for value, meaning in zip(raw_values, raw_meanings, strict=False)
    }
    valid_attr = status.attrs.get("valid_values")
    if valid_attr is not None:
        valid_values = frozenset(int(value) for value in np.atleast_1d(valid_attr))
    elif labels:
        valid_values = frozenset(
            value
            for value, label in labels.items()
            if label.lower() in {"valid", "success", "converged"}
        )
    else:
        valid_values = frozenset({0})
    return StatusDefinition(valid_values or frozenset({0}), labels)


def _align_status(
    status: xr.DataArray,
    output: xr.DataArray,
    *,
    x_dim: str,
    y_dim: str,
    selections: dict[str, Any],
    method: SliceMethod,
) -> xr.DataArray:
    extra = set(status.dims) - set(output.dims)
    missing_selection = sorted(extra - set(selections))
    if missing_selection:
        raise SliceError(f"Status variable requires selections for dimensions: {missing_selection}")
    if x_dim in status.dims and y_dim in status.dims:
        sliced = make_slice(
            status,
            x_dim=x_dim,
            y_dim=y_dim,
            selections=selections,
            method="nearest" if method == "linear" else method,
        )
    else:
        sliced = status.sel({dim: selections[dim] for dim in extra})
        if any(dim not in {x_dim, y_dim} for dim in sliced.dims):
            raise SliceError("Status variable cannot be aligned unambiguously to the output slice.")
    try:
        return sliced.broadcast_like(output).transpose(y_dim, x_dim)
    except ValueError as exc:
        raise SliceError(
            f"Status variable is incompatible with the selected output: {exc}"
        ) from exc


def classify_slice(
    output: xr.DataArray,
    *,
    status: xr.DataArray | None = None,
    validity: xr.DataArray | None = None,
    x_dim: str,
    y_dim: str,
    selections: dict[str, Any],
    method: SliceMethod,
) -> SliceStatus:
    """Classify output cells without collapsing invalid and missing semantics."""
    missing = output.isnull()
    labels: dict[int, str] = {}
    aligned_status: xr.DataArray | None = None
    semantic_valid = xr.ones_like(output, dtype=bool)
    if status is not None:
        definition = parse_status_definition(status)
        aligned_status = _align_status(
            status, output, x_dim=x_dim, y_dim=y_dim, selections=selections, method=method
        )
        semantic_valid = aligned_status.isin(list(definition.valid_values))
        labels = definition.labels
    if validity is not None:
        aligned_validity = _align_status(
            validity, output, x_dim=x_dim, y_dim=y_dim, selections=selections, method=method
        )
        semantic_valid = semantic_valid & aligned_validity.astype(bool)
    valid = semantic_valid & ~missing
    invalid = ~semantic_valid & ~missing
    return SliceStatus(valid, invalid, missing, labels, aligned_status)


def apply_strict_validity(
    data: xr.DataArray, validity: xr.DataArray | None, status: xr.DataArray | None
) -> xr.DataArray:
    """Mask source samples before interpolation so invalid support is never filled."""
    mask = data.notnull()
    if validity is not None:
        try:
            mask = mask & validity.broadcast_like(data).astype(bool)
        except ValueError as exc:
            raise SliceError(f"Validity variable cannot be broadcast to output: {exc}") from exc
    if status is not None:
        definition = parse_status_definition(status)
        try:
            mask = mask & status.broadcast_like(data).isin(list(definition.valid_values))
        except ValueError as exc:
            raise SliceError(f"Status variable cannot be broadcast to output: {exc}") from exc
    return data.where(mask)
