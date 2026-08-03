"""Incremental construction of rectilinear HyperSlice datasets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


class DatasetBuilderError(ValueError):
    """Base exception raised for invalid dataset-builder operations."""


class InputSchemaMismatchError(DatasetBuilderError):
    """A design point uses different input parameter keys."""


class OutputOverwriteError(DatasetBuilderError):
    """An additive update would overwrite an existing point output."""


class InvalidParameterError(DatasetBuilderError):
    """An input or output parameter is not a supported scalar value."""


class DatasetBuilder:
    """Accumulate design points and materialize an xarray rectilinear dataset.

    Repeated calls for an identical input point are additive: new output keys
    are merged into that point. Existing output keys can never be overwritten.
    """

    def __init__(
        self,
        *,
        coordinate_attrs: Mapping[str, Mapping[str, Any]] | None = None,
        variable_attrs: Mapping[str, Mapping[str, Any]] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        self.coordinate_attrs = {
            name: dict(metadata) for name, metadata in (coordinate_attrs or {}).items()
        }
        self.variable_attrs = {
            name: dict(metadata) for name, metadata in (variable_attrs or {}).items()
        }
        self.attrs = dict(attrs or {})
        self._input_names: tuple[str, ...] | None = None
        self._points: dict[tuple[Any, ...], dict[str, float]] = {}
        self._levels: dict[str, list[Any]] = {}
        self._output_names: list[str] = []

    @property
    def input_names(self) -> tuple[str, ...]:
        """Input parameter names in stable dimension order."""
        return self._input_names or ()

    @property
    def output_names(self) -> tuple[str, ...]:
        """Output parameter names in first-observed order."""
        return tuple(self._output_names)

    @property
    def point_count(self) -> int:
        """Number of distinct input design points accumulated."""
        return len(self._points)

    def add_point(
        self,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
    ) -> DatasetBuilder:
        """Add outputs for one input point and return this builder.

        If *inputs* identifies an existing point, *outputs* is merged with the
        existing output dictionary. An overlapping output name raises
        :class:`OutputOverwriteError`, even when its proposed value is equal.
        """
        normalized_inputs = self._validate_inputs(inputs)
        normalized_outputs = self._validate_outputs(outputs)
        point_key = tuple(normalized_inputs[name] for name in self.input_names)
        existing = self._points.setdefault(point_key, {})
        overlap = sorted(existing.keys() & normalized_outputs.keys())
        if overlap:
            raise OutputOverwriteError(
                f"Outputs already exist for this input point and cannot be overwritten: {overlap}"
            )
        existing.update(normalized_outputs)
        for name in normalized_outputs:
            if name not in self._output_names:
                self._output_names.append(name)
        for name, value in normalized_inputs.items():
            if value not in self._levels[name]:
                self._levels[name].append(value)
        return self

    def to_dataset(self) -> xr.Dataset:
        """Materialize all accumulated points as an xarray Dataset.

        The independent coordinate levels define a Cartesian rectilinear grid.
        Unsupplied Cartesian combinations and point-specific outputs are
        represented by ``NaN``.
        """
        if not self._points:
            raise DatasetBuilderError("At least one design point is required.")
        if not self._output_names:
            raise DatasetBuilderError("At least one output parameter is required.")
        coordinates = {name: self._ordered_levels(self._levels[name]) for name in self.input_names}
        dimension_names = tuple(name for name in self.input_names if len(coordinates[name]) > 1)
        constant_names = tuple(name for name in self.input_names if len(coordinates[name]) == 1)
        shape = tuple(len(coordinates[name]) for name in dimension_names)
        indexes = {
            name: {value: index for index, value in enumerate(coordinates[name])}
            for name in dimension_names
        }
        variables = {name: np.full(shape, np.nan, dtype=float) for name in self._output_names}
        for point, outputs in self._points.items():
            point_values = dict(zip(self.input_names, point, strict=True))
            location = tuple(indexes[name][point_values[name]] for name in dimension_names)
            for name, value in outputs.items():
                variables[name][location] = value
        dataset = xr.Dataset(
            {name: (dimension_names, values) for name, values in variables.items()},
            coords={
                name: (name, np.asarray(values))
                for name, values in coordinates.items()
                if name in dimension_names
            },
            attrs=dict(self.attrs),
        )
        dataset = dataset.assign_coords({name: coordinates[name][0] for name in constant_names})
        for name, metadata in self.coordinate_attrs.items():
            if name in dataset.coords:
                dataset.coords[name].attrs.update(metadata)
        for name, metadata in self.variable_attrs.items():
            if name in dataset.data_vars:
                dataset[name].attrs.update(metadata)
        dataset.attrs["hyperslice_builder_point_count"] = self.point_count
        dataset.attrs["hyperslice_builder_constant_inputs"] = ", ".join(constant_names)
        return dataset

    def write(self, target: str | Path) -> Path:
        """Write NetCDF or Zarr based on *target* and return its path."""
        path = Path(target).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset = self.to_dataset()
        if path.suffix.lower() in {".nc", ".nc4", ".cdf", ".netcdf"}:
            dataset.to_netcdf(path)
        elif path.suffix.lower() == ".zarr":
            dataset.to_zarr(path, mode="w")
        else:
            raise DatasetBuilderError(
                f"Unsupported output format for '{path}'. Use NetCDF (.nc/.nc4) or Zarr (.zarr)."
            )
        return path

    def _validate_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        if not inputs:
            raise InvalidParameterError("Input parameters cannot be empty.")
        if any(not isinstance(name, str) or not name for name in inputs):
            raise InvalidParameterError("Input parameter names must be nonempty strings.")
        keys = tuple(inputs)
        if self._input_names is None:
            self._input_names = keys
            self._levels = {name: [] for name in keys}
        elif set(keys) != set(self._input_names):
            missing = sorted(set(self._input_names) - set(keys))
            unexpected = sorted(set(keys) - set(self._input_names))
            raise InputSchemaMismatchError(
                "Input parameter keys differ from the established schema. "
                f"Missing: {missing}; unexpected: {unexpected}."
            )
        normalized: dict[str, Any] = {}
        for name in self.input_names:
            value = self._scalar(inputs[name], parameter=f"Input '{name}'")
            try:
                hash(value)
            except TypeError as exc:
                raise InvalidParameterError(
                    f"Input '{name}' must be hashable; received {value!r}."
                ) from exc
            if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
                raise InvalidParameterError(f"Input '{name}' cannot be None or NaN.")
            normalized[name] = value
        return normalized

    def _validate_outputs(self, outputs: Mapping[str, Any]) -> dict[str, float]:
        if not outputs:
            raise InvalidParameterError("Output parameters cannot be empty.")
        normalized: dict[str, float] = {}
        for name, raw_value in outputs.items():
            if not isinstance(name, str) or not name:
                raise InvalidParameterError("Output parameter names must be nonempty strings.")
            value = self._scalar(raw_value, parameter=f"Output '{name}'")
            if isinstance(value, (str, bytes, complex)) or value is None:
                raise InvalidParameterError(
                    f"Output '{name}' must be a real numeric scalar; received {value!r}."
                )
            try:
                normalized[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise InvalidParameterError(
                    f"Output '{name}' must be a real numeric scalar; received {value!r}."
                ) from exc
        return normalized

    @staticmethod
    def _scalar(value: Any, *, parameter: str) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if not np.isscalar(value):
            raise InvalidParameterError(
                f"{parameter} must be scalar; received {type(value).__name__}."
            )
        return value

    @staticmethod
    def _ordered_levels(values: list[Any]) -> list[Any]:
        try:
            return sorted(values)
        except TypeError:
            return list(values)
