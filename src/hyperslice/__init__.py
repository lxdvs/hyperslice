"""HyperSlice public API."""

from hyperslice._version import __version__
from hyperslice.builder import (
    DatasetBuilder,
    DatasetBuilderError,
    InputSchemaMismatchError,
    InvalidParameterError,
    OutputOverwriteError,
)
from hyperslice.explorer import Explorer
from hyperslice.loading import load_dataset, load_points_json

__all__ = [
    "DatasetBuilder",
    "DatasetBuilderError",
    "Explorer",
    "InputSchemaMismatchError",
    "InvalidParameterError",
    "OutputOverwriteError",
    "__version__",
    "load_dataset",
    "load_points_json",
]
