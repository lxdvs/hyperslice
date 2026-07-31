"""Application-specific exceptions."""


class HyperSliceError(Exception):
    """Base exception for actionable HyperSlice failures."""


class DatasetLoadError(HyperSliceError):
    """A dataset could not be opened."""


class DatasetSchemaError(HyperSliceError):
    """A dataset cannot be explored safely."""


class SliceError(HyperSliceError):
    """A requested two-dimensional slice is invalid."""
