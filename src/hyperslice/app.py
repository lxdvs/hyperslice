"""Panel application factory."""

from __future__ import annotations

from hyperslice.explorer import Explorer
from hyperslice.loading import DatasetSource


def create_app(source: DatasetSource, **kwargs: object) -> Explorer:
    """Construct an Explorer for embedding or serving."""
    return Explorer(source, **kwargs)
