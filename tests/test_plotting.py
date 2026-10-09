from __future__ import annotations

import holoviews as hv
import numpy as np
import pytest
import xarray as xr

from hyperslice.exceptions import SliceError
from hyperslice.plotting import build_plot
from hyperslice.status import SliceStatus


def _planar_slice() -> tuple[xr.DataArray, xr.Dataset, SliceStatus]:
    x = np.array([0.0, 1.0, 3.0, 6.0])
    y = np.array([0.0, 2.0, 5.0])
    source_y, source_x = np.meshgrid(y, x, indexing="ij")
    data = xr.DataArray(
        source_x + source_y,
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name="response",
        attrs={"long_name": "Response", "units": "K"},
    )
    valid = xr.ones_like(data, dtype=bool)
    status = SliceStatus(
        mask_valid=valid,
        mask_invalid=xr.zeros_like(valid),
        mask_missing=xr.zeros_like(valid),
        labels={},
    )
    return data, xr.Dataset({"response": data}), status


def _contour_layers(plot: hv.Element) -> list[hv.Contours]:
    return [layer for layer in plot.traverse() if isinstance(layer, hv.Contours)]


def test_interpolated_contours_follow_the_linear_surface() -> None:
    data, dataset, status = _planar_slice()
    plot = build_plot(
        data,
        dataset=dataset,
        x_dim="x",
        y_dim="y",
        plot_type="heatmap",
        title="",
        status=status,
        show_samples=False,
        show_invalid=False,
        contours=True,
        contour_levels=5,
    )
    layers = _contour_layers(plot)
    assert layers, "the heatmap should carry a contour overlay"
    (isolines,) = layers
    assert isolines.label == "Interpolated contours"
    heights = isolines.dimension_values("response")
    levels = np.unique(heights[np.isfinite(heights)])  # NaN separates sub-paths
    assert 3 <= len(levels) <= 12
    assert levels.min() >= 0.0 and levels.max() <= 11.0
    # Every vertex of an isoline sits where the plane x + y equals that level,
    # which only holds when the lines are traced on the interpolated surface.
    for level in levels:
        line = isolines.select(response=level)
        vertex_heights = line.dimension_values("x") + line.dimension_values("y")
        vertex_heights = vertex_heights[np.isfinite(vertex_heights)]
        assert vertex_heights.size and np.allclose(vertex_heights, level, atol=1e-6)
    # Each height is written once, on its own line.
    (labels,) = [layer for layer in plot.traverse() if isinstance(layer, hv.Labels)]
    assert list(labels.dimension_values("text")) == [f"{level:g}" for level in levels]
    label_heights = labels.dimension_values("x") + labels.dimension_values("y")
    assert np.allclose(label_heights, levels, atol=1e-6)


def test_contours_are_opt_in() -> None:
    data, dataset, status = _planar_slice()
    common = dict(
        dataset=dataset,
        x_dim="x",
        y_dim="y",
        plot_type="image",
        title="",
        status=status,
        show_samples=True,
        show_invalid=True,
    )
    assert not _contour_layers(build_plot(data, **common))
    with_contours = build_plot(data, **common, contours=True)
    labels = {layer.label for layer in _contour_layers(with_contours)}
    assert labels == {"Interpolated contours"}


def test_contours_refuse_invalid_support() -> None:
    data, dataset, status = _planar_slice()
    status.mask_invalid[1, 1] = True
    with pytest.raises(SliceError, match="semantically invalid"):
        build_plot(
            data,
            dataset=dataset,
            x_dim="x",
            y_dim="y",
            plot_type="heatmap",
            title="",
            status=status,
            show_samples=False,
            show_invalid=False,
            contours=True,
        )
