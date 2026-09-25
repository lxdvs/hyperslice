# HyperSlice

HyperSlice is an engineering-focused interactive explorer for tagged, rectilinear,
N-dimensional scientific datasets stored as xarray-compatible NetCDF or Zarr data.
Choose an output, put any two of its dimensions on the plot, and select or interpolate
all remaining coordinates.

> Screenshot placeholder: the MVP UI has a compact scrollable control column and a
> large metadata-aware HoloViews/Bokeh plot with status and sample overlays.

## Quickstart

```bash
git clone <repository-url>
cd hyperslice
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python examples/create_synthetic_dataset.py
hyperslice examples/data/synthetic_sweep.nc --show
```

Zarr stores work the same way:

```bash
hyperslice examples/data/synthetic_sweep.zarr --variable peak_temperature \
  --x drum_angle --y fuel_temperature --status-variable status --show
```

Use `--address`, `--port`, and `--no-browser` for server control.

## Python and notebook use

```python
import xarray as xr
from hyperslice import Explorer

ds = xr.open_dataset("examples/data/synthetic_sweep.nc")
explorer = Explorer(ds, status_variable="status")
explorer  # embeddable Panel view in a notebook
explorer.show()
```

Application state is reproducible:

```python
state = explorer.get_state()
explorer.set_state(state)
```

## Building datasets incrementally

`DatasetBuilder` creates xarray-compatible rectilinear datasets from individual
design-point results:

```python
from hyperslice import DatasetBuilder

builder = DatasetBuilder(
    coordinate_attrs={
        "temperature": {"long_name": "Temperature", "units": "K"},
        "configuration": {"long_name": "Design configuration"},
    },
    variable_attrs={
        "efficiency": {"long_name": "Efficiency", "units": "1"},
        "margin": {"long_name": "Safety margin", "units": "MPa"},
    },
    attrs={"title": "Parameter sweep"},
)

point = {"temperature": 600.0, "configuration": "baseline"}
builder.add_point(point, {"efficiency": 0.81})
builder.add_point(point, {"margin": 2.4})  # additive for the same point

builder.add_point(
    {"temperature": 750.0, "configuration": "reinforced"},
    {"efficiency": 0.84, "margin": 3.1},
)

dataset = builder.to_dataset()
builder.write("sweep.nc")
builder.write("sweep.zarr")
```

Every input dictionary must contain exactly the same keys. Reusing an input
point is allowed only to add new output names; attempting to overwrite an
output already stored at that point raises `OutputOverwriteError`. The
Cartesian product of observed coordinate levels becomes the dataset grid, and
unsupplied combinations or point-specific outputs remain `NaN`. Inputs with
one identical value across every design point are removed from the dimensional
grid and retained as scalar coordinates, preserving their provenance without
creating singleton controls or axes.

## Supported data model

Every dimension must have an independent one-dimensional coordinate. Coordinates may
be numeric, datetime-like, or categorical and numeric spacing may be irregular. A
plottable response is currently a numeric variable with at least two dimensions.

```python
xr.Dataset(
    {
        "response": (
            ("temperature", "pressure", "material"),
            response_values,
            {"long_name": "Response", "units": "MPa"},
        ),
        "status": (
            ("temperature", "pressure", "material"),
            status_values,
            {
                "flag_values": [0, 1, 2],
                "flag_meanings": "valid nonconverged physically_invalid",
            },
        ),
    },
    coords={
        "temperature": ("temperature", [300, 450, 700], {"units": "K"}),
        "pressure": ("pressure", [1.0, 2.5, 8.0], {"units": "MPa"}),
        "material": ["A", "B"],
    },
)
```

HyperSlice recognizes CF-style integer flags, Boolean validity arrays, NaN, `_FillValue`,
and `missing_value`. Missing output and semantically invalid output remain distinct.
Linear interpolation masks invalid input first, does not extrapolate, and leaves output
missing when required support is incomplete.

## Grid terminology

- A **uniform grid** has equally spaced coordinate levels.
- A **rectilinear grid** has independent coordinate arrays; equal spacing is not required.
- A **full-factorial sweep** samples the Cartesian product of all coordinate levels.
- An **incomplete rectilinear dataset** retains rectilinear coordinates but lacks some
  Cartesian samples.
- **Scattered data** is not representable as a complete tensor product. Tensor-product
  interpolation needs complete local support; incomplete data may require masking or a
  future scattered-data method.

## Features

- Exact, nearest, and strict linear selection of fixed dimensions
- Heatmap/image, filled contour, and contour-line views with hover and colorbar
- Units and long names on axes and values
- Valid, invalid, and missing sample overlays and counts
- Filter view opening on the most interesting fields: outputs before inputs,
  ranked by how many distinct values each takes
- Design-point selection linked across the filter and slicer views, clearing any
  filter that would hide the selected point
- A local relative-sensitivity table of every output against every input at the
  selected design point: dimensionless elasticities `(x / y) dy/dx`, differenced
  across neighbouring grid samples
- Long-form CSV and metadata-preserving NetCDF downloads
- JSON-compatible application state
- Lazy xarray opening and Dask-compatible named operations

PNG export is available in the UI and through `hyperslice.export.save_png`. It uses
Bokeh's browser renderer when available and falls back to a headless Matplotlib render.

## Development

```bash
pip install -e ".[dev]"
ruff format .
ruff check .
mypy src/hyperslice
pytest
```

The synthetic generator writes deterministic NetCDF and Zarr examples under
`examples/data/`; generated binary data is intentionally gitignored.

## Current limitations and roadmap

The MVP focuses on two-dimensional slices of local datasets. Datetime interpolation is
delegated to xarray and categorical coordinates are exact-selected in linear mode.
The headless PNG fallback preserves the primary plot, title, labels, and colorbar but
does not yet reproduce every interactive overlay. Cubic and scattered interpolation are
deliberately not offered: they need explicit capability checks and valid local support.

Next priorities are per-status-category legends and counts, provenance for interpolated
support, and Dask performance tuning.
Longer-term directions include linked profile/surface views, comparisons, animation,
uncertainty and sensitivity maps, remote Zarr, workspaces, and plugin-based interpolation.
