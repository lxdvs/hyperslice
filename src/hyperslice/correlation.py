"""Relative sensitivity of every output to every input over filtered samples.

The samples inside the filters are reduced, for each output and input, to a
*mean response profile*: the output averaged at each value the input takes.
Averaging cancels the scatter the other inputs add, leaving one curve of the
output against this input alone. Inputs taking many distinct values, as in an
irregular design, are first grouped into bins of equal sample count.

A least-squares line through the profile gives:

* the *relative sensitivity* ``S = b * mean(x) / mean(y)``, the line's slope
  scaled like the slicer's local sensitivities: the fractional change in the
  output per fractional change in the input, averaged over the filtered range
  rather than taken at one design point.
* the *nonlinearity* ``1 - R²`` of the line on the profile: the share of the
  profile's variation the line misses, 0 for a straight profile and near 1
  for one that bends back on itself.
* the *bend*, the sign of a parabola's squared term through the profile:
  positive for a profile that curves upward, negative for one that curves down.

Every figure depends only on the samples passing the filters, so narrowing a
filter asks how the outputs respond over that sub-range.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd
from bokeh.palettes import RdBu

from hyperslice.schema import DatasetSchema
from hyperslice.sensitivity import UNDEFINED, input_dimensions, output_order

#: Diverging bands from -1 (dark blue) through 0 (near white) to 1 (dark red),
#: matching the sign convention of the sensitivity bars.
CORRELATION_PALETTE: list[str] = list(RdBu[11])
#: Fill for a cell the filtered samples cannot support.
UNDEFINED_COLOR = "#f3f5f7"

#: Most distinct input values profiled one by one. A denser input, typically
#: from an irregular design, is grouped into :data:`PROFILE_BINS` instead,
#: since a level holding one sample averages nothing away.
PROFILE_LEVEL_LIMIT = 20
#: Bins of equal sample count for an input past :data:`PROFILE_LEVEL_LIMIT`.
PROFILE_BINS = 10

#: Profile spread, relative to its magnitude, below which the profile counts
#: as flat, so rounding noise in a constant output never reads as curvature.
NEGLIGIBLE_SPREAD = 1e-12

#: Relative sensitivity below which a straight profile counts as flat. Far
#: below any response worth reading, far above the rounding left in a profile
#: of an output that ignores the input.
NEGLIGIBLE_SENSITIVITY = 1e-12

#: Marks for a profile curving upward and downward (the union and
#: intersection signs, which read as cup and cap).
BEND_UP = "\u222a"
BEND_DOWN = "\u2229"

#: Nonlinearity below which no bend glyph is drawn: the profile is near enough
#: straight that the sign of its curvature is not worth reading.
BEND_THRESHOLD = 0.05


@dataclass(frozen=True)
class ProfileFit:
    """Figures for one output against one input; NaN where undefined."""

    sensitivity: float
    nonlinearity: float
    bend: float
    correlation: float
    levels: int


UNDEFINED_FIT = ProfileFit(float("nan"), float("nan"), float("nan"), float("nan"), 0)


def mean_profile(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of *y* at each value of *x*, or in bins of *x* when it is dense.

    Expects finite pairs. Returns the profile's positions (the mean *x* of each
    level or bin) and means, in increasing order of position.
    """
    levels, inverse = np.unique(x, return_inverse=True)
    if levels.size > PROFILE_LEVEL_LIMIT:
        edges = np.quantile(x, np.linspace(0.0, 1.0, PROFILE_BINS + 1))
        inverse = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, PROFILE_BINS - 1)
    counts = np.bincount(inverse)
    occupied = counts > 0
    positions = np.bincount(inverse, weights=x)[occupied] / counts[occupied]
    means = np.bincount(inverse, weights=y)[occupied] / counts[occupied]
    return positions, means


def fit_profile(x: np.ndarray, y: np.ndarray) -> ProfileFit:
    """Fit a line through the mean response profile of *y* against *x*.

    Pairs with a missing value on either side are dropped. Everything is NaN
    when the profile has fewer than two levels or the output averages to zero,
    where no fraction of it exists to compare against. The nonlinearity and
    bend are also NaN for a two-level profile, which any line fits exactly.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return UNDEFINED_FIT
    positions, means = mean_profile(x, y)
    y_mean = float(y.mean())
    if positions.size < 2 or y_mean == 0:
        return UNDEFINED_FIT
    x_spread = float(positions.std())
    # Centred and scaled so the squared term stays well conditioned.
    u = (positions - positions.mean()) / x_spread
    centred = means - means.mean()
    slope = float(np.mean(u * centred)) / x_spread
    sensitivity = slope * float(x.mean()) / y_mean
    y_spread = float(y.std())
    correlation = (
        float(np.clip(np.mean((x - x.mean()) * (y - y_mean)) / (x.std() * y_spread), -1, 1))
        if y_spread > 0
        else float("nan")
    )
    if positions.size < 3:
        return ProfileFit(sensitivity, float("nan"), float("nan"), correlation, positions.size)
    variation = float(np.sum(centred**2))
    if variation <= NEGLIGIBLE_SPREAD * max(float(np.sum(means**2)), np.finfo(float).tiny):
        return ProfileFit(sensitivity, 0.0, 0.0, correlation, positions.size)
    linear = slope * x_spread * u
    nonlinearity = float(np.clip(np.sum((centred - linear) ** 2) / variation, 0.0, 1.0))
    coefficients, *_ = np.linalg.lstsq(
        np.column_stack([np.ones_like(u), u, u**2]), means, rcond=None
    )
    return ProfileFit(
        sensitivity, nonlinearity, float(np.sign(coefficients[2])), correlation, positions.size
    )


@dataclass(frozen=True)
class ProfileFrames:
    """One frame per figure, outputs as rows and inputs as columns."""

    sensitivity: pd.DataFrame
    nonlinearity: pd.DataFrame
    bend: pd.DataFrame
    correlation: pd.DataFrame
    levels: pd.DataFrame


def profile_frames(samples: pd.DataFrame, schema: DatasetSchema) -> ProfileFrames:
    """Profile fits of every output (rows) against every input (columns).

    Only the outputs and inputs present in *samples* appear: numeric, swept
    inputs, and numeric outputs ordered from the most distinct values to the
    fewest.
    """
    inputs = [name for name in input_dimensions(schema) if name in samples.columns]
    outputs = [
        name
        for name in output_order(schema)
        if name in samples.columns and samples[name].dtype.kind in "iufc"
    ]
    fits = [
        [fit_profile(samples[dim].to_numpy(), samples[output].to_numpy()) for dim in inputs]
        for output in outputs
    ]

    def frame(field: str) -> pd.DataFrame:
        values = [[getattr(fit, field) for fit in row] for row in fits]
        return pd.DataFrame(values, index=outputs, columns=inputs, dtype=float)

    return ProfileFrames(
        sensitivity=frame("sensitivity"),
        nonlinearity=frame("nonlinearity"),
        bend=frame("bend"),
        correlation=frame("correlation"),
        levels=frame("levels"),
    )


def row_fractions(row: pd.Series) -> pd.Series:
    """Each sensitivity relative to the largest finite magnitude in its *row*.

    Sensitivities are unbounded, so colour compares inputs within one output:
    the strongest driver of each output takes the darkest band.
    """
    values = row.astype(float)
    largest = float(values.abs().max()) if values.notna().any() else 0.0
    if not np.isfinite(largest) or largest <= 0:
        return values * 0.0
    return values / largest


def informative(frames: ProfileFrames) -> pd.DataFrame:
    """Cells that say something: a defined sensitivity that is nonzero or curved.

    A symmetric profile has no net slope yet bends strongly, so a cell with
    zero sensitivity still counts when its nonlinearity does.
    """
    sensitivity = frames.sensitivity.abs()
    curved = frames.nonlinearity.fillna(0.0) > 0
    return sensitivity.notna() & ((sensitivity > NEGLIGIBLE_SENSITIVITY) | curved)


#: Typographic minus and times for equations (escaped, as they read like ASCII).
MINUS_SIGN = "\u2212"
TIMES_SIGN = "\u00d7"


@dataclass(frozen=True)
class LineFit:
    """Least-squares line ``y = slope * x + intercept`` through *count* points."""

    slope: float
    intercept: float
    r_squared: float
    count: int

    def equation(self) -> str:
        """The line as ``y = a·x + b``, with its fit quality."""
        sign = MINUS_SIGN if self.intercept < 0 else "+"
        quality = f"R² = {self.r_squared:.3f}" if np.isfinite(self.r_squared) else "R² —"
        return (
            f"y = {self.slope:.4g}·x {sign} {abs(self.intercept):.4g} "
            f"({quality}, n = {self.count:,})"
        )


def best_fit(x: np.ndarray, y: np.ndarray) -> LineFit | None:
    """Least-squares line through the points with a value on both axes.

    None when either side is not numeric, fewer than two such points remain,
    or *x* does not vary, so no single line is determined. R² is NaN when *y*
    does not vary, since there is no variation for the line to explain.
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        return None
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or float(x.std()) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    total = float(np.sum((y - y.mean()) ** 2))
    residual = float(np.sum((y - (slope * x + intercept)) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else float("nan")
    return LineFit(float(slope), float(intercept), r_squared, int(x.size))


def band_color(value: float) -> str:
    """Palette band for a fraction in [-1, 1]."""
    last = len(CORRELATION_PALETTE) - 1
    index = round((float(value) + 1.0) / 2.0 * last)
    return CORRELATION_PALETTE[min(max(index, 0), last)]


def _dark(fraction: float) -> bool:
    """Whether a fraction's band is dark enough to need light text."""
    return abs(float(fraction)) >= 0.7


def bend_glyph(nonlinearity: float, bend: float) -> str:
    """Cup or cap for a profile curved enough to read, otherwise nothing."""
    if not np.isfinite(nonlinearity) or nonlinearity < BEND_THRESHOLD or not bend:
        return ""
    return f" {BEND_UP}" if bend > 0 else f" {BEND_DOWN}"


CORRELATION_STYLES = """
.hs-corr-wrap {
  overflow-x: auto;
}
.hs-corr {
  border-collapse: separate;
  border-spacing: 2px;
  font-size: 12px;
}
.hs-corr th {
  font-weight: 600;
  color: #33475b;
}
.hs-corr thead th {
  max-width: 120px;
  padding: 2px 4px;
  vertical-align: bottom;
  text-align: center;
  white-space: normal;
  overflow-wrap: anywhere;
}
.hs-corr tbody th {
  padding: 0 8px 0 0;
  text-align: right;
  white-space: nowrap;
}
.hs-corr td {
  min-width: 78px;
  height: 38px;
  padding: 2px 6px;
  text-align: center;
  border-radius: 3px;
  color: #1f2d3a;
}
.hs-corr td[data-output] {
  cursor: pointer;
}
.hs-corr td[data-output]:hover {
  outline: 2px solid #536878;
  outline-offset: -2px;
}
.hs-corr td.selected {
  outline: 3px solid #17324d;
  outline-offset: -3px;
}
.hs-corr td.dark {
  color: #ffffff;
}
.hs-corr td.undefined {
  color: #8a98a6;
}
.hs-corr-slope {
  display: block;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
  font-weight: 650;
}
.hs-corr-nonlinear {
  display: block;
  font-size: 10.5px;
  opacity: 0.85;
}
.hs-corr-legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
  font-size: 11px;
  color: #536878;
}
.hs-corr-hidden {
  margin-top: 6px;
  font-size: 11px;
  color: #8a98a6;
}
.hs-corr-legend-bands {
  display: flex;
}
.hs-corr-legend-bands span {
  width: 22px;
  height: 12px;
}
"""


def _format(value: float, pattern: str) -> str:
    return format(value, pattern) if np.isfinite(value) else UNDEFINED


def _cell(
    frames: ProfileFrames, output: str, dim: str, fraction: float, label: str, selected: bool
) -> str:
    # Data attributes name the pair a click on the cell asks to plot.
    target = f'data-output="{escape(output)}" data-input="{escape(dim)}"'
    picked = " selected" if selected else ""
    sensitivity = float(frames.sensitivity.at[output, dim])
    if not np.isfinite(sensitivity):
        return (
            f'<td class="undefined{picked}" {target} style="background: {UNDEFINED_COLOR}" '
            f'title="{label}: undefined">{UNDEFINED}</td>'
        )
    nonlinearity = float(frames.nonlinearity.at[output, dim])
    curve = _format(nonlinearity, ".0%")
    glyph = bend_glyph(nonlinearity, float(frames.bend.at[output, dim]))
    correlation = _format(float(frames.correlation.at[output, dim]), "+.2f")
    levels = int(frames.levels.at[output, dim])
    classes = ("dark" if _dark(fraction) else "") + picked
    return (
        f'<td class="{classes.strip()}" {target} style="background: {band_color(fraction)}" '
        f'title="{label}: sensitivity {sensitivity:+.4g}, nonlinearity {curve}{glyph}, '
        f'r {correlation}, {levels} profile levels · click to plot">'
        f'<span class="hs-corr-slope">{sensitivity:+.3g}</span>'
        f'<span class="hs-corr-nonlinear">NL {curve}{glyph}</span>'
        "</td>"
    )


def render_matrix(
    frames: ProfileFrames,
    input_labels: dict[str, str],
    output_labels: dict[str, str],
    selected: tuple[str, str] | None = None,
) -> str:
    """HTML table of sensitivity and nonlinearity cells with a colour legend beneath.

    Outputs and inputs whose every cell is undefined or flat are left out, and
    named in a note beneath, so the table holds only responses worth reading.
    The cell for *selected*, an ``(output, input)`` pair, is outlined.
    """
    shown = informative(frames)
    kept_rows = [str(name) for name in shown.index if shown.loc[name].any()]
    columns = [str(name) for name in shown.columns if shown[name].any()]
    hidden_rows = [str(name) for name in shown.index if str(name) not in kept_rows]
    hidden_columns = [str(name) for name in shown.columns if str(name) not in columns]
    notes = [
        f"{kind}: {', '.join(escape(labels.get(name, name)) for name in names)}"
        for kind, names, labels in (
            ("Outputs", hidden_rows, output_labels),
            ("Inputs", hidden_columns, input_labels),
        )
        if names
    ]
    hidden = (
        '<div class="hs-corr-hidden">Hidden, with no defined or nonzero sensitivity within '
        f"the filters: {'; '.join(notes)}</div>"
        if notes
        else ""
    )
    if not kept_rows:
        return (
            '<div class="hs-corr-hidden">No output responds to any input within the filters.'
            f"</div>{hidden}"
        )
    header = "".join(f"<th>{escape(input_labels.get(name, name))}</th>" for name in columns)
    rows = []
    for output in kept_rows:
        output_label = escape(output_labels.get(output, output))
        fractions = row_fractions(frames.sensitivity.loc[output])
        cells = "".join(
            _cell(
                frames,
                output,
                dim,
                float(fractions[dim]),
                f"{output_label} vs {escape(input_labels.get(dim, dim))}",
                selected == (output, dim),
            )
            for dim in columns
        )
        rows.append(f"<tr><th>{output_label}</th>{cells}</tr>")
    bands = "".join(f'<span style="background: {color}"></span>' for color in CORRELATION_PALETTE)
    legend = (
        '<div class="hs-corr-legend">'
        f'<span>&minus;max</span><span class="hs-corr-legend-bands">{bands}</span><span>+max</span>'
        "<span>&nbsp;relative sensitivity of the mean response profile, coloured against "
        "each output's largest · NL: share of the profile a line misses, "
        f"{BEND_UP}/{BEND_DOWN} its bend</span>"
        "</div>"
    )
    return (
        '<div class="hs-corr-wrap"><table class="hs-corr">'
        f"<thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table></div>{legend}{hidden}"
    )
