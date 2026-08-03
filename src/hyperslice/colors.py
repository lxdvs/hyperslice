"""Shared colour ramps for every rendering surface."""

from __future__ import annotations

#: Anchor colours for the default ramp: blue at the low end, purple through the
#: middle, red at the high end.
BLUE_PURPLE_RED_ANCHORS: tuple[str, ...] = ("#0000ff", "#800080", "#ff0000")


def _channels(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def ramp(anchors: tuple[str, ...] = BLUE_PURPLE_RED_ANCHORS, steps: int = 256) -> list[str]:
    """Interpolate *anchors* into a smooth *steps*-colour palette."""
    if steps < 2 or len(anchors) < 2:
        return list(anchors)
    stops = [_channels(color) for color in anchors]
    spans = len(stops) - 1
    palette: list[str] = []
    for step in range(steps):
        position = step / (steps - 1) * spans
        index = min(int(position), spans - 1)
        weight = position - index
        start, end = stops[index], stops[index + 1]
        channels = (round(a + (b - a) * weight) for a, b in zip(start, end, strict=True))
        palette.append("#{:02x}{:02x}{:02x}".format(*channels))
    return palette


#: Default continuous palette used by the plots, the point cloud, and PNG export.
BLUE_PURPLE_RED: list[str] = ramp()
