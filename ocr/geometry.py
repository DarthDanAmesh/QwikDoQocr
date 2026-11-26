"""Geometry helpers: reading order and coordinate scaling."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import median

from .models import Box, OcrLine, Point


def sort_reading_order(lines: Iterable[OcrLine], tolerance: float | None = None) -> list[OcrLine]:
    """Order lines top-to-bottom, then left-to-right within a visual row.

    Detection order is not reading order: on a slightly skewed photo a line can
    arrive after one that sits below it. Grouping by vertical centre, then
    sorting each group by left edge, keeps multi-column rows intact while
    restoring document order.
    """
    items = list(lines)
    if len(items) < 2:
        return items

    heights = [line.height for line in items if line.height > 0]
    if tolerance is None:
        tolerance = max(2.0, (median(heights) if heights else 0.0) * 0.5)

    rows: list[dict] = []
    for line in sorted(items, key=lambda item: item.center_y):
        if rows and abs(line.center_y - rows[-1]["center"]) <= tolerance:
            row = rows[-1]
            row["items"].append(line)
            row["center"] = sum(item.center_y for item in row["items"]) / len(row["items"])
        else:
            rows.append({"center": line.center_y, "items": [line]})

    ordered: list[OcrLine] = []
    for row in rows:
        ordered.extend(sorted(row["items"], key=lambda item: item.left))
    return ordered


def scale_box(box: Box, sx: float, sy: float) -> Box:
    return (
        (box[0][0] * sx, box[0][1] * sy),
        (box[1][0] * sx, box[1][1] * sy),
        (box[2][0] * sx, box[2][1] * sy),
        (box[3][0] * sx, box[3][1] * sy),
    )


def scale_lines(lines: Sequence[OcrLine], sx: float, sy: float) -> list[OcrLine]:
    return [OcrLine(line.text, line.confidence, scale_box(line.box, sx, sy)) for line in lines]


def clamp_scale(scale: float, low: float = 0.1, high: float = 4.0) -> float:
    return max(low, min(high, scale))


def fit_scale(image_size: tuple[int, int], view_size: tuple[int, int], cap: float = 1.0) -> float:
    image_width, image_height = image_size
    view_width, view_height = view_size
    if image_width <= 0 or image_height <= 0 or view_width <= 1 or view_height <= 1:
        return 1.0
    return clamp_scale(min(view_width / image_width, view_height / image_height, cap))


def point_in_box(point: Point, box: Box) -> bool:
    x, y = point
    left = min(px for px, _ in box)
    right = max(px for px, _ in box)
    top = min(py for _, py in box)
    bottom = max(py for _, py in box)
    return left <= x <= right and top <= y <= bottom
