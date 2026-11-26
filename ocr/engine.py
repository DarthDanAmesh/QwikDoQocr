"""Adapter around PaddleOCR 2.9.1.

PaddleOCR's API is stable but heavy: importing it pulls in paddlepaddle and
downloads models on first use. Keeping it behind this class means the UI,
exports, and tests never need it installed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from .geometry import scale_lines, sort_reading_order
from .models import Box, OcrLine, OcrResult, Point


class EngineUnavailable(RuntimeError):
    """PaddleOCR is not installed or failed to initialise."""


def _normalise_box(points: list[Point]) -> Box:
    """Paddle usually returns 4 corners; fall back to the bounding rectangle."""
    if len(points) == 4:
        return (points[0], points[1], points[2], points[3])
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return ((min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys)))


def _iter_entries(raw: Any):
    """Yield ``[box, (text, score)]`` entries from Paddle's nested result.

    ``ocr()`` returns ``[[line, ...]]`` (one list per image) but older builds
    and single-image calls return the flat ``[line, ...]``; flatten both.
    """
    pending = list(raw or [])
    while pending:
        item = pending.pop(0)
        if not isinstance(item, (list, tuple)):
            continue
        if (
            len(item) == 2
            and isinstance(item[1], (list, tuple))
            and len(item[1]) >= 2
            and isinstance(item[1][0], str)
        ):
            yield item[0], item[1]
        else:
            pending[0:0] = list(item)


def lines_from_raw(raw: Any) -> list[OcrLine]:
    """Convert a PaddleOCR result into model lines, keeping detection order."""
    lines: list[OcrLine] = []
    for box, payload in _iter_entries(raw):
        points = [(float(x), float(y)) for x, y in box]
        if not points:
            continue
        lines.append(OcrLine(str(payload[0]), float(payload[1]), _normalise_box(points)))
    return lines


def _prepare_image(path: Path, max_side: int | None) -> tuple[Any, int, int, float, float]:
    """Return (array_or_None, width, height, sx, sy).

    ``array_or_None`` is None when no downscale is needed, letting the engine
    read the file itself. When downscaled, sx/sy map OCR coordinates back to
    the original image size.
    """
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        longest = max(width, height)
        if not max_side or longest <= max_side:
            return None, width, height, 1.0, 1.0
        ratio = max_side / longest
        size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    import numpy as np

    # PaddleOCR expects BGR for numpy input; RGB -> BGR is a channel flip.
    array = np.asarray(resized)[:, :, ::-1].copy()
    return array, width, height, width / size[0], height / size[1]


class PaddleEngine:
    """Lazy PaddleOCR wrapper. ``run()`` is safe to call from a worker thread."""

    name = "paddleocr"

    def __init__(self, lang: str = "en", max_side: int | None = 2000) -> None:
        self.lang = lang
        self.max_side = max_side
        self._ocr = None

    def _ensure_engine(self):
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # ImportError, DLL load failure, missing numpy ABI
            raise EngineUnavailable(
                "PaddleOCR could not be imported. It needs Python 3.8-3.12 and "
                "paddlepaddle==2.6.2; this interpreter is "
                f"{sys.version.split()[0]}.\n\n{exc}"
            ) from exc
        try:
            self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        except Exception as exc:
            raise EngineUnavailable(f"PaddleOCR failed to initialise: {exc}") from exc
        return self._ocr

    def run(self, path: Path) -> OcrResult:
        path = Path(path)
        engine = self._ensure_engine()
        array, width, height, sx, sy = _prepare_image(path, self.max_side)

        started = time.perf_counter()
        raw = engine.ocr(array if array is not None else str(path), cls=True)
        elapsed = time.perf_counter() - started

        lines = lines_from_raw(raw)
        if sx != 1.0 or sy != 1.0:
            lines = scale_lines(lines, sx, sy)
        return OcrResult(sort_reading_order(lines), width, height, elapsed)
