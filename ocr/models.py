"""Core OCR data types shared by the engine, exporter, and UI."""

from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]
Box = tuple[Point, Point, Point, Point]


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: Box

    @property
    def top(self) -> float:
        return min(y for _, y in self.box)

    @property
    def bottom(self) -> float:
        return max(y for _, y in self.box)

    @property
    def left(self) -> float:
        return min(x for x, _ in self.box)

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class OcrResult:
    lines: list[OcrLine]
    width: int
    height: int
    elapsed: float = 0.0

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)
