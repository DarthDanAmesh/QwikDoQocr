"""Write OCR lines to Markdown or XLSX without a pandas round trip."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .models import OcrLine

HEADERS = ("Box Coordinates", "Recognized Text", "Confidence")
LOW_CONFIDENCE = 0.85


def document_text(lines: Sequence[OcrLine]) -> str:
    """Plain text in reading order, one OCR line per line of output."""
    return "\n".join(line.text for line in lines)


def _coords(box) -> str:
    return ", ".join(f"({round(x)}, {round(y)})" for x, y in box)


def _summary(lines: Sequence[OcrLine], source: str | None) -> str:
    if not lines:
        return "No text detected."
    mean = sum(line.confidence for line in lines) / len(lines)
    low = sum(1 for line in lines if line.confidence < LOW_CONFIDENCE)
    parts = [f"{len(lines)} lines", f"mean confidence {mean:.3f}"]
    if low:
        parts.append(f"{low} below {LOW_CONFIDENCE:.2f}")
    if source:
        parts.append(f"source `{source}`")
    return " · ".join(parts)


def write_markdown(path: Path | str, lines: Sequence[OcrLine], source: str | None = None) -> None:
    path = Path(path)
    title = Path(source).stem if source else path.stem
    body = [
        f"# {title}",
        "",
        _summary(lines, source),
        "",
        document_text(lines),
        "",
        "## Lines",
        "",
        "| # | Recognized text | Confidence | Box (x, y) |",
        "|---|-----------------|------------|------------|",
    ]
    for index, line in enumerate(lines, 1):
        text = line.text.replace("|", "\\|")
        body.append(f"| {index} | {text} | {line.confidence:.3f} | ({_coords(line.box)}) |")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def write_xlsx(path: Path | str, lines: Sequence[OcrLine], source: str | None = None) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = Path(source).stem[:31] if source else "OCR"
    sheet.append(HEADERS)

    for line in lines:
        sheet.append([_coords(line.box), line.text, round(line.confidence, 4)])

    for cell in sheet[1]:
        cell.font = Font(bold=True)
    sheet.freeze_panes = "A2"
    sheet.column_dimensions["A"].width = 42
    sheet.column_dimensions["B"].width = 80
    sheet.column_dimensions["C"].width = 12

    for row in sheet.iter_rows(min_row=2, max_col=3):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=cell.column == 2)
        if row[2].value is not None and float(row[2].value) < LOW_CONFIDENCE:
            for cell in row:
                cell.font = Font(color="B45309")

    workbook.save(path)
