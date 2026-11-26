from openpyxl import load_workbook

from ocr.export import HEADERS, document_text, write_markdown, write_xlsx
from ocr.models import OcrLine


def line(text, conf=0.9, left=10):
    return OcrLine(text, conf, ((left, 10), (left + 100, 10), (left + 100, 24), (left, 24)))


LINES = [
    line("This Agreement represents", 0.95),
    line("between the parties", 0.93),
    line("garbled | text", 0.62, left=40),
]


def test_document_text_is_one_line_per_ocr_line():
    assert document_text(LINES) == (
        "This Agreement represents\nbetween the parties\ngarbled | text"
    )
    assert document_text([]) == ""


def test_markdown_has_summary_text_and_table(tmp_path):
    path = tmp_path / "out.md"
    write_markdown(path, LINES, source="docu.jpg")
    text = path.read_text(encoding="utf-8")

    assert text.startswith("# docu")
    assert "3 lines · mean confidence 0.833 · 1 below 0.85 · source `docu.jpg`" in text
    assert "This Agreement represents\nbetween the parties\ngarbled | text" in text
    assert "| 3 | garbled \\| text | 0.620 |" in text


def test_xlsx_matches_the_documented_columns(tmp_path):
    path = tmp_path / "out.xlsx"
    write_xlsx(path, LINES, source="docu.jpg")

    sheet = load_workbook(path).active
    rows = list(sheet.iter_rows(values_only=True))

    assert rows[0] == HEADERS
    assert rows[1] == ("(10, 10), (110, 10), (110, 24), (10, 24)", "This Agreement represents", 0.95)
    assert rows[3][2] == 0.62
    assert sheet.freeze_panes == "A2"
    assert sheet.title == "docu"


def test_xlsx_marks_low_confidence_rows(tmp_path):
    path = tmp_path / "low.xlsx"
    write_xlsx(path, LINES)
    sheet = load_workbook(path).active
    assert sheet.cell(row=4, column=2).font.color.rgb.endswith("B45309")
    assert sheet.cell(row=2, column=2).font.bold is not True


def test_empty_result_still_writes_files(tmp_path):
    write_markdown(tmp_path / "empty.md", [])
    write_xlsx(tmp_path / "empty.xlsx", [])
    assert "No text detected." in (tmp_path / "empty.md").read_text(encoding="utf-8")
    assert list(load_workbook(tmp_path / "empty.xlsx").active.iter_rows(values_only=True)) == [HEADERS]
