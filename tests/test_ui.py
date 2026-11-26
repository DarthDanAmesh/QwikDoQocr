"""GUI tests run a withdrawn Tk window with a fake engine.

They prove wiring: OCR results become rows and overlays, edits reach the
model, selection is shared, and exports use the edited text. They do not
prove that the widgets look right on screen, which needs a human or a
screenshot.
"""

from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter")

from PIL import Image  # noqa: E402

from ocr.models import OcrLine, OcrResult  # noqa: E402
from ocr.ui import OcrApp  # noqa: E402


def line(text, top, conf=0.9):
    return OcrLine(text, conf, ((10, top), (200, top), (200, top + 14), (10, top + 14)))


def sample_lines():
    """Fresh line objects per test; editing them mutates the model in place."""
    return [
        line("This Agreement represents", 358, 0.95),
        line("between the parties", 372, 0.93),
        line("between them.", 380, 0.62),
    ]


class FakeEngine:
    def __init__(self, lines):
        self.lines = lines
        self.calls = 0

    def run(self, path):
        self.calls += 1
        return OcrResult(self.lines, 448, 599, 0.42)


class Click:
    """The two attributes ImageCanvas._on_click reads from a Tk event."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


@pytest.fixture(scope="module")
def root():
    try:
        window = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - headless CI
        pytest.skip(f"Tk cannot open a display: {exc}")
    window.withdraw()
    try:
        yield window
    finally:
        try:
            window.destroy()
        except tk.TclError:
            pass


@pytest.fixture(autouse=True)
def clear_window(root):
    """One root for the module; each test starts with an empty window.

    Creating and destroying a fresh Tk root per test intermittently fails in
    this Anaconda build while the previous root's PhotoImages are collected.
    """
    yield
    for child in root.winfo_children():
        child.destroy()
    root.update_idletasks()


@pytest.fixture
def page(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (448, 599), "white").save(path)
    return path


def make_app(root, lines=None):
    app = OcrApp(engine=FakeEngine(lines if lines is not None else sample_lines()), root=root, interactive=False)
    root.update_idletasks()
    return app


def test_ocr_populates_rows_overlays_and_stats(root, page):
    app = make_app(root)
    assert app.load_image(page)
    assert app.run_ocr(blocking=True)
    root.update_idletasks()

    assert len(app.list._rows) == 3
    assert [row["entry"].get() for row in app.list._rows] == [
        "This Agreement represents", "between the parties", "between them.",
    ]
    assert len(app.canvas.canvas.find_withtag("overlay")) == 6  # polygon + number per line
    assert "3 lines" in app.stats_label.cget("text")
    assert "Low confidence rows: 3" in app.status_label.cget("text")
    assert app.file_label.cget("text") == "page.png"


def test_edit_updates_model_and_dirty_title(root, page):
    app = make_app(root)
    app.load_image(page)
    app.run_ocr(blocking=True)

    entry = app.list._rows[1]["entry"]
    entry.delete(0, "end")
    entry.insert(0, "between the parties and successors")
    root.update_idletasks()

    assert app.list.lines[1].text == "between the parties and successors"
    assert app._dirty
    assert app.root.title().startswith("page.png *")


def test_canvas_click_selects_row_and_highlights_box(root, page):
    app = make_app(root)
    app.load_image(page)
    app.run_ocr(blocking=True)
    root.update_idletasks()

    scale = app.canvas.scale
    offset_x, offset_y = app.canvas._offset
    app.canvas._on_click(Click(offset_x + 20 * scale, offset_y + 388 * scale))
    root.update_idletasks()

    assert app.list._selected == 2
    assert app.canvas._selected == 2
    assert app.list._rows[2]["frame"].cget("background") != app.list._rows[0]["frame"].cget("background")


def test_clicking_empty_space_clears_selection(root, page):
    app = make_app(root)
    app.load_image(page)
    app.run_ocr(blocking=True)
    app._select(0)
    root.update_idletasks()

    app.canvas._on_click(Click(1, 1))
    root.update_idletasks()

    assert app.list._selected is None
    assert app.canvas._selected is None


def test_selection_survives_zoom_and_refit(root, page):
    app = make_app(root)
    app.load_image(page)
    app.run_ocr(blocking=True)
    app._select(1)

    app.zoom(2.0)
    assert app.zoom_label.cget("text") == "200%"
    assert app.list._selected == 1

    app.fit_view()
    assert app.canvas._selected == 1


def test_export_writes_edited_text(root, page, tmp_path, monkeypatch):
    import ocr.ui as ui

    app = make_app(root)
    app.load_image(page)
    app.run_ocr(blocking=True)

    entry = app.list._rows[2]["entry"]
    entry.delete(0, "end")
    entry.insert(0, "between them. Corrected.")

    target = tmp_path / "review.xlsx"
    monkeypatch.setattr(ui.filedialog, "asksaveasfilename", lambda **_kwargs: str(target))
    app.export("xlsx")

    from openpyxl import load_workbook

    rows = list(load_workbook(target).active.iter_rows(values_only=True))
    assert rows[3][1] == "between them. Corrected."
    assert not app._dirty
    assert target.name in app.status_label.cget("text")


def test_markdown_export_uses_reading_order(root, page, tmp_path, monkeypatch):
    import ocr.ui as ui

    app = make_app(root)
    app.load_image(page)
    app.run_ocr(blocking=True)

    target = tmp_path / "review.md"
    monkeypatch.setattr(ui.filedialog, "asksaveasfilename", lambda **_kwargs: str(target))
    app.export("md")

    text = target.read_text(encoding="utf-8")
    assert text.index("This Agreement") < text.index("between the parties") < text.index("between them.")


def test_copy_text_puts_document_on_clipboard(root, page):
    app = make_app(root)
    app.load_image(page)
    app.run_ocr(blocking=True)

    app.copy_text()
    assert root.clipboard_get() == "This Agreement represents\nbetween the parties\nbetween them."


def test_empty_result_disables_exports_and_says_so(root, page):
    app = make_app(root, lines=[])
    app.load_image(page)
    assert app.run_ocr(blocking=True)
    root.update_idletasks()

    assert app.list._rows == []
    assert app.stats_label.cget("text") == "0 lines"
    assert "No text detected" in app.status_label.cget("text")
    assert app.xlsx_btn.instate(["disabled"])
    assert app.copy_btn.instate(["disabled"])


def test_actions_disabled_before_image_or_ocr(root, page):
    app = make_app(root)
    assert app.run_btn.instate(["disabled"])
    assert app.zoom_in_btn.instate(["disabled"])
    assert app.fit_btn.instate(["disabled"])

    app.load_image(page)
    assert app.run_btn.instate(["!disabled"])
    assert app.zoom_in_btn.instate(["!disabled"])
    assert app.copy_btn.instate(["disabled"])
    assert app.export("xlsx") is None  # no dialog, no crash, status explains
    assert "run OCR first" in app.status_label.cget("text")


def test_bad_image_reports_instead_of_crashing(root, tmp_path):
    app = make_app(root)
    assert app.load_image(tmp_path / "missing.png") is False
    assert "Cannot open image" in app.status_label.cget("text")
