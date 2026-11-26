# QwikDoQocr

Desktop review app for OCR of document photos. PaddleOCR recognises the text;
the app is built around the part that actually costs time — checking and
correcting it. Every recognised line is an editable row tied to its box on the
image, and both exports come straight from the corrected text.

## Why the review step

The previous script (`ocrpy.py`, removed) ran OCR and wrote XLSX and Markdown
in one pass. Two problems showed up on its own sample (`docu.jpg`):

- **Wrong reading order.** Detection returns boxes in scan order, not document
  order. In section 11 the line at y=380 was written before the line at y=372,
  so the Markdown said "between them…" before "between the parties…".
- **No way to fix text, and scores do not flag mistakes.** All 32 lines scored
  at least 0.91 (mean 0.953) while several are visibly wrong:
  "11.Miscellancous", "conceming", "representationsexpress", "lata" → "data".
  A reviewer has to compare against the image, which is what the UI is for.

## What it does

- Opens PNG/JPG/TIFF/BMP/WebP images; OCR runs on a worker thread with a
  progress bar, so the window stays responsive.
- Draws every recognised box over the image. Click a box to focus its row;
  focus a row to highlight its box. The selected row scrolls into view.
- Rows are editable in place. Edits update the model immediately; exports and
  copy use the corrected text.
- Reading order is rebuilt from the boxes: group by vertical centre with a
  tolerance of half the median line height, then sort each row left to right.
- `max_side=2000` downscales large photos before recognition (configurable,
  `0` disables) and scales the boxes back.
- Low-confidence rows (< 0.85) are amber in the list and in the spreadsheet, so
  they are checked first.

## Install

Python **3.8–3.12**. `paddlepaddle 2.6.2` has no wheels for 3.13+, so the
recognition engine cannot be installed on newer interpreters — the UI, exports
and tests still run.

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first OCR run downloads the detection, recognition and classification
models (about 15 MB) into the PaddleOCR cache.

## Run

```bat
python ocr_app.py                          :: empty window
python ocr_app.py path\to\page.jpg         :: window with the image loaded
python ocr_app.py page.jpg --headless      :: no window; writes page.xlsx and page.md
```

`--headless` options: `--outdir DIR`, `--max-side PIXELS` (default 2000, 0 =
never downscale), `--lang en|ch|fr|de|…`. Exit codes: `0` exported, `1` no text
found, `2` PaddleOCR unavailable.

### Shortcuts

| Keys | Action |
|------|--------|
| Ctrl+O | Open image |
| F5 / Ctrl+Enter | Run OCR |
| Ctrl+S | Export XLSX |
| Ctrl+Shift+S | Export Markdown |
| Ctrl+C | Copy all text (ignored while editing a row, so normal copy works) |
| Ctrl+0 | Fit to window |
| Ctrl+= / Ctrl+- | Zoom in / out |
| Click | Select the line under the pointer |

## Exports

**XLSX** keeps the old column layout — `Box Coordinates`, `Recognized Text`,
`Confidence` — with a frozen header row and the source sheet named after the
file. Boxes are formatted as `(x, y)` corner pairs; the old format was a Python
list repr, so anything that parsed that string needs updating.

**Markdown** contains the document text (one line per OCR line), a summary
line, and a table of line number, text, confidence and box.

Neither export reflows paragraphs: line breaks follow the page.

## Layout

```
ocr_app.py        entry point: GUI or headless CLI
ocr/models.py     OcrLine / OcrResult dataclasses
ocr/geometry.py   reading order, zoom scaling, hit-testing
ocr/engine.py     PaddleOCR 2.9.1 adapter (lazy import, downscale, box scaling)
ocr/export.py     Markdown and XLSX writers
ocr/ui.py         Tkinter window: canvas overlay, editable rows, exports
tests/            pytest suite (runs without PaddleOCR)
```

PaddleOCR is imported lazily and only inside `PaddleEngine`, so the UI, exports
and tests never need it installed. `EngineUnavailable` carries the Python
version hint shown in the error dialog.

## Tests

```bat
pip install pytest pillow openpyxl numpy
python -m pytest tests -q
```

41 tests: reading order against the real box coordinates from `docu.jpg`,
coordinate scaling, splitting Paddle's nested result (wrapped page, flat list,
empty, malformed, 5-point box), export contents/round trip, and a withdrawn-Tk
smoke test that loads an image with a fake engine, checks rows and overlays,
edits a row, exports it, and checks the clipboard.

What they do **not** prove: recognition quality. The engine is faked in every
test, and real PaddleOCR inference was not measured here because paddlepaddle
cannot be installed on the interpreter used (3.13). The GUI was checked by
driving it with real boxes from a previous real run and screenshotting each
state; exact colours and spacing were reviewed visually, not pixel-asserted.
