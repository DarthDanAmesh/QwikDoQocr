"""Entry point: open the review window, or run OCR headless for scripting.

    python ocr_app.py                      # empty window
    python ocr_app.py docu.jpg             # window with the image loaded
    python ocr_app.py docu.jpg --headless  # write docu.xlsx and docu.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ocr.engine import EngineUnavailable, PaddleEngine
from ocr.export import write_markdown, write_xlsx


def enable_dpi_awareness() -> None:
    """Crisp text and images on scaled Windows displays."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def run_headless(image: Path, outdir: Path | None, max_side: int, lang: str) -> int:
    target_dir = outdir or image.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    engine = PaddleEngine(lang=lang, max_side=max_side or None)
    try:
        result = engine.run(image)
    except EngineUnavailable as exc:
        print(exc, file=sys.stderr)
        return 2

    if not result.lines:
        print(f"No text detected in {image.name}.")
        return 1

    xlsx_path = target_dir / f"{image.stem}.xlsx"
    md_path = target_dir / f"{image.stem}.md"
    write_xlsx(xlsx_path, result.lines, source=image.name)
    write_markdown(md_path, result.lines, source=image.name)

    low = sum(1 for line in result.lines if line.confidence < 0.85)
    print(
        f"{image.name}: {len(result.lines)} lines, "
        f"mean confidence {result.mean_confidence:.3f}, {low} below 0.85, "
        f"{result.elapsed:.2f} s"
    )
    print(xlsx_path)
    print(md_path)
    return 0


def run_gui(image: Path | None, max_side: int, lang: str) -> int:
    import tkinter as tk

    from ocr.ui import OcrApp

    enable_dpi_awareness()
    root = tk.Tk()
    app = OcrApp(engine=PaddleEngine(lang=lang, max_side=max_side or None), root=root)
    if image is not None:
        app.load_image(image)
    root.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="QwikDoQocr", description="Review and export OCR results for document images."
    )
    parser.add_argument("image", nargs="?", type=Path, help="image to open or process")
    parser.add_argument(
        "--headless", action="store_true",
        help="no window: write <name>.xlsx and <name>.md next to the image",
    )
    parser.add_argument("--outdir", type=Path, help="directory for headless exports")
    parser.add_argument(
        "--max-side", type=int, default=2000,
        help="downscale the longest edge before OCR (0 disables, default 2000)",
    )
    parser.add_argument("--lang", default="en", help="PaddleOCR language model (default en)")
    args = parser.parse_args(argv)

    if args.headless and args.image is None:
        parser.error("--headless needs an image path")

    if args.headless:
        return run_headless(args.image, args.outdir, args.max_side, args.lang)
    return run_gui(args.image, args.max_side, args.lang)


if __name__ == "__main__":
    raise SystemExit(main())
