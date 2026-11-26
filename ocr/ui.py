"""Tkinter desktop app for reviewing and correcting OCR output.

The image and the recognised lines are kept in sync: clicking a box focuses
its row and focusing a row highlights its box, so a reviewer can check every
line against the source without zooming around the document.
"""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from PIL import Image, ImageTk

from .engine import PaddleEngine
from .export import LOW_CONFIDENCE, document_text, write_markdown, write_xlsx
from .geometry import clamp_scale, fit_scale, point_in_box
from .models import OcrLine, OcrResult

APP_NAME = "QwikDoQocr"
IMAGE_TYPES = [
    ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]

BG = "#161a21"
PANEL = "#1f242d"
ROW = "#1f242d"
ROW_SELECTED = "#2c3b53"
FIELD = "#12151b"
FG = "#e7ebf2"
MUTED = "#98a2b3"
ACCENT = "#4c9aff"
WARN = "#f0a35e"
FONT = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI Semibold", 9)


class ImageCanvas(ttk.Frame):
    """Scrollable image view with polygon overlays for recognised lines."""

    def __init__(self, master, on_select=None, on_zoom=None):
        super().__init__(master)
        self.on_select = on_select
        self.on_zoom = on_zoom

        self.canvas = tk.Canvas(self, background="#0e1117", highlightthickness=0, bd=0)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", self._on_configure)

        self._image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._lines: list[OcrLine] = []
        self._selected: int | None = None
        self._scale = 1.0
        self._auto_fit = True
        self._offset = (0, 0)
        self._scroll_size = (1, 1)
        self._hint = "Open an image to start\nCtrl+O"

    @property
    def has_image(self) -> bool:
        return self._image is not None

    def set_image(self, image: Image.Image | None) -> None:
        self._image = image
        self._lines = []
        self._selected = None
        self._auto_fit = True
        if image is None:
            self._hint = "Open an image to start\nCtrl+O"
        self._render()

    def set_lines(self, lines: list[OcrLine]) -> None:
        self._lines = lines
        self._selected = None
        self._auto_fit = True
        self._render()

    @property
    def scale(self) -> float:
        return self._scale

    def fit(self) -> None:
        self._auto_fit = True
        self._render()

    def set_zoom(self, scale: float) -> None:
        if self._image is None:
            return
        canvas = self.canvas
        old = self._scale
        centre_x = (canvas.canvasx(canvas.winfo_width() / 2) - self._offset[0]) / old
        centre_y = (canvas.canvasy(canvas.winfo_height() / 2) - self._offset[1]) / old

        self._auto_fit = False
        self._scale = clamp_scale(scale)
        self._render()

        total_width, total_height = self._scroll_size
        if total_width > canvas.winfo_width():
            self.canvas.xview_moveto(
                max(0.0, (centre_x * self._scale + self._offset[0] - canvas.winfo_width() / 2) / total_width)
            )
        if total_height > canvas.winfo_height():
            self.canvas.yview_moveto(
                max(0.0, (centre_y * self._scale + self._offset[1] - canvas.winfo_height() / 2) / total_height)
            )
        if self._selected is not None:
            self._scroll_to(self._selected)

    def zoom_by(self, factor: float) -> None:
        self.set_zoom(self._scale * factor)

    def select(self, index: int | None) -> None:
        self._selected = index
        self._draw_overlays()
        if index is not None:
            self._scroll_to(index)

    def _on_configure(self, _event) -> None:
        if self._image is None or self._auto_fit:
            self._render()

    def _render(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if self._image is None:
            canvas.configure(scrollregion=(0, 0, max(width, 1), max(height, 1)))
            canvas.create_text(
                max(width, 1) // 2,
                max(height, 1) // 2,
                text=self._hint,
                fill=MUTED,
                font=("Segoe UI", 11),
                anchor="center",
                justify="center",
            )
            self._photo = None
            self.hbar.grid_remove()
            self.vbar.grid_remove()
            return
        self.hbar.grid()
        self.vbar.grid()

        image_width, image_height = self._image.size
        if self._auto_fit:
            self._scale = fit_scale(
                (image_width, image_height), (width - 2, height - 2)
            )
        view = (
            max(1, round(image_width * self._scale)),
            max(1, round(image_height * self._scale)),
        )
        self._offset = (max(0, (width - view[0]) // 2), max(0, (height - view[1]) // 2))
        self._scroll_size = (max(view[0], width), max(view[1], height))
        if view == self._image.size:
            shown = self._image
        else:
            resample = (
                Image.Resampling.LANCZOS
                if self._scale < 1
                else Image.Resampling.BILINEAR
            )
            shown = self._image.resize(view, resample)
        self._photo = ImageTk.PhotoImage(shown)
        canvas.create_image(*self._offset, image=self._photo, anchor="nw")
        canvas.configure(scrollregion=(0, 0, self._scroll_size[0], self._scroll_size[1]))
        self._draw_overlays()
        if self.on_zoom:
            self.on_zoom(self._scale)

    def _draw_overlays(self) -> None:
        canvas = self.canvas
        canvas.delete("overlay")
        if self._image is None:
            return
        scale = self._scale
        offset_x, offset_y = self._offset
        font = ("Segoe UI", max(7, min(11, round(9 * scale))))
        for index, line in enumerate(self._lines):
            coords = [
                value
                for x, y in line.box
                for value in (x * scale + offset_x, y * scale + offset_y)
            ]
            selected = index == self._selected
            canvas.create_polygon(
                *coords,
                outline=ACCENT if selected else "#2b6fb0",
                width=2 if selected else 1,
                fill=ACCENT if selected else "",
                stipple="gray25",
                tags="overlay",
            )
            chip_colour = "#ffffff" if selected else "#7fb2e5"
            if line.height * scale >= 8:
                if line.left * scale >= 16:
                    canvas.create_text(
                        line.left * scale + offset_x - 4,
                        (line.top + line.height / 2) * scale + offset_y,
                        text=str(index + 1),
                        fill=chip_colour,
                        font=font,
                        anchor="e",
                        tags="overlay",
                    )
                else:
                    canvas.create_text(
                        coords[0] + 2,
                        coords[1] + 1,
                        text=str(index + 1),
                        fill=chip_colour,
                        font=font,
                        anchor="nw",
                        tags="overlay",
                    )

    def _scroll_to(self, index: int) -> None:
        line = self._lines[index]
        scale = self._scale
        offset_x, offset_y = self._offset
        total_width, total_height = self._scroll_size
        top = line.top * scale + offset_y
        bottom = line.bottom * scale + offset_y
        left = line.left * scale + offset_x
        right = max(x for x, _ in line.box) * scale + offset_x

        first, last = self.canvas.yview()
        if total_height and (top < first * total_height or bottom > last * total_height):
            self.canvas.yview_moveto(max(0.0, (top - 60) / total_height))
        first, last = self.canvas.xview()
        if total_width and (left < first * total_width or right > last * total_width):
            self.canvas.xview_moveto(max(0.0, (left - 60) / total_width))

    def _on_click(self, event) -> None:
        if self._image is None or not self._lines:
            return
        x = (self.canvas.canvasx(event.x) - self._offset[0]) / self._scale
        y = (self.canvas.canvasy(event.y) - self._offset[1]) / self._scale
        hit = None
        for index in range(len(self._lines) - 1, -1, -1):
            if point_in_box((x, y), self._lines[index].box):
                hit = index
                break
        if self.on_select:
            self.on_select(hit)


class LineList(ttk.Frame):
    """Scrollable list of recognised lines, each editable in place."""

    def __init__(self, master, on_edit=None, on_focus=None):
        super().__init__(master)
        self.on_edit = on_edit
        self.on_focus = on_focus
        self.lines: list[OcrLine] = []
        self._rows: list[dict] = []
        self._selected: int | None = None

        header = tk.Frame(self, background=PANEL)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(
            header, text="Recognized text", background=PANEL, foreground=MUTED,
            font=FONT_BOLD, anchor="w", padx=8, pady=6,
        ).pack(side="left")
        tk.Label(
            header, text="Confidence", background=PANEL, foreground=MUTED,
            font=FONT, padx=8, pady=6,
        ).pack(side="right")

        self.canvas = tk.Canvas(self, background=PANEL, highlightthickness=0, bd=0)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = tk.Frame(self.canvas, background=PANEL)
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.scroll.grid(row=1, column=1, sticky="ns")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.body.columnconfigure(0, weight=1)

        self.body.bind("<Configure>", lambda _e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")
        ))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self._window, width=e.width
        ))
        for widget in (self.canvas, self.body):
            widget.bind("<MouseWheel>", self._on_wheel)

        self._placeholder = "No text yet — open an image and run OCR."
        self._build_placeholder()

    def _build_placeholder(self) -> None:
        self._placeholder_label = tk.Label(
            self.body, text=self._placeholder, background=PANEL, foreground=MUTED,
            font=FONT, anchor="w", justify="left", padx=10, pady=14,
        )
        self._placeholder_label.grid(row=0, column=0, sticky="ew")

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        if not self._rows:
            self._placeholder_label.configure(text=text)

    def set_lines(self, lines: list[OcrLine]) -> None:
        for child in self.body.winfo_children():
            child.destroy()
        self.lines = lines
        self._rows = []
        self._selected = None
        if not lines:
            self._build_placeholder()
            return
        for index, line in enumerate(lines):
            self._build_row(index, line)
        self.canvas.yview_moveto(0.0)

    def _build_row(self, index: int, line: OcrLine) -> None:
        row = tk.Frame(self.body, background=ROW, padx=6, pady=2)
        row.grid(row=index, column=0, sticky="ew")
        row.columnconfigure(1, weight=1)

        number = tk.Label(
            row, text=str(index + 1), background=ROW, foreground=MUTED,
            font=FONT, width=3, anchor="e",
        )
        number.grid(row=0, column=0, padx=(0, 6))

        variable = tk.StringVar(value=line.text)
        entry = tk.Entry(
            row, textvariable=variable, background=ROW, foreground=FG,
            insertbackground=FG, relief="flat", highlightthickness=1,
            highlightbackground=ROW, highlightcolor=ACCENT, font=FONT,
        )
        entry.grid(row=0, column=1, sticky="ew")
        variable.trace_add("write", lambda *_args, i=index, var=variable: self._edited(i, var))

        low = line.confidence < LOW_CONFIDENCE
        confidence = tk.Label(
            row, text=f"{line.confidence * 100:.0f}%", background=ROW,
            foreground=WARN if low else MUTED, font=FONT, width=5, anchor="e",
        )
        confidence.grid(row=0, column=2, padx=(6, 2))

        for widget in (row, number, confidence):
            widget.bind("<Button-1>", lambda _e, entry=entry: entry.focus_set())
            widget.bind("<MouseWheel>", self._on_wheel)
        entry.bind("<MouseWheel>", self._on_wheel)
        entry.bind("<FocusIn>", lambda _e, i=index: self._focused(i))

        self._rows.append(
            {"frame": row, "entry": entry, "number": number, "confidence": confidence}
        )

    def _edited(self, index: int, variable: tk.StringVar) -> None:
        self.lines[index].text = variable.get()
        if self.on_edit:
            self.on_edit(index, variable.get())

    def _focused(self, index: int) -> None:
        self.select(index)
        if self.on_focus:
            self.on_focus(index)

    def select(self, index: int | None) -> None:
        self._selected = index
        for position, row in enumerate(self._rows):
            selected = position == index
            colour = ROW_SELECTED if selected else ROW
            row["frame"].configure(background=colour)
            row["number"].configure(background=colour)
            row["confidence"].configure(background=colour)
            row["entry"].configure(
                background=colour, highlightbackground=ACCENT if selected else colour
            )

    def _on_wheel(self, event) -> str:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"


class OcrApp:
    """Owns the window, the current image, the OCR result, and the exports."""

    def __init__(self, engine=None, root: tk.Tk | None = None, interactive: bool = True) -> None:
        self.root = root or tk.Tk()
        self.engine = engine if engine is not None else PaddleEngine()
        self.interactive = interactive

        self.root.title(APP_NAME)
        self.root.geometry("1200x780")
        self.root.minsize(900, 560)

        self._path: Path | None = None
        self._result: OcrResult | None = None
        self._dirty = False
        self._busy = False
        self._queue: queue.Queue = queue.Queue()
        self._poll_id: str | None = None

        self._configure_theme()
        self._build_toolbar()
        self._build_body()
        self._build_status()
        self._build_menu()
        self._bind_keys()
        self._refresh_actions()
        self._set_status("Open an image with Ctrl+O to begin.")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # -- construction -----------------------------------------------------

    def _configure_theme(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(
            ".", background=BG, foreground=FG, fieldbackground=FIELD,
            bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL, font=FONT,
        )
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Toolbar.TFrame", background=PANEL)
        style.configure(
            "TButton", background="#2b323d", foreground=FG, padding=(10, 5),
            borderwidth=0, focusthickness=0,
        )
        style.map(
            "TButton",
            background=[("disabled", "#232932"), ("pressed", "#39424f"), ("active", "#343c48")],
            foreground=[("disabled", "#5d6673")],
        )
        style.configure(
            "Accent.TButton", background=ACCENT, foreground="#0a1420", padding=(14, 5),
        )
        style.map(
            "Accent.TButton",
            background=[("disabled", "#2f4258"), ("pressed", "#3b86e8"), ("active", "#67a9ff")],
            foreground=[("disabled", "#7d90a6")],
        )
        style.configure("TScrollbar", background=PANEL, troughcolor=BG, borderwidth=0)
        for orient in ("Vertical", "Horizontal"):
            style.configure(
                f"{orient}.TScrollbar",
                background="#39424f", troughcolor=BG, bordercolor=BG,
                arrowcolor=MUTED, lightcolor=BG, darkcolor=BG, borderwidth=0,
            )
            style.map(f"{orient}.TScrollbar", background=[("active", "#4c5666")])
        style.configure("Status.TFrame", background=PANEL)
        style.configure("Status.TLabel", background=PANEL, foreground=MUTED, font=FONT)
        style.configure("Hint.TLabel", background=PANEL, foreground=MUTED, font=FONT)
        style.configure("Title.TLabel", background=PANEL, foreground=FG, font=FONT_BOLD)
        style.configure("Accent.Horizontal.TProgressbar", background=ACCENT, troughcolor=PANEL)
        style.configure("TPanedwindow", background=BG)

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, style="Toolbar.TFrame", padding=(10, 8))
        bar.pack(side="top", fill="x")
        self.root.configure(background=BG)

        self.open_btn = ttk.Button(bar, text="Open image…", command=self.open_image)
        self.open_btn.pack(side="left")
        ttk.Frame(bar, style="Toolbar.TFrame", width=8).pack(side="left")
        self.run_btn = ttk.Button(
            bar, text="Run OCR", style="Accent.TButton", command=self.run_ocr
        )
        self.run_btn.pack(side="left")
        self.copy_btn = ttk.Button(bar, text="Copy text", command=self.copy_text)
        self.copy_btn.pack(side="left", padx=(8, 0))
        self.xlsx_btn = ttk.Button(bar, text="Export XLSX…", command=lambda: self.export("xlsx"))
        self.xlsx_btn.pack(side="left", padx=(8, 0))
        self.md_btn = ttk.Button(bar, text="Export Markdown…", command=lambda: self.export("md"))
        self.md_btn.pack(side="left", padx=(6, 0))

        self.fit_btn = ttk.Button(bar, text="Fit", command=self.fit_view)
        self.zoom_out_btn = ttk.Button(bar, text="−", width=3, command=lambda: self.zoom(1 / 1.25))
        self.zoom_label = ttk.Label(bar, text="100%", style="Hint.TLabel", width=6, anchor="center")
        self.zoom_in_btn = ttk.Button(bar, text="+", width=3, command=lambda: self.zoom(1.25))
        self.zoom_in_btn.pack(side="right")
        self.zoom_label.pack(side="right")
        self.zoom_out_btn.pack(side="right")
        self.fit_btn.pack(side="right", padx=(0, 8))

        self.file_label = ttk.Label(bar, text="No image", style="Hint.TLabel")
        self.file_label.pack(side="right", padx=(0, 16))

    def _build_body(self) -> None:
        panes = ttk.Panedwindow(self.root, orient="horizontal")
        panes.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(panes)
        panes.add(left, weight=3)
        self.canvas = ImageCanvas(left, on_select=self._select, on_zoom=self._show_zoom)
        self.canvas.pack(fill="both", expand=True)

        right = ttk.Frame(panes)
        panes.add(right, weight=2)
        self.list = LineList(right, on_edit=self._edited, on_focus=self._select)
        self.list.pack(fill="both", expand=True)

    def _build_status(self) -> None:
        bar = ttk.Frame(self.root, style="Status.TFrame", padding=(10, 5))
        bar.pack(side="bottom", fill="x")
        self.status_label = ttk.Label(bar, text="", style="Status.TLabel")
        self.status_label.pack(side="left")
        self.stats_label = ttk.Label(bar, text="", style="Status.TLabel")
        self.stats_label.pack(side="right")
        self.progress = ttk.Progressbar(
            bar, mode="indeterminate", length=120,
            style="Accent.Horizontal.TProgressbar",
        )
        self.progress.pack(side="right", padx=(0, 12))
        self.progress.pack_forget()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open image…", accelerator="Ctrl+O", command=self.open_image)
        file_menu.add_command(label="Run OCR", accelerator="F5", command=self.run_ocr)
        file_menu.add_separator()
        file_menu.add_command(
            label="Export XLSX…", accelerator="Ctrl+S", command=lambda: self.export("xlsx")
        )
        file_menu.add_command(
            label="Export Markdown…", accelerator="Ctrl+Shift+S", command=lambda: self.export("md")
        )
        file_menu.add_command(label="Copy text", accelerator="Ctrl+C", command=self.copy_text)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.close)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_command(label="Fit to window", accelerator="Ctrl+0", command=self.fit_view)
        view_menu.add_command(label="Zoom in", accelerator="Ctrl++", command=lambda: self.zoom(1.25))
        view_menu.add_command(label="Zoom out", accelerator="Ctrl+-", command=lambda: self.zoom(1 / 1.25))
        menubar.add_cascade(label="View", menu=view_menu)
        self.root.configure(menu=menubar)

    def _bind_keys(self) -> None:
        self.root.bind("<Control-o>", lambda _e: self.open_image())
        self.root.bind("<F5>", lambda _e: self.run_ocr())
        self.root.bind("<Control-Return>", lambda _e: self.run_ocr())
        self.root.bind("<Control-s>", lambda _e: self.export("xlsx"))
        self.root.bind("<Control-Shift-S>", lambda _e: self.export("md"))
        self.root.bind("<Control-c>", self.copy_text)
        self.root.bind("<Control-0>", lambda _e: self.fit_view())
        self.root.bind("<Control-plus>", lambda _e: self.zoom(1.25))
        self.root.bind("<Control-equal>", lambda _e: self.zoom(1.25))
        self.root.bind("<Control-minus>", lambda _e: self.zoom(1 / 1.25))

    # -- actions ----------------------------------------------------------

    def open_image(self) -> None:
        if self._busy:
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="Open image", filetypes=IMAGE_TYPES
        )
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> bool:
        try:
            image = Image.open(path)
            image.load()
        except Exception as exc:
            self._error("Cannot open image", exc)
            return False
        self._path = Path(path)
        self._result = None
        self._dirty = False
        self.canvas.set_image(image)
        self.list.set_lines([])
        self.list.set_placeholder("Press Run OCR to recognise this image.")
        self.file_label.configure(text=path.name)
        self.stats_label.configure(text="")
        width, height = image.size
        self._set_status(f"Loaded {path.name} ({width}×{height}) — press Run OCR.")
        self._update_title()
        self._refresh_actions()
        return True

    def run_ocr(self, blocking: bool = False) -> bool:
        if self._path is None:
            self._set_status("Open an image first.")
            return False
        if self._busy:
            return False
        path = self._path
        if blocking:
            try:
                result = self.engine.run(path)
            except Exception as exc:
                self._error("OCR failed", exc)
                return False
            self._apply_result(path, result)
            return True

        self._set_busy(True, f"Running OCR on {path.name}…")
        threading.Thread(target=self._worker, args=(path,), daemon=True).start()
        self._poll_id = self.root.after(60, self._poll)
        return True

    def _worker(self, path: Path) -> None:
        try:
            self._queue.put(("result", path, self.engine.run(path)))
        except Exception as exc:
            self._queue.put(("error", path, exc))

    def _poll(self) -> None:
        self._poll_id = None
        try:
            while True:
                kind, path, payload = self._queue.get_nowait()
                if kind == "result":
                    self._apply_result(path, payload)
                else:
                    self._set_busy(False)
                    self._error("OCR failed", payload)
        except queue.Empty:
            pass
        if self._busy:
            self._poll_id = self.root.after(60, self._poll)

    def _apply_result(self, path: Path, result: OcrResult) -> None:
        self._set_busy(False)
        if path != self._path:
            return
        self._result = result
        self._dirty = False
        self.canvas.set_lines(result.lines)
        self.list.set_lines(result.lines)
        self._update_title()
        self._refresh_actions()

        lines = result.lines
        if not lines:
            self.stats_label.configure(text="0 lines")
            self._set_status("No text detected in this image.")
            return
        low = [str(index + 1) for index, line in enumerate(lines) if line.confidence < LOW_CONFIDENCE]
        stats = (
            f"{len(lines)} lines · mean confidence {result.mean_confidence * 100:.1f}%"
            f" · {result.elapsed:.2f} s"
        )
        self.stats_label.configure(text=stats)
        message = f"Recognised {len(lines)} lines."
        if low:
            shown = ", ".join(low[:8]) + ("…" if len(low) > 8 else "")
            message += f" Low confidence rows: {shown} (highlighted)."
        self._set_status(message)

    def copy_text(self, _event=None):
        widget = self.root.focus_get()
        if isinstance(widget, (tk.Entry, ttk.Entry)):
            return None
        if not self._result or not self._result.lines:
            self._set_status("Nothing to copy — run OCR first.")
            return None
        text = document_text(self._result.lines)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status(f"Copied {len(self._result.lines)} lines to the clipboard.")
        return "break"

    def export(self, kind: str) -> None:
        if not self._result or not self._result.lines:
            self._set_status("Nothing to export — run OCR first.")
            return
        if kind == "md":
            writer, extension = write_markdown, ".md"
            filetypes = [("Markdown", "*.md"), ("All files", "*.*")]
        else:
            writer, extension = write_xlsx, ".xlsx"
            filetypes = [("Excel workbook", "*.xlsx"), ("All files", "*.*")]
        stem = self._path.stem if self._path else "ocr"
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title=f"Export {kind.upper()}",
            defaultextension=extension,
            initialfile=stem + extension,
            filetypes=filetypes,
        )
        if not path:
            return
        try:
            writer(path, self._result.lines, source=self._path.name if self._path else None)
        except Exception as exc:
            self._error("Export failed", exc)
            return
        self._dirty = False
        self._update_title()
        self._set_status(f"Saved {Path(path).name}.")

    def fit_view(self) -> None:
        self.canvas.fit()
        self._show_zoom(self.canvas.scale)

    def zoom(self, factor: float) -> None:
        self.canvas.zoom_by(factor)

    def close(self) -> None:
        if self._dirty and self._result and self._result.lines:
            if not messagebox.askokcancel(
                APP_NAME, "Edits have not been exported. Close anyway?", parent=self.root
            ):
                return
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        self.root.destroy()

    # -- helpers ----------------------------------------------------------

    def _select(self, index: int | None) -> None:
        self.canvas.select(index)
        self.list.select(index)

    def _edited(self, _index: int, _text: str) -> None:
        self._dirty = True
        self._update_title()

    def _show_zoom(self, scale: float) -> None:
        self.zoom_label.configure(text=f"{scale * 100:.0f}%")

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self._busy = busy
        if busy:
            self.progress.pack(side="right", padx=(0, 12))
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()
        if message:
            self._set_status(message)
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        has_lines = bool(self._result and self._result.lines)
        ready = has_lines and not self._busy
        for button in (self.copy_btn, self.xlsx_btn, self.md_btn):
            button.state(["!disabled"] if ready else ["disabled"])
        for button in (self.fit_btn, self.zoom_in_btn, self.zoom_out_btn):
            button.state(["!disabled"] if self.canvas.has_image else ["disabled"])
        if self._busy:
            self.run_btn.state(["disabled"])
            self.open_btn.state(["disabled"])
        else:
            self.run_btn.state(["!disabled"] if self._path else ["disabled"])
            self.open_btn.state(["!disabled"])

    def _update_title(self) -> None:
        name = self._path.name if self._path else "no image"
        mark = " *" if self._dirty else ""
        self.root.title(f"{name}{mark} — {APP_NAME}")

    def _error(self, title: str, exc: Exception) -> None:
        if isinstance(exc, FileNotFoundError):
            message = f"The file could not be found:\n{exc.filename}"
        else:
            message = str(exc) or exc.__class__.__name__
        self._set_status(f"{title}: {message.splitlines()[0]}")
        if self.interactive:
            messagebox.showerror(title, message, parent=self.root)
        else:
            print(f"{title}: {message}", file=sys.stderr)
