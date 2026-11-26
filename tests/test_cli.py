"""Headless mode: argument handling and export of the recognised lines."""

from __future__ import annotations

import pytest
from PIL import Image

import ocr_app
from ocr.engine import EngineUnavailable
from ocr.models import OcrLine, OcrResult


def line(text, top, conf=0.94):
    return OcrLine(text, conf, ((10, top), (90, top), (90, top + 12), (10, top + 12)))


class FakeEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, path):
        return OcrResult(
            [line("first line", 10), line("second line", 30, 0.7)], 100, 50, 0.31
        )


class BrokenEngine:
    def __init__(self, **_kwargs):
        pass

    def run(self, _path):
        raise EngineUnavailable("paddlepaddle needs Python 3.8-3.12")


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (100, 50), "white").save(path)
    return path


def test_headless_writes_both_exports(tmp_path, image, monkeypatch, capsys):
    monkeypatch.setattr(ocr_app, "PaddleEngine", FakeEngine)

    code = ocr_app.run_headless(image, tmp_path / "out", max_side=0, lang="en")

    assert code == 0
    assert (tmp_path / "out" / "page.xlsx").exists()
    assert (tmp_path / "out" / "page.md").exists()
    printed = capsys.readouterr().out
    assert "page.png: 2 lines" in printed
    assert "1 below 0.85" in printed


def test_headless_defaults_to_image_directory(tmp_path, image, monkeypatch):
    monkeypatch.setattr(ocr_app, "PaddleEngine", FakeEngine)

    assert ocr_app.run_headless(image, None, max_side=2000, lang="en") == 0
    assert (tmp_path / "page.xlsx").exists()


def test_headless_returns_1_when_no_text(tmp_path, image, monkeypatch):
    class EmptyEngine(FakeEngine):
        def run(self, path):
            return OcrResult([], 100, 50, 0.1)

    monkeypatch.setattr(ocr_app, "PaddleEngine", EmptyEngine)

    assert ocr_app.run_headless(image, tmp_path / "out", max_side=0, lang="en") == 1
    assert not (tmp_path / "out" / "page.xlsx").exists()


def test_headless_returns_2_when_paddle_missing(tmp_path, image, monkeypatch):
    monkeypatch.setattr(ocr_app, "PaddleEngine", BrokenEngine)

    assert ocr_app.run_headless(image, tmp_path / "out", max_side=0, lang="en") == 2


def test_headless_flag_requires_an_image():
    with pytest.raises(SystemExit) as info:
        ocr_app.main(["--headless"])
    assert info.value.code == 2
