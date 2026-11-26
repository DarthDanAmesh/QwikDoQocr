"""Engine adapter tests using a stand-in paddleocr module.

These prove the parsing, coordinate scaling, and ordering contract. They do
not prove anything about PaddleOCR's own recognition quality, which needs
the real package on Python 3.8-3.12.
"""

from __future__ import annotations

import sys
import types

import pytest
from PIL import Image

from ocr.engine import EngineUnavailable, PaddleEngine, lines_from_raw


def box(left, top, right, bottom):
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def test_lines_from_wrapped_result():
    raw = [[[box(10, 20, 110, 40), ("hello", 0.9)]]]
    lines = lines_from_raw(raw)
    assert [(item.text, item.confidence) for item in lines] == [("hello", 0.9)]
    assert lines[0].left == 10 and lines[0].bottom == 40


def test_lines_from_flat_result():
    raw = [[box(0, 0, 10, 10), ("a", 0.5)], [box(0, 20, 10, 30), ("b", 0.6)]]
    assert [item.text for item in lines_from_raw(raw)] == ["a", "b"]


def test_lines_from_single_line_and_empty():
    assert [item.text for item in lines_from_raw([[box(0, 0, 1, 1), ("only", 0.4)]])] == ["only"]
    assert lines_from_raw([]) == []
    assert lines_from_raw(None) == []
    assert lines_from_raw([[]]) == []


def test_lines_from_non_four_point_box_uses_bounding_rect():
    raw = [[[[5, 5], [25, 5], [25, 15], [5, 15], [5, 5]], ("five", 0.7)]]
    result = lines_from_raw(raw)[0]
    assert result.left == 5 and result.bottom == 15


def test_lines_from_result_ignores_malformed_entries():
    raw = [[None, "junk", [box(0, 0, 5, 5), ("ok", 0.8)]]]
    assert [item.text for item in lines_from_raw(raw)] == ["ok"]


@pytest.fixture
def fake_paddle(monkeypatch):
    class FakePaddleOCR:
        kwargs: dict = {}
        received: tuple = ()

        def __init__(self, **kwargs):
            FakePaddleOCR.kwargs = kwargs

        def ocr(self, image, cls=True):
            FakePaddleOCR.received = (image, cls)
            return [[
                [box(10, 40, 90, 60), ("second", 0.9)],
                [box(10, 10, 90, 30), ("first", 0.95)],
            ]]

    module = types.ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    return FakePaddleOCR


def test_engine_orders_lines_and_passes_image_path(tmp_path, fake_paddle):
    image = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image)

    result = PaddleEngine(max_side=None).run(image)

    assert [item.text for item in result.lines] == ["first", "second"]
    assert (result.width, result.height) == (200, 100)
    assert result.mean_confidence == pytest.approx(0.925)
    assert fake_paddle.received == (str(image), True)
    assert fake_paddle.kwargs["show_log"] is False


def test_engine_downscales_and_scales_boxes_back(tmp_path, fake_paddle):
    pytest.importorskip("numpy")
    image = tmp_path / "big.png"
    Image.new("RGB", (400, 200), "white").save(image)

    result = PaddleEngine(max_side=100).run(image)

    array, cls = fake_paddle.received
    assert cls is True
    assert array.shape == (50, 100, 3)
    assert (result.width, result.height) == (400, 200)
    # Fake boxes were produced in the 100px-wide image, so they come back 4x larger.
    assert result.lines[0].left == 40
    assert result.lines[0].height == 80


def test_engine_reports_missing_paddleocr(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    with pytest.raises(EngineUnavailable) as info:
        PaddleEngine().run("unused.png")
    assert "paddlepaddle" in str(info.value)
