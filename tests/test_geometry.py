"""Reading order is the behaviour that the flat OCR output got wrong.

The boxes below are the real coordinates PaddleOCR 2.9.1 produced for
section 11 of docu.jpg (output/docu.xlsx). Detection returned the line at
y=380 before the line at y=372; the reading order must not.
"""

from ocr.geometry import (
    clamp_scale,
    fit_scale,
    point_in_box,
    scale_box,
    scale_lines,
    sort_reading_order,
)
from ocr.models import OcrLine


def line(text: str, left: float, top: float, right: float, bottom: float, conf: float = 0.95) -> OcrLine:
    return OcrLine(
        text, conf,
        ((left, top), (right, top), (right, bottom), (left, bottom)),
    )


REAL_SECTION_11 = [
    line("This Agreement represents the complete agreement", 92, 358, 413, 371),
    line("between them.It may be amended only by a writing", 92, 380, 390, 394),
    line("between the parties and supersedes all prior agreements", 94, 372, 412, 382),
]


def test_real_detection_order_is_not_reading_order():
    assert [item.top for item in REAL_SECTION_11] == [358, 380, 372]


def test_reading_order_undoes_skewed_detection():
    ordered = sort_reading_order(REAL_SECTION_11)
    assert [item.top for item in ordered] == [358, 372, 380]
    assert ordered[1].text.startswith("between the parties")
    assert ordered[2].text.startswith("between them.")


def test_single_row_sorts_left_to_right():
    row = [line("right", 300, 10, 400, 22), line("left", 20, 12, 200, 24)]
    ordered = sort_reading_order(row)
    assert [item.text for item in ordered] == ["left", "right"]


def test_separate_paragraphs_are_not_merged():
    paragraphs = [line("second", 20, 100, 200, 112), line("first", 20, 10, 200, 22)]
    assert [item.text for item in sort_reading_order(paragraphs)] == ["first", "second"]


def test_columns_at_same_height_stay_sorted_by_x():
    columns = [
        line("header", 20, 10, 120, 22),
        line("col2-a", 220, 40, 320, 52),
        line("col2-b", 220, 70, 320, 82),
        line("col1-a", 20, 40, 120, 52),
        line("col1-b", 20, 70, 120, 82),
    ]
    ordered = sort_reading_order(columns)
    assert [item.text for item in ordered] == [
        "header", "col1-a", "col2-a", "col1-b", "col2-b",
    ]


def test_already_ordered_input_is_stable():
    ordered = sort_reading_order(REAL_SECTION_11)
    assert sort_reading_order(ordered) == ordered


def test_empty_and_single_inputs_pass_through():
    assert sort_reading_order([]) == []
    only = [line("only", 0, 0, 10, 10)]
    assert sort_reading_order(only) == only


def test_scale_box_moves_every_corner():
    box = scale_box(((1, 2), (3, 4), (5, 6), (7, 8)), 2.0, 0.5)
    assert box == ((2.0, 1.0), (6.0, 2.0), (10.0, 3.0), (14.0, 4.0))


def test_scale_lines_keeps_text_and_confidence():
    scaled = scale_lines([line("text", 20, 30, 40, 40, 0.8)], 3.0, 3.0)
    assert scaled[0].text == "text"
    assert scaled[0].confidence == 0.8
    assert scaled[0].left == 60 and scaled[0].top == 90


def test_clamp_scale_bounds_zoom():
    assert clamp_scale(0.01) == 0.1
    assert clamp_scale(99) == 4.0
    assert clamp_scale(1.5) == 1.5


def test_fit_scale_fits_and_caps():
    assert fit_scale((1000, 500), (500, 500)) == 0.5
    assert fit_scale((100, 100), (500, 500)) == 1.0  # cap at 100%
    assert fit_scale((1000, 500), (4000, 4000), cap=4.0) == 4.0
    assert fit_scale((1000, 500), (1, 1)) == 1.0  # not laid out yet


def test_point_in_box_hit_test():
    box = ((10, 20), (110, 20), (110, 60), (10, 60))
    assert point_in_box((50, 40), box)
    assert point_in_box((10, 20), box)
    assert not point_in_box((5, 40), box)
    assert not point_in_box((50, 70), box)
