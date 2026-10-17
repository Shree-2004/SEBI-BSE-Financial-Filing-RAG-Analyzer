"""
Tests for eval/strict_score.py — the strict answer matcher that replaced
the any-keyword metric as the headline eval number.

Run: pytest tests/test_strict_score.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from strict_score import strict_match  # noqa: E402


def test_indian_digit_grouping_matches_by_value():
    assert strict_match("Nine-month revenue was 1,90,845 crore", [["1,90,845"]])
    assert strict_match("Nine-month revenue was 190845 crore", [["1,90,845"]])


def test_number_must_match_whole_value_not_substring():
    assert not strict_match("Finance costs were 1,234 crore", [["234"]])
    assert not strict_match("Revenue was 59 crore", [["63,973"]])


def test_citation_page_numbers_do_not_count():
    assert not strict_match("Face value is not stated [TCS | Q3FY25 | Page 10]", [["10"]])


def test_decimal_equivalence():
    assert strict_match("A dividend of Rs 5.50 per share", [["5.5"]])


def test_text_gold_ignores_spacing_and_punctuation():
    assert strict_match("Audited by B S R & Co. LLP", [["BSR"]])


def test_all_groups_required():
    gold = [["10", "76"], ["66", "76"]]
    assert strict_match("Interim 10 and special 66 per share", gold)
    assert strict_match("Total dividend of 76 per share", gold)
    assert not strict_match("Interim dividend of 10 per share", gold)


def test_fullwidth_citation_brackets_do_not_count():
    # gpt-oss-120b cites as 【TCS | Q3FY25 | Page 10】 instead of [...]
    assert not strict_match("Not stated 【TCS | Q3FY25 | Page 10】", [["10"]])


def test_stray_bracket_does_not_swallow_later_text():
    # OCR'd chunk with an unclosed "[" joined to a later chunk that has a "]"
    context = "Profit [Refer Note\nOwners of the Company 2,318.35\nTotal [a] 10"
    assert strict_match(context, [["2,318.35"]])
