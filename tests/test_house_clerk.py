"""Parser of official House periodic transaction reports, on a real filing."""

from pathlib import Path

from src.event_study import holm
from src.house_clerk import lower_bound, parse

FIX = Path(__file__).parent / "fixtures" / "ptr_20024277.txt"


def test_parses_every_stock_row_of_a_real_report():
    text = FIX.read_text()
    rows = parse(text)
    assert len(rows) == text.count("[ST]") == 4
    assert [r["ticker"] for r in rows] == ["ALB", "ALB", "SCHW", "NEE"]
    assert [r["transaction_type"] for r in rows] == ["S", "S", "P", "S"]
    assert rows[2]["transaction_date"] == "12/14/2023"
    assert rows[2]["notification_date"] == "01/08/2024"
    assert rows[2]["amount_low"] == 50001.0
    assert all(r["asset_type"] == "ST" for r in rows)


def test_amount_lower_bound():
    assert lower_bound("$1,001 - $15,000") == 1001.0
    assert lower_bound("Over $50,000,000") == 50000000.0
    assert lower_bound(None) is None


def test_holm_is_monotone_and_capped():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert out["a"] == 0.03 and out["b"] == 0.08 and out["c"] == 0.5
