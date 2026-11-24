"""Parser of Senate eFD periodic transaction reports, on a real report table."""

from pathlib import Path

import pytest

from src.senate_efd import parse, session

FIX = Path(__file__).parent / "fixtures" / "senate_ptr.html"


def test_parses_a_real_senate_report():
    rows = parse(FIX.read_text())
    assert [r["ticker"] for r in rows] == ["PFS", "REVB", "PFS", "REVB", "GE"]
    assert [r["transaction_type"] for r in rows] == ["S", "S", "S", "S", "P"]
    assert [r["partial"] for r in rows] == [False, False, False, True, False]
    assert rows[4]["transaction_date"] == "12/07/2017"
    assert all(r["asset_type"] == "ST" and r["amount_low"] == 1001.0 for r in rows)


def test_terms_are_never_accepted_without_the_flag():
    with pytest.raises(SystemExit):
        session(accept_terms=False)
