"""Feature, ranking and backtest-timing tests (no network access)."""

import numpy as np
import pandas as pd
import pytest

import config
from src import backtest as bt
from src import signal as sg
from src.features import (
    compute_composite_score,
    compute_direction_flag,
    compute_disclosure_lag,
)


@pytest.fixture(autouse=True)
def _tmp_outputs(tmp_path, monkeypatch):
    """Keep tests from writing into the repo's outputs/ directory."""
    monkeypatch.setattr(config, "POSITIONS_CSV", tmp_path / "positions.csv")
    monkeypatch.setattr(config, "RESULTS_CSV", tmp_path / "results.csv")


def test_disclosure_lag_drops_impossible_rows():
    df = pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(["2023-01-01"] * 4),
            "disclosure_date": pd.to_datetime(
                ["2023-01-10", "2023-02-15", "2023-03-01", "2022-12-25"]
            ),
        }
    )
    out = compute_disclosure_lag(df)
    # 9 days ok; 45 days ok (boundary); 59 days > STOCK Act limit; negative lag
    assert list(out["lag_days"]) == [9, 45]


def test_direction_flag_mapping_and_unknown():
    df = pd.DataFrame({"transaction_type": ["Buy", "SALE", "purchase", "exchange"]})
    out = compute_direction_flag(df)
    assert list(out["direction"]) == [1, -1, 1, 0]


def test_composite_score_formula_and_sign():
    df = pd.DataFrame(
        {
            "direction": [1, -1],
            "lag_days": [30, 30],
            "committee_weight": [1.0, 1.0],
            "coverage_score": [0, 3],
        }
    )
    out = compute_composite_score(df)
    assert out["composite_score"].iloc[0] == pytest.approx(np.log1p(30))
    assert out["composite_score"].iloc[1] == pytest.approx(-np.log1p(30) / 4)


def test_composite_score_requires_columns():
    with pytest.raises(ValueError):
        compute_composite_score(pd.DataFrame({"direction": [1]}))


def _scored(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ticker": [f"T{i % 20}" for i in range(n)],
            "disclosure_date": pd.date_range("2023-01-02", periods=n, freq="D"),
            "composite_score": rng.normal(size=n),
        }
    )


def test_rank_signals_deciles_and_positions():
    out = sg.rank_signals(_scored())
    assert out["decile"].min() == 1 and out["decile"].max() == 10
    longs = out[out["position"] == "LONG"]
    shorts = out[out["position"] == "SHORT"]
    assert len(longs) >= 9 and len(shorts) >= 9
    assert longs["composite_score"].min() > shorts["composite_score"].max()
    assert (longs["composite_score"] > 0).all() and (shorts["composite_score"] < 0).all()


def test_weekly_positions_deduplicates_same_ticker_week():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "CCC"],
            "disclosure_date": pd.to_datetime(
                ["2023-04-03", "2023-04-05", "2023-04-04", "2023-04-04"]
            ),
            "position": ["LONG", "LONG", "SHORT", "NEUTRAL"],
            "composite_score": [1.0, 2.5, -1.0, 0.1],
        }
    )
    out = sg.generate_weekly_positions(df)
    assert set(out["week"]) == {"2023-W14"}
    assert len(out) == 2  # NEUTRAL dropped, AAA de-duplicated
    kept = out[out["ticker"] == "AAA"]["composite_score"].iloc[0]
    assert kept == 2.5  # strongest signal retained


def test_week_to_entry_date_is_monday():
    assert bt._week_to_entry_date("2023-W14") == pd.Timestamp("2023-04-03")


def _prices():
    idx = pd.bdate_range("2020-12-01", "2021-03-31")
    n = len(idx)
    return pd.DataFrame(
        {"UP": 100 * 1.01 ** np.arange(n), "DOWN": 100 * 0.99 ** np.arange(n)}, index=idx
    )


def test_backtest_enters_after_the_disclosure_week_and_flips_short():
    px = _prices()
    pos = pd.DataFrame(
        {
            "week": ["2021-W02", "2021-W02"],
            "ticker": ["UP", "DOWN"],
            "position": ["LONG", "SHORT"],
        }
    )
    res = bt.run_backtest(pos, px)
    entry = pd.Timestamp("2021-01-18")  # Monday of W03
    i = px.index.get_indexer([entry])[0]
    w = config.POST_DISCLOSURE_WINDOW
    exp_long = px["UP"].iloc[i + w] / px["UP"].iloc[i] - 1
    exp_short_raw = px["DOWN"].iloc[i + w] / px["DOWN"].iloc[i] - 1
    assert res["long_return"].iloc[0] == pytest.approx(exp_long)
    assert res["ls_return"].iloc[0] == pytest.approx(exp_long - exp_short_raw)
    assert res["ls_return"].iloc[0] > 0  # long winner + short loser


def test_backtest_ignores_prices_before_entry():
    px = _prices()
    pos = pd.DataFrame({"week": ["2021-W02"], "ticker": ["UP"], "position": ["LONG"]})
    base = bt.run_backtest(pos, px)["long_return"].iloc[0]
    px2 = px.copy()
    px2.loc[:"2021-01-15", "UP"] *= 5.0
    assert bt.run_backtest(pos, px2)["long_return"].iloc[0] == pytest.approx(base)


def test_backtest_handles_53_week_iso_year():
    """2020 has 53 ISO weeks; W53 must enter the following Monday, not be skipped."""
    idx = pd.bdate_range("2020-12-01", "2021-03-31")
    px = pd.DataFrame({"AAA": np.linspace(100, 130, len(idx))}, index=idx)
    pos = pd.DataFrame({"week": ["2020-W52"], "ticker": ["AAA"], "position": ["LONG"]})
    res = bt.run_backtest(pos, px)
    i = px.index.get_indexer([pd.Timestamp("2020-12-28")])[0]  # Monday of W53
    w = config.POST_DISCLOSURE_WINDOW
    assert res["long_return"].iloc[0] == pytest.approx(px["AAA"].iloc[i + w] / px["AAA"].iloc[i] - 1)


def test_empty_positions_returns_empty_frame():
    assert bt.run_backtest(pd.DataFrame(columns=["week", "ticker", "position"]), _prices()).empty
