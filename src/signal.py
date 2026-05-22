"""
src/signal.py
=============
OLS regression, signal ranking, and weekly position generation.

This module bridges the feature-engineering outputs and the backtest layer.
It answers two questions:

1. Do the features statistically predict forward returns?  (OLS regression)
2. Which disclosures are actionable longs / shorts this week?  (ranking)

Usage
-----
    python -m src.signal

Runs an end-to-end demo using synthetic data when live data is unavailable.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


# ── Forward return computation ────────────────────────────────────────────────

def _compute_forward_return(
    ticker: str,
    disclosure_date: pd.Timestamp,
    price_data: pd.DataFrame,
    window: int = config.POST_DISCLOSURE_WINDOW,
) -> float | None:
    """
    Compute the *window*-day forward return for *ticker* starting on
    *disclosure_date*.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    disclosure_date : pd.Timestamp
        The date on which the disclosure was made public (day 0).
    price_data : pd.DataFrame
        Wide-format price DataFrame indexed by date, one column per ticker.
    window : int
        Number of trading days over which to measure the return.

    Returns
    -------
    float or None
        Arithmetic return ``(P_{t+window} / P_t) - 1`` if sufficient data
        exist, otherwise ``None``.
    """
    if ticker not in price_data.columns:
        return None

    prices = price_data[ticker].dropna()
    future_prices = prices[prices.index >= disclosure_date]

    if len(future_prices) < window + 1:
        return None

    p0 = future_prices.iloc[0]
    pn = future_prices.iloc[window]

    if p0 == 0 or np.isnan(p0):
        return None

    return float(pn / p0 - 1.0)


# ── OLS regression ────────────────────────────────────────────────────────────

def run_ols_regression(
    df: pd.DataFrame,
    price_data: pd.DataFrame,
) -> tuple[object, pd.DataFrame]:
    """
    Estimate OLS: ``forward_return ~ direction + lag_days + coverage_score
    + committee_weight``.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched disclosures DataFrame.  Must contain columns:
        ``ticker``, ``disclosure_date``, ``direction``, ``lag_days``,
        ``coverage_score``, ``committee_weight``.
    price_data : pd.DataFrame
        Wide-format adjusted close prices indexed by date.

    Returns
    -------
    model : statsmodels RegressionResultsWrapper
        Fitted OLS model (provides ``.summary()``, ``.params``, etc.).
    df_enriched : pd.DataFrame
        Copy of *df* with a ``forward_return`` column appended.  Rows for
        which a forward return could not be computed are dropped.

    Notes
    -----
    The OLS specification is deliberately simple: its primary purpose is to
    establish whether the sign and magnitude of coefficients align with the
    informational-advantage hypothesis (positive ``direction`` coefficient,
    positive ``lag_days`` coefficient, negative ``coverage_score`` coefficient).

    Interpretation caveats:

    * ``disclosure_date`` is publicly observable (no look-ahead), so using it
      as the event date is clean.
    * ``coverage_score`` is computed from a snapshot that may post-date the
      disclosure for historical rows — a minor form of look-ahead for
      back-period data.
    * The sample size may be small for some committee strata; treat t-stats
      with appropriate scepticism.
    """
    df = df.copy()

    logger.info("Computing %d-day forward returns...", config.POST_DISCLOSURE_WINDOW)
    df["forward_return"] = df.apply(
        lambda row: _compute_forward_return(
            row["ticker"], row["disclosure_date"], price_data
        ),
        axis=1,
    )

    df_valid = df.dropna(
        subset=["forward_return", "direction", "lag_days", "coverage_score", "committee_weight"]
    )
    n_dropped = len(df) - len(df_valid)
    if n_dropped > 0:
        logger.warning(
            "Dropped %d rows with missing forward returns or features.", n_dropped
        )

    if len(df_valid) < 10:
        logger.warning(
            "Only %d valid rows for regression — results will be unreliable.", len(df_valid)
        )

    X = df_valid[["direction", "lag_days", "coverage_score", "committee_weight"]]
    X = sm.add_constant(X)
    y = df_valid["forward_return"]

    logger.info("Fitting OLS with %d observations, 4 regressors...", len(df_valid))
    model = sm.OLS(y, X).fit()

    logger.info("\n%s", model.summary())
    return model, df_valid


# ── Signal ranking ────────────────────────────────────────────────────────────

def rank_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort disclosures by ``composite_score`` and assign decile ranks.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched disclosures DataFrame with a ``composite_score`` column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two additional columns:

        ``decile``
            Integer 1–10; 10 = highest composite score.
        ``position``
            String ``"LONG"`` for the top decile, ``"SHORT"`` for the bottom
            decile, ``"NEUTRAL"`` otherwise.

    Notes
    -----
    Decile assignment uses ``pd.qcut`` with duplicate-edge handling via
    ``duplicates="drop"``.  When the score distribution is degenerate (many
    ties), fewer than 10 distinct buckets may be produced; ``decile`` values
    are then mapped to the available buckets.
    """
    df = df.copy().sort_values("composite_score", ascending=False)

    try:
        df["decile"] = pd.qcut(
            df["composite_score"],
            q=10,
            labels=False,
            duplicates="drop",
        )
        # pd.qcut labels are 0-indexed; shift to 1-indexed
        df["decile"] = df["decile"] + 1
    except ValueError:
        logger.warning(
            "qcut failed (likely too many ties) — falling back to rank-based decile."
        )
        df["decile"] = (
            df["composite_score"]
            .rank(pct=True, method="first")
            .mul(10)
            .clip(1, 10)
            .round()
            .astype(int)
        )

    top_cut = df["composite_score"].quantile(config.TOP_DECILE_THRESHOLD)
    bot_cut = df["composite_score"].quantile(1 - config.TOP_DECILE_THRESHOLD)

    df["position"] = "NEUTRAL"
    df.loc[df["composite_score"] >= top_cut, "position"] = "LONG"
    df.loc[df["composite_score"] <= bot_cut, "position"] = "SHORT"

    logger.info(
        "Signal ranking complete. LONG: %d, SHORT: %d, NEUTRAL: %d",
        (df["position"] == "LONG").sum(),
        (df["position"] == "SHORT").sum(),
        (df["position"] == "NEUTRAL").sum(),
    )
    return df


# ── Weekly position generation ────────────────────────────────────────────────

def generate_weekly_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group ranked disclosures by ISO week and extract actionable positions.

    Parameters
    ----------
    df : pd.DataFrame
        Ranked disclosures DataFrame; must contain columns:
        ``disclosure_date``, ``ticker``, ``position``, ``composite_score``.

    Returns
    -------
    pd.DataFrame
        Columns: ``[week, ticker, position, composite_score]`` where
        ``week`` is an ISO week string ``"YYYY-Www"`` (e.g. ``"2023-W14"``).
        Only ``"LONG"`` and ``"SHORT"`` rows are returned; ``"NEUTRAL"``
        rows are excluded.

    Notes
    -----
    The position assigned in week *W* is intended to be entered at the open
    of the first trading day following the end of week *W* and held for
    ``POST_DISCLOSURE_WINDOW`` trading days.  The backtest layer
    (:mod:`src.backtest`) implements this timing convention.

    If multiple disclosures of the same ticker appear in the same week with
    the same position direction, the one with the highest |composite_score|
    is retained to avoid double-counting.
    """
    df = df.copy()

    df["week"] = df["disclosure_date"].dt.isocalendar().apply(
        lambda r: f"{int(r['year'])}-W{int(r['week']):02d}", axis=1
    )

    active = df[df["position"].isin(["LONG", "SHORT"])].copy()

    # Deduplicate: keep strongest signal per (week, ticker, position)
    active["abs_score"] = active["composite_score"].abs()
    active = (
        active.sort_values("abs_score", ascending=False)
        .drop_duplicates(subset=["week", "ticker", "position"])
        .drop(columns=["abs_score"])
    )

    positions_df = active[["week", "ticker", "position", "composite_score"]].reset_index(
        drop=True
    )
    positions_df = positions_df.sort_values(["week", "position", "composite_score"])

    logger.info(
        "Weekly positions: %d rows across %d unique weeks",
        len(positions_df),
        positions_df["week"].nunique(),
    )

    # Persist to disk
    positions_df.to_csv(config.POSITIONS_CSV, index=False)
    logger.info("Positions saved to %s", config.POSITIONS_CSV)

    return positions_df


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    logger.info("=== Signal module demo (synthetic data) ===")

    # Build a minimal synthetic dataset so the module runs end-to-end
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2022-01-03", periods=n, freq="B")

    synthetic = pd.DataFrame(
        {
            "ticker": rng.choice(["AAPL", "MSFT", "LMT", "RTX", "GS"], size=n),
            "disclosure_date": rng.choice(dates, size=n),
            "transaction_date": None,
            "direction": rng.choice([-1, 1], size=n),
            "lag_days": rng.integers(1, 45, size=n),
            "coverage_score": rng.integers(0, 100, size=n),
            "committee_weight": rng.choice([0.5, 0.75, 0.85, 1.0], size=n),
            "composite_score": rng.standard_normal(size=n),
            "transaction_type": rng.choice(["buy", "sell"], size=n),
        }
    )
    synthetic["disclosure_date"] = pd.to_datetime(synthetic["disclosure_date"])

    # Build a synthetic price table
    all_tickers = ["AAPL", "MSFT", "LMT", "RTX", "GS"]
    price_dates = pd.date_range("2022-01-01", periods=300, freq="B")
    price_data = pd.DataFrame(
        np.cumprod(1 + rng.normal(0.0005, 0.015, size=(300, len(all_tickers))), axis=0) * 100,
        index=price_dates,
        columns=all_tickers,
    )

    # OLS
    model, enriched = run_ols_regression(synthetic, price_data)
    print("\n=== OLS Summary ===")
    print(model.summary())

    # Ranking
    ranked = rank_signals(synthetic)
    print("\nTop 10 signals:")
    print(ranked.head(10)[["ticker", "composite_score", "decile", "position"]].to_string())

    # Weekly positions
    positions = generate_weekly_positions(ranked)
    print("\nWeekly positions sample:")
    print(positions.head(10).to_string())

    sys.exit(0)
