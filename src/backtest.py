"""
src/backtest.py
===============
Event-driven long/short backtest for the Congressional Trading Lag Arbitrage
strategy.

Implementation uses only ``pandas`` and ``numpy`` — no external backtest
framework.  The backtest is deliberately simple: equal-weighted long and
short books, positions entered at the close of the disclosure week and held
for exactly ``POST_DISCLOSURE_WINDOW`` trading days.

Usage
-----
    python -m src.backtest

Runs an end-to-end demo using synthetic data and prints performance metrics.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


# ── Return computation helpers ────────────────────────────────────────────────

def _get_return_for_position(
    ticker: str,
    entry_date: pd.Timestamp,
    price_data: pd.DataFrame,
    window: int = config.POST_DISCLOSURE_WINDOW,
) -> float | None:
    """
    Compute the arithmetic return for *ticker* held for *window* trading days
    starting from *entry_date*.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    entry_date : pd.Timestamp
        First trading day after the ISO week ends (i.e. Monday of the
        following week, adjusted for market holidays by choosing the next
        available price).
    price_data : pd.DataFrame
        Wide-format adjusted close prices indexed by date.
    window : int
        Number of trading days to hold.

    Returns
    -------
    float or None
        ``(P_{t+window} / P_t) - 1`` if sufficient data exist; otherwise
        ``None`` (position is excluded from that week's return calculation).
    """
    if ticker not in price_data.columns:
        return None

    prices = price_data[ticker].dropna()
    future = prices[prices.index >= entry_date]

    if len(future) < window + 1:
        return None

    p0 = future.iloc[0]
    pn = future.iloc[window]

    if p0 == 0 or np.isnan(p0):
        return None

    return float(pn / p0 - 1.0)


def _week_to_entry_date(week_str: str) -> pd.Timestamp:
    """
    Convert an ISO week string (``"YYYY-Www"``) to the Monday of that week.

    Parameters
    ----------
    week_str : str
        ISO week label, e.g. ``"2023-W14"``.

    Returns
    -------
    pd.Timestamp
        Monday of the specified ISO week.
    """
    year, week = week_str.split("-W")
    return pd.Timestamp.fromisocalendar(int(year), int(week), 1)  # Monday = day 1


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    positions_df: pd.DataFrame,
    price_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run a weekly long/short backtest over all positions in *positions_df*.

    Parameters
    ----------
    positions_df : pd.DataFrame
        Output of :func:`src.signal.generate_weekly_positions`.  Required
        columns: ``[week, ticker, position]``.
    price_data : pd.DataFrame
        Wide-format adjusted close prices indexed by date.

    Returns
    -------
    pd.DataFrame
        Weekly PnL DataFrame with columns:

        ``week``
            ISO week label.
        ``long_return``
            Equal-weighted mean return of the LONG book (NaN if no longs).
        ``short_return``
            Equal-weighted mean return of the SHORT book (NaN if no shorts),
            **with sign flipped** so it represents the profit from a short
            position.
        ``ls_return``
            ``long_return - short_return`` (before sign flip of short).
            Represents the gross long-short return for the week.
        ``cumulative_pnl``
            Cumulative sum of ``ls_return`` (arithmetic, not compounded).

    Notes
    -----
    **Entry timing**: positions are entered at the close of the disclosure
    week's last trading day (approximated as the Monday of the *following*
    ISO week) and held for ``POST_DISCLOSURE_WINDOW`` trading days.

    **Weighting**: equal weights within each book; no leverage; no
    transaction cost adjustment (treat as gross alpha).

    **Short convention**: ``short_return`` in the raw output is the return
    of the shorted stock multiplied by -1.  The ``ls_return`` column is
    ``long_book_return + (-1 × short_book_return)``, i.e. profit if the
    short side falls.

    **Caveats**: this backtest does not account for:
    - Bid/ask spreads or market impact
    - Short-selling constraints or borrow costs
    - Delisting survivorship bias in price data
    - Tax lots or wash-sale rules
    """
    if positions_df.empty:
        logger.warning("positions_df is empty — returning empty results.")
        return pd.DataFrame(
            columns=["week", "long_return", "short_return", "ls_return", "cumulative_pnl"]
        )

    weeks = sorted(positions_df["week"].unique())
    logger.info("Running backtest over %d weeks...", len(weeks))

    records = []
    for week in weeks:
        week_df = positions_df[positions_df["week"] == week]

        # Entry on Monday of the *next* ISO week
        # Monday of this ISO week + 7 days (handles 53-week ISO years)
        entry_date = _week_to_entry_date(week) + pd.Timedelta(days=7)

        long_tickers = week_df[week_df["position"] == "LONG"]["ticker"].tolist()
        short_tickers = week_df[week_df["position"] == "SHORT"]["ticker"].tolist()

        long_returns = [
            r for t in long_tickers
            if (r := _get_return_for_position(t, entry_date, price_data)) is not None
        ]
        short_returns = [
            r for t in short_tickers
            if (r := _get_return_for_position(t, entry_date, price_data)) is not None
        ]

        long_ret = float(np.mean(long_returns)) if long_returns else np.nan
        short_ret = float(np.mean(short_returns)) if short_returns else np.nan

        # Long-short: profit from long side + profit from being short (flip sign)
        if np.isnan(long_ret) and np.isnan(short_ret):
            ls_ret = np.nan
        elif np.isnan(long_ret):
            ls_ret = -short_ret  # only short book active
        elif np.isnan(short_ret):
            ls_ret = long_ret    # only long book active
        else:
            ls_ret = long_ret - short_ret

        records.append(
            {
                "week": week,
                "long_return": long_ret,
                "short_return": short_ret,
                "ls_return": ls_ret,
                "n_long": len(long_returns),
                "n_short": len(short_returns),
            }
        )

    results_df = pd.DataFrame(records)
    results_df = results_df.dropna(subset=["ls_return"])
    results_df["cumulative_pnl"] = results_df["ls_return"].cumsum()

    results_df.to_csv(config.RESULTS_CSV, index=False)
    logger.info("Backtest results saved to %s", config.RESULTS_CSV)

    return results_df


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_performance_metrics(results_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute standard performance statistics for the long-short strategy.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of :func:`run_backtest`.  Must contain columns:
        ``ls_return`` and ``cumulative_pnl``.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:

        ``sharpe_ratio``
            Annualised Sharpe Ratio assuming 52 independent weekly
            observations per year.  Mean / StdDev × √52.
        ``max_drawdown``
            Maximum peak-to-trough decline of ``cumulative_pnl``
            (negative number; more negative = worse).
        ``win_rate``
            Fraction of weeks with ``ls_return > 0``.
        ``n_trades``
            Total count of trade legs (long + short positions initiated).
        ``avg_holding_return``
            Mean ``ls_return`` across all active weeks.

    Notes
    -----
    The Sharpe Ratio uses the weekly return series without a risk-free-rate
    adjustment (i.e. excess return ≈ raw return) for simplicity.  Multiply
    the weekly mean by 52 and the weekly standard deviation by √52 to
    annualise, which gives ``mean / std × √52``.

    Maximum drawdown is computed on the cumulative arithmetic PnL series
    (not log-returns or NAV), which is the most interpretable metric for a
    dollar-neutral strategy where PnL is measured in return points.
    """
    if results_df.empty or "ls_return" not in results_df.columns:
        logger.warning("Empty or malformed results_df — returning zero metrics.")
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
            "avg_holding_return": 0.0,
        }

    ls = results_df["ls_return"].dropna()
    cum_pnl = results_df["cumulative_pnl"].dropna()

    # Sharpe
    mean_ret = ls.mean()
    std_ret = ls.std(ddof=1)
    sharpe = float(mean_ret / std_ret * np.sqrt(52)) if std_ret > 0 else 0.0

    # Max drawdown
    rolling_max = cum_pnl.cummax()
    drawdowns = cum_pnl - rolling_max
    max_dd = float(drawdowns.min())

    # Win rate
    win_rate = float((ls > 0).mean())

    # Trade count
    n_trades = int(results_df[["n_long", "n_short"]].sum().sum()) if {
        "n_long", "n_short"
    } <= set(results_df.columns) else len(results_df) * 2

    # Average holding return
    avg_ret = float(ls.mean())

    metrics = {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "avg_holding_return": avg_ret,
    }

    # Log formatted table
    logger.info("\n" + "=" * 45)
    logger.info("PERFORMANCE METRICS")
    logger.info("=" * 45)
    logger.info("  Annualised Sharpe Ratio : %+.4f", sharpe)
    logger.info("  Maximum Drawdown        : %+.4f", max_dd)
    logger.info("  Win Rate                : %.1f%%", win_rate * 100)
    logger.info("  Total Trades            : %d", n_trades)
    logger.info("  Avg Holding Return      : %+.4f", avg_ret)
    logger.info("=" * 45)

    # Persist
    pd.DataFrame([metrics]).to_csv(config.METRICS_CSV, index=False)
    logger.info("Metrics saved to %s", config.METRICS_CSV)

    return metrics


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logger.info("=== Backtest module demo (synthetic data) ===")

    rng = np.random.default_rng(0)

    # Synthetic price data: 5 tickers, 2 years of daily prices
    tickers = ["AAPL", "MSFT", "LMT", "RTX", "GS"]
    price_dates = pd.date_range("2021-01-01", periods=520, freq="B")
    price_data = pd.DataFrame(
        np.cumprod(1 + rng.normal(0.0005, 0.015, size=(520, len(tickers))), axis=0) * 100,
        index=price_dates,
        columns=tickers,
    )

    # Synthetic positions
    weeks = [f"2021-W{w:02d}" for w in range(1, 41)]
    rows = []
    for week in weeks:
        for _ in range(3):
            rows.append(
                {
                    "week": week,
                    "ticker": rng.choice(tickers),
                    "position": rng.choice(["LONG", "SHORT"]),
                    "composite_score": float(rng.standard_normal()),
                }
            )
    positions_df = pd.DataFrame(rows)

    # Run backtest
    results = run_backtest(positions_df, price_data)
    print("\nBacktest results sample:")
    print(results.head(10).to_string())

    # Compute metrics
    metrics = compute_performance_metrics(results)
    print("\nPerformance Metrics:")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.4f}")

    sys.exit(0)
