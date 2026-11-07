"""
src/features.py
===============
Feature engineering pipeline for the Congressional Trading Lag Arbitrage strategy.

Each function accepts the combined disclosures DataFrame (output of
:func:`src.scraper.combine_disclosures`) and returns an enriched copy.
No function mutates its input in-place.

Functions
---------
compute_disclosure_lag        : Derive `lag_days`; drop data-quality outliers.
compute_coverage_score        : Proxy analyst coverage via Finviz article counts.
compute_committee_weight      : Map members to committee influence weights.
compute_direction_flag        : Encode buy/sell as +1/-1.
compute_composite_score       : Combine features into the ranking signal.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

# ── Disclosure lag ────────────────────────────────────────────────────────────

def compute_disclosure_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the number of calendar days between transaction and disclosure.

    Parameters
    ----------
    df : pd.DataFrame
        Combined disclosures DataFrame with ``transaction_date`` and
        ``disclosure_date`` columns (both ``datetime64``).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added integer column ``lag_days``.  Rows
        where ``lag_days > MAX_LEGAL_LAG`` (45 days, per the STOCK Act) are
        flagged as data errors and dropped.

    Notes
    -----
    The STOCK Act (2012) requires members of Congress to disclose trades
    within 45 calendar days.  Any row with ``lag_days > 45`` is almost
    certainly a data entry error in the source system (e.g. transposed
    year digits) and is removed before analysis.  Rows with negative lag
    (disclosure precedes transaction, also impossible) are likewise dropped.
    """
    df = df.copy()
    df["lag_days"] = (
        df["disclosure_date"] - df["transaction_date"]
    ).dt.days

    before = len(df)
    df = df[df["lag_days"].between(0, config.MAX_LEGAL_LAG)]
    after = len(df)
    dropped = before - after
    if dropped > 0:
        logger.warning(
            "Dropped %d rows with lag_days outside [0, %d]", dropped, config.MAX_LEGAL_LAG
        )
    else:
        logger.info("All rows have valid lag_days (0 – %d).", config.MAX_LEGAL_LAG)

    return df.reset_index(drop=True)


# ── Analyst coverage score ────────────────────────────────────────────────────

_FINVIZ_NEWS_URL = "https://finviz.com/quote.ashx?t={ticker}&ty=c&ta=1&p=d"
_FINVIZ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def _count_finviz_articles(ticker: str) -> int:
    """
    Scrape the Finviz news tab and count the number of article rows.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    int
        Number of news article rows found on the Finviz page.  Returns 0 on
        any network or parsing failure.

    Notes
    -----
    Finviz renders all headlines on a single HTML page without pagination, so
    a simple row count of the ``<tr>`` elements inside the news table gives a
    rough proxy for analyst/media coverage intensity.

    Be courteous to Finviz's servers: callers should sleep between requests.
    This function does NOT sleep; throttling is handled by the caller
    :func:`compute_coverage_score`.
    """
    url = _FINVIZ_NEWS_URL.format(ticker=ticker.upper())
    try:
        resp = requests.get(
            url,
            headers=_FINVIZ_HEADERS,
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("Finviz request failed for %s: %s", ticker, exc)
        return 0

    soup = BeautifulSoup(resp.text, "lxml")
    # Finviz news table has id="news-table" in most versions
    news_table = soup.find("table", id="news-table")
    if news_table is None:
        # Fallback: try to find the table near the word "news"
        tables = soup.find_all("table", class_=lambda c: c and "news" in c.lower())
        news_table = tables[0] if tables else None

    if news_table is None:
        return 0

    rows = news_table.find_all("tr")
    return len(rows)


def compute_coverage_score(
    df: pd.DataFrame,
    tickers: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Proxy analyst coverage by counting recent Finviz news articles per ticker.

    Parameters
    ----------
    df : pd.DataFrame
        Combined disclosures DataFrame with a ``ticker`` column.
    tickers : list[str], optional
        Subset of tickers to score.  If ``None``, all unique tickers in *df*
        are scored (can be slow for large universes).

    Returns
    -------
    df_enriched : pd.DataFrame
        Input DataFrame with two new columns:

        ``coverage_score``
            Raw article count from Finviz (integer).
        ``low_coverage``
            Boolean flag; ``True`` if ``coverage_score`` is below the
            10th percentile of the scored universe.

    coverage_dict : dict[str, int]
        Mapping ``{ticker: article_count}`` for every ticker that was scored.

    Notes
    -----
    Finviz article count is used as a crude coverage proxy because:

    1. It is freely available without an API key.
    2. It correlates reasonably with analyst following (larger, more-covered
       firms have more news).

    A production implementation should replace this with IBES analyst count
    data or a Bloomberg field (``ANALYST_RECS_TOTAL``).

    A 1-second sleep is inserted between requests to avoid rate-limiting.
    Tickers that fail (HTTP error, timeout) receive a score of 0.
    """
    df = df.copy()

    if tickers is None:
        tickers = df["ticker"].dropna().unique().tolist()

    logger.info("Fetching Finviz coverage for %d unique tickers...", len(tickers))
    coverage_dict: dict[str, int] = {}

    for i, ticker in enumerate(tickers):
        count = _count_finviz_articles(ticker)
        coverage_dict[ticker] = count
        logger.debug("[%d/%d] %s → %d articles", i + 1, len(tickers), ticker, count)
        time.sleep(config.REQUEST_SLEEP)

    df["coverage_score"] = df["ticker"].map(coverage_dict).fillna(0).astype(int)

    p10 = df["coverage_score"].quantile(0.10)
    df["low_coverage"] = df["coverage_score"] < p10

    logger.info(
        "Coverage scores computed. Low-coverage rows: %d (< p10 = %.0f articles)",
        df["low_coverage"].sum(),
        p10,
    )
    return df, coverage_dict


# ── Committee weight ──────────────────────────────────────────────────────────

def compute_committee_weight(
    df: pd.DataFrame,
    committee_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Assign an influence weight based on the member's most powerful committee.

    Parameters
    ----------
    df : pd.DataFrame
        Combined disclosures DataFrame with a ``filer_name`` column.
    committee_csv_path : str, optional
        Path to a CSV with columns ``[filer_name, committee]`` that maps
        members to their committee assignments.  Defaults to
        ``config.COMMITTEE_CSV``.  If the file does not exist, all members
        receive the ``"other"`` weight (0.5).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with a new ``committee_weight`` column (float).

    Notes
    -----
    The committee assignment data is **not** available directly in the
    disclosure APIs.  The recommended workflow is:

    1. Download a current member–committee mapping from
       ``clerk.house.gov/xml/lists/MemberInfo.xml`` or
       ``senate.gov/general/committee_membership/committee_memberships_current.xml``.
    2. Save it as a CSV with columns ``[filer_name, committee]``.
    3. Pass the path via *committee_csv_path* or update ``config.COMMITTEE_CSV``.

    A placeholder CSV with 10 example rows is provided in ``data/`` for
    reference.

    The weight mapping (from ``config.COMMITTEE_WEIGHTS``) reflects the
    degree to which each committee oversees publicly traded sectors:

    +-----------------+--------+
    | Committee       | Weight |
    +=================+========+
    | Armed Services  |   1.00 |
    | Intelligence    |   1.00 |
    | Appropriations  |   0.85 |
    | Finance         |   0.75 |
    | Energy          |   0.70 |
    | other           |   0.50 |
    +-----------------+--------+
    """
    df = df.copy()

    csv_path = committee_csv_path or str(config.COMMITTEE_CSV)

    # Default weight for all members
    df["committee_weight"] = config.COMMITTEE_WEIGHTS.get("other", 0.5)

    try:
        committee_df = pd.read_csv(csv_path)
        if "filer_name" not in committee_df.columns or "committee" not in committee_df.columns:
            logger.warning(
                "Committee CSV at %s is missing required columns [filer_name, committee]. "
                "Defaulting all weights to 0.5.",
                csv_path,
            )
            return df

        # Build a lookup: for each member, take the highest-weight committee
        committee_df["weight"] = committee_df["committee"].map(
            lambda c: config.COMMITTEE_WEIGHTS.get(c, config.COMMITTEE_WEIGHTS["other"])
        )
        best_weight = (
            committee_df.groupby("filer_name")["weight"].max().to_dict()
        )

        df["committee_weight"] = df["filer_name"].map(best_weight).fillna(
            config.COMMITTEE_WEIGHTS["other"]
        )
        logger.info(
            "Committee weights applied from %s. "
            "Unique weights in data: %s",
            csv_path,
            df["committee_weight"].value_counts().to_dict(),
        )
    except FileNotFoundError:
        logger.info(
            "Committee CSV not found at %s — using default weight %.2f for all members.",
            csv_path,
            config.COMMITTEE_WEIGHTS["other"],
        )

    return df


# ── Direction flag ────────────────────────────────────────────────────────────

def compute_direction_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode transaction direction as a numeric sign: +1 for buy, -1 for sell.

    Parameters
    ----------
    df : pd.DataFrame
        Combined disclosures DataFrame with a ``transaction_type`` column
        containing lowercase strings (``"buy"`` or ``"sell"``).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with a new ``direction`` column (int: +1 or -1).
        Rows with an unrecognised transaction type receive ``direction = 0``
        and a warning is logged.

    Notes
    -----
    Direction is the primary sign of the trading signal: a Congressional buy
    is treated as a bullish signal and a sell as bearish.  The value of ±1
    ensures the composite score formula preserves sign.
    """
    df = df.copy()
    direction_map = {"buy": 1, "purchase": 1, "sell": -1, "sale": -1}
    df["direction"] = df["transaction_type"].str.lower().map(direction_map)

    unknown = df["direction"].isna().sum()
    if unknown > 0:
        logger.warning(
            "%d rows have unrecognised transaction_type — direction set to 0.", unknown
        )
    df["direction"] = df["direction"].fillna(0).astype(int)
    return df


# ── Composite score ───────────────────────────────────────────────────────────

def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the ranking composite signal for each disclosure event.

    Parameters
    ----------
    df : pd.DataFrame
        Disclosures DataFrame enriched with columns:
        ``lag_days``, ``committee_weight``, ``direction``, ``coverage_score``.
        All four must be present; call the preceding feature functions first.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with a new ``composite_score`` column (float).

    Notes
    -----
    **Formula**::

        composite = direction
                    × log(1 + lag_days)
                    × committee_weight
                    × 1 / (1 + coverage_score)

    **Rationale for each term**:

    * ``direction`` (±1) — preserves the sign: buys score positively
      (LONG signal) and sells score negatively (SHORT signal).
    * ``log(1 + lag_days)`` — dampens the influence of outlier lags while
      rewarding trades that approach the 45-day filing deadline, which
      suggests the member may be trying to delay a particularly informative
      disclosure.  Log is bounded and grows slowly, preventing a 44-day lag
      from overwhelming a 10-day lag by a factor of 4.4×.
    * ``committee_weight`` — amplifies signals from members with legislative
      influence over the sector in which the traded company operates.
    * ``1 / (1 + coverage_score)`` — rewards purchases in *obscure* names
      with low analyst coverage, where informational asymmetry between a
      well-connected legislator and the public market is largest.
      High-coverage mega-caps (``coverage_score >> 0``) are discounted
      toward zero.

    A positive composite score → LONG candidate.
    A negative composite score → SHORT candidate.
    """
    required = {"direction", "lag_days", "committee_weight", "coverage_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            "Run compute_direction_flag, compute_disclosure_lag, "
            "compute_committee_weight, and compute_coverage_score first."
        )

    df = df.copy()
    df["composite_score"] = (
        df["direction"]
        * np.log1p(df["lag_days"])
        * df["committee_weight"]
        * (1.0 / (1.0 + df["coverage_score"]))
    )
    logger.info(
        "Composite scores computed. Range: [%.4f, %.4f]",
        df["composite_score"].min(),
        df["composite_score"].max(),
    )
    return df
