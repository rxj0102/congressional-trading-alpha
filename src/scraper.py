"""
src/scraper.py
==============
Data acquisition layer for congressional stock disclosures and market prices.

Functions fetch from public APIs (housestockwatcher.com, senatestockwatcher.com)
and yfinance, with local JSON caching keyed by fetch date to avoid redundant
network calls on re-runs.

Usage
-----
    python -m src.scraper

This runs a quick integration test that prints the shape of each returned
DataFrame.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

# ── Amount-range midpoint lookup ──────────────────────────────────────────────
_AMOUNT_MIDPOINTS: dict[str, float] = {
    "$1,001 - $15,000":         8_000.5,
    "$15,001 - $50,000":       32_500.5,
    "$50,001 - $100,000":      75_000.5,
    "$100,001 - $250,000":    175_000.5,
    "$250,001 - $500,000":    375_000.5,
    "$500,001 - $1,000,000":  750_000.5,
    "$1,000,001 - $5,000,000": 3_000_000.5,
    "Over $5,000,000":         5_000_001.0,
    "Unknown":                       0.0,
}


def _parse_amount_midpoint(raw: str) -> float:
    """
    Map a STOCK Act amount-range string to its numeric midpoint.

    Parameters
    ----------
    raw : str
        Raw amount string from the disclosure, e.g. ``"$15,001 - $50,000"``.

    Returns
    -------
    float
        Midpoint dollar value; 0.0 if the string is unrecognised.

    Notes
    -----
    The mapping covers every bucket defined in the STOCK Act filing forms.
    Unrecognised strings fall back to 0.0 and trigger a debug-level log.
    """
    cleaned = raw.strip() if isinstance(raw, str) else "Unknown"
    if cleaned in _AMOUNT_MIDPOINTS:
        return _AMOUNT_MIDPOINTS[cleaned]
    # Try to extract numbers directly
    nums = re.findall(r"[\d,]+", cleaned)
    if len(nums) >= 2:
        lo = float(nums[0].replace(",", ""))
        hi = float(nums[1].replace(",", ""))
        return (lo + hi) / 2.0
    logger.debug("Unrecognised amount range: %r — defaulting to 0.0", raw)
    return 0.0


def _cache_path(template: Path, fetch_date: Optional[date] = None) -> Path:
    """Resolve a cache-file path by substituting today's date."""
    today = fetch_date or date.today()
    return Path(str(template).format(date=today.isoformat()))


def _load_cached(path: Path) -> Optional[list | dict]:
    """Return parsed JSON from *path*, or None if file does not exist."""
    if path.exists():
        logger.info("Loading cached data from %s", path)
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _save_cache(data: list | dict, path: Path) -> None:
    """Serialise *data* to *path* as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Cached response written to %s", path)


# ── House disclosures ─────────────────────────────────────────────────────────

def fetch_house_disclosures(fetch_date: Optional[date] = None) -> pd.DataFrame:
    """
    Retrieve House member stock disclosures from housestockwatcher.com.

    Results are cached as a JSON file in ``data/`` keyed by *fetch_date* so
    that repeated runs on the same day skip the network call.

    Parameters
    ----------
    fetch_date : date, optional
        The reference date used to name the cache file.  Defaults to today.

    Returns
    -------
    pd.DataFrame
        Columns: ``filer_name``, ``ticker``, ``transaction_date``,
        ``disclosure_date``, ``transaction_type``, ``amount_range``,
        ``district``, ``party``, ``chamber``.

    Raises
    ------
    requests.HTTPError
        Propagated after logging if the API returns a non-2xx status code.
    requests.Timeout
        Propagated after logging if the request times out.

    Notes
    -----
    The housestockwatcher.com API returns a list of disclosure objects.
    Rows with a null or empty ticker (e.g. mutual-fund trades) are dropped.
    """
    cache_path = _cache_path(config.HOUSE_CACHE, fetch_date)
    raw = _load_cached(cache_path)

    if raw is None:
        logger.info("Fetching House disclosures from %s", config.HOUSE_URL)
        try:
            resp = requests.get(
                config.HOUSE_URL,
                timeout=config.REQUEST_TIMEOUT,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            raw = resp.json()
            _save_cache(raw, cache_path)
        except requests.Timeout:
            logger.error("Timeout fetching House disclosures from %s", config.HOUSE_URL)
            raise
        except requests.HTTPError as exc:
            logger.error("HTTP error fetching House disclosures: %s", exc)
            raise
        time.sleep(config.REQUEST_SLEEP)

    if not raw:
        logger.warning("House disclosures API returned an empty payload.")
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df = _clean_house(df)
    logger.info("House disclosures loaded: %d rows", len(df))
    return df


def _clean_house(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise raw House API payload to the canonical column schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from the housestockwatcher.com JSON response.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardised column names and dtypes.
    """
    rename_map = {
        "representative":   "filer_name",
        "ticker":           "ticker",
        "transaction_date": "transaction_date",
        "disclosure_date":  "disclosure_date",
        "type":             "transaction_type",
        "amount":           "amount_range",
        "district":         "district",
        "party":            "party",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing)

    # Fill missing columns with sensible defaults
    for col in ["district", "party"]:
        if col not in df.columns:
            df[col] = "Unknown"

    df["chamber"] = "House"

    # Drop rows without a tradeable ticker
    df = df[df["ticker"].notna() & (df["ticker"].str.strip() != "")]
    df["ticker"] = df["ticker"].str.upper().str.strip()

    # Normalise transaction type
    if "transaction_type" in df.columns:
        df["transaction_type"] = (
            df["transaction_type"]
            .str.lower()
            .str.strip()
            .map(lambda x: "buy" if "purchase" in x or "buy" in x else
                           "sell" if "sale" in x or "sell" in x else x)
        )

    # Parse dates
    for col in ["transaction_date", "disclosure_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Amount midpoint
    if "amount_range" in df.columns:
        df["amount_midpoint"] = df["amount_range"].apply(_parse_amount_midpoint)
    else:
        df["amount_range"] = "Unknown"
        df["amount_midpoint"] = 0.0

    return df[
        ["filer_name", "ticker", "transaction_date", "disclosure_date",
         "transaction_type", "amount_range", "amount_midpoint",
         "district", "party", "chamber"]
    ]


# ── Senate disclosures ────────────────────────────────────────────────────────

def fetch_senate_disclosures(fetch_date: Optional[date] = None) -> pd.DataFrame:
    """
    Retrieve Senate member stock disclosures from senatestockwatcher.com.

    Results are cached identically to the House endpoint.

    Parameters
    ----------
    fetch_date : date, optional
        Reference date for the cache filename; defaults to today.

    Returns
    -------
    pd.DataFrame
        Same schema as :func:`fetch_house_disclosures`.

    Raises
    ------
    requests.HTTPError
        Propagated after logging if the API returns a non-2xx status.
    requests.Timeout
        Propagated after logging on network timeout.

    Notes
    -----
    The senatestockwatcher.com JSON schema differs slightly from the House
    endpoint; ``_clean_senate`` handles the field-name divergence.
    """
    cache_path = _cache_path(config.SENATE_CACHE, fetch_date)
    raw = _load_cached(cache_path)

    if raw is None:
        logger.info("Fetching Senate disclosures from %s", config.SENATE_URL)
        try:
            resp = requests.get(
                config.SENATE_URL,
                timeout=config.REQUEST_TIMEOUT,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            raw = resp.json()
            _save_cache(raw, cache_path)
        except requests.Timeout:
            logger.error("Timeout fetching Senate disclosures from %s", config.SENATE_URL)
            raise
        except requests.HTTPError as exc:
            logger.error("HTTP error fetching Senate disclosures: %s", exc)
            raise
        time.sleep(config.REQUEST_SLEEP)

    if not raw:
        logger.warning("Senate disclosures API returned an empty payload.")
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df = _clean_senate(df)
    logger.info("Senate disclosures loaded: %d rows", len(df))
    return df


def _clean_senate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise raw Senate API payload to the canonical column schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from the senatestockwatcher.com JSON response.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardised column names and dtypes.
    """
    rename_map = {
        "senator":          "filer_name",
        "ticker":           "ticker",
        "transaction_date": "transaction_date",
        "disclosure_date":  "disclosure_date",
        "type":             "transaction_type",
        "amount":           "amount_range",
        "state":            "district",
        "party":            "party",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing)

    for col in ["district", "party"]:
        if col not in df.columns:
            df[col] = "Unknown"

    df["chamber"] = "Senate"

    df = df[df["ticker"].notna() & (df["ticker"].str.strip() != "")]
    df["ticker"] = df["ticker"].str.upper().str.strip()

    if "transaction_type" in df.columns:
        df["transaction_type"] = (
            df["transaction_type"]
            .str.lower()
            .str.strip()
            .map(lambda x: "buy" if "purchase" in x or "buy" in x else
                           "sell" if "sale" in x or "sell" in x else x)
        )

    for col in ["transaction_date", "disclosure_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "amount_range" in df.columns:
        df["amount_midpoint"] = df["amount_range"].apply(_parse_amount_midpoint)
    else:
        df["amount_range"] = "Unknown"
        df["amount_midpoint"] = 0.0

    return df[
        ["filer_name", "ticker", "transaction_date", "disclosure_date",
         "transaction_type", "amount_range", "amount_midpoint",
         "district", "party", "chamber"]
    ]


# ── Combined disclosures ──────────────────────────────────────────────────────

def combine_disclosures(
    house_df: pd.DataFrame,
    senate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge House and Senate disclosures into a single, deduplicated DataFrame.

    Parameters
    ----------
    house_df : pd.DataFrame
        Output of :func:`fetch_house_disclosures`.
    senate_df : pd.DataFrame
        Output of :func:`fetch_senate_disclosures`.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame sorted by ``disclosure_date`` descending, with a
        ``amount_midpoint`` column derived from ``amount_range`` strings.

    Notes
    -----
    Deduplication key: ``(filer_name, ticker, transaction_date,
    transaction_type)``.  When both chambers report the same trade (rare but
    possible for members who serve in both capacities during transitions),
    the House record is preferred.
    """
    frames = [df for df in [house_df, senate_df] if df is not None and len(df) > 0]
    if not frames:
        logger.warning("Both House and Senate DataFrames are empty.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    dedup_keys = ["filer_name", "ticker", "transaction_date", "transaction_type"]
    combined = combined.drop_duplicates(subset=dedup_keys, keep="first")

    # Drop rows with unparseable dates in critical columns
    combined = combined.dropna(subset=["transaction_date", "disclosure_date"])

    combined = combined.sort_values("disclosure_date", ascending=False)
    combined = combined.reset_index(drop=True)

    logger.info(
        "Combined disclosures: %d rows (%d House, %d Senate)",
        len(combined),
        (combined["chamber"] == "House").sum(),
        (combined["chamber"] == "Senate").sum(),
    )
    return combined


# ── Price data ────────────────────────────────────────────────────────────────

def fetch_price_data(
    tickers: list[str],
    start_date: str | date,
    end_date: str | date,
    fetch_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Download adjusted daily close prices for a list of tickers via yfinance.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to download (e.g. ``["AAPL", "MSFT"]``).
    start_date : str or date
        Earliest date for price data (inclusive), ISO format or ``date``.
    end_date : str or date
        Latest date for price data (inclusive).
    fetch_date : date, optional
        Reference date for the cache filename; defaults to today.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame indexed by ``datetime`` with one column per
        ticker.  Tickers for which no price data could be retrieved are
        silently omitted.

    Notes
    -----
    Raw price data are cached as JSON so that the notebook can be re-run
    without hitting yfinance repeatedly.  The JSON is stored as a records
    dict with ISO date keys for portability.

    The ``start_date`` passed to yfinance should pre-date the earliest
    disclosure by at least ``POST_DISCLOSURE_WINDOW`` trading days to ensure
    forward-return computation is possible for every row.
    """
    cache_path = _cache_path(config.PRICE_CACHE, fetch_date)
    cache_key = f"{start_date}_{end_date}_{'_'.join(sorted(tickers))}"

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached.get("key") == cache_key:
            logger.info("Loading cached price data from %s", cache_path)
            records = cached["data"]
            df = pd.DataFrame(records)
            df.index = pd.to_datetime(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index
            return df

    logger.info(
        "Fetching price data for %d tickers (%s → %s)", len(tickers), start_date, end_date
    )

    if not tickers:
        return pd.DataFrame()

    try:
        raw = yf.download(
            tickers=tickers,
            start=str(start_date),
            end=str(end_date),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:  # yfinance raises plain Exception on hard failures
        logger.error("yfinance download failed: %s", exc)
        raise

    # yfinance returns MultiIndex columns when len(tickers) > 1
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]}) if len(tickers) == 1 else raw

    prices = prices.dropna(how="all")

    # Cache result
    cache_payload = {
        "key": cache_key,
        "data": prices.reset_index().rename(columns={"Date": "date"})
                      .set_index("date")
                      .to_dict(orient="index"),
    }
    _save_cache(cache_payload, cache_path)
    logger.info("Price data cached: %s rows × %s columns", len(prices), prices.shape[1])
    return prices


# ── CLI integration test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logger.info("=== Scraper integration test ===")

    try:
        house_df = fetch_house_disclosures()
        logger.info("House shape: %s", house_df.shape)
        print("\nHouse sample:")
        print(house_df.head(3).to_string())
    except Exception as exc:
        logger.error("House fetch failed: %s", exc)

    try:
        senate_df = fetch_senate_disclosures()
        logger.info("Senate shape: %s", senate_df.shape)
        print("\nSenate sample:")
        print(senate_df.head(3).to_string())
    except Exception as exc:
        logger.error("Senate fetch failed: %s", exc)
        senate_df = pd.DataFrame()
        house_df = pd.DataFrame()

    combined = combine_disclosures(house_df, senate_df)
    logger.info("Combined shape: %s", combined.shape)

    sys.exit(0)
