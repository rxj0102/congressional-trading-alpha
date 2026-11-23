"""Download what the event study needs and freeze it.

    python scripts/fetch_prices.py

Reads data/house_ptr_transactions.csv, downloads split- and dividend-adjusted daily
closes from Yahoo Finance (yfinance) for every traded ticker, and the Fama-French
daily factors from the Ken French Data Library. Writes data/prices.parquet (not
committed, it is large) and data/ff3_daily.csv. The event study then freezes the
per-event returns in data/snapshot/, which is committed.
"""

from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FF_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


def yahoo_symbol(t: str) -> str:
    return t.replace(".", "-")


def fama_french() -> pd.DataFrame:
    r = requests.get(FF_URL, headers={"User-Agent": "curl/8"}, timeout=60)
    r.raise_for_status()
    raw = zipfile.ZipFile(io.BytesIO(r.content)).read(
        "F-F_Research_Data_Factors_daily.csv"
    ).decode()
    rows = [
        line.split(",")
        for line in raw.splitlines()
        if line[:8].strip().isdigit() and len(line.split(",")) == 5
    ]
    df = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rf"])
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%Y%m%d")
    for c in ("mkt_rf", "smb", "hml", "rf"):
        df[c] = df[c].astype(float) / 100.0
    return df


def main() -> None:
    import yfinance as yf

    tx = pd.read_csv(ROOT / "data" / "house_ptr_transactions.csv")
    from src.event_study import load_events

    tickers = sorted(load_events()["ticker"].unique())
    out = ROOT / "data" / "prices.parquet"
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    frames = [have] if not have.empty else []
    missing = [t for t in tickers if t not in have.columns]
    # Yahoo rate limits large runs; fetch the missing tickers in small paced batches and
    # retry the ones that come back empty, so only genuinely unavailable names remain.
    for attempt in range(3):
        if not missing:
            break
        got = []
        for i in range(0, len(missing), 40):
            batch = [yahoo_symbol(t) for t in missing[i : i + 40]]
            px = yf.download(batch, start="2012-12-01", auto_adjust=True, progress=False, threads=False)["Close"]
            if isinstance(px, pd.Series):
                px = px.to_frame(batch[0])
            px = px.dropna(axis=1, how="all")
            px.columns = [c.replace("-", ".") for c in px.columns]
            frames.append(px)
            got += list(px.columns)
            time.sleep(3 + 10 * attempt)
        missing = [t for t in missing if t not in set(got)]
        merged = pd.concat(frames, axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()].dropna(axis=1, how="all")
        merged.to_parquet(out)  # save after every pass so an interrupted run keeps its progress
        frames = [merged]
        print(f"pass {attempt + 1}: {len(missing)} still missing", flush=True)
        if got == [] and attempt > 0:
            break  # nothing more is coming back: the rest are not on Yahoo
    prices = pd.concat(frames, axis=1)
    prices.columns = [c.replace("-", ".") for c in prices.columns]
    prices = prices.loc[:, ~prices.columns.duplicated()].dropna(axis=1, how="all")
    prices.to_parquet(out)
    fama_french().to_csv(ROOT / "data" / "ff3_daily.csv", index=False)
    print(f"{prices.shape[1]} of {len(tickers)} tickers have prices")


if __name__ == "__main__":
    main()
