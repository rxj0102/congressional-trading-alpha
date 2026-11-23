"""
Do House members' disclosed stock trades carry information an outsider can still use?

Rules fixed before the first run on the rebuilt data:

* Events: House periodic transaction reports, electronic filings, 2015 onward (the
  first year the Clerk's index lists them), purchases and sales (exchanges dropped) of
  assets coded [ST] (stock), or, on the older forms that carry no asset codes, any asset
  with a ticker whose description does not mention options; a ticker with Yahoo prices;
  a filing date on or after the transaction date and within 365 days of it.
* Public date: the report's filing date in the Clerk's index. Entry is the close of
  the first trading day after it, so a filing made late in the day is not traded
  before it could be seen.
* Abnormal return: log return of the stock minus log return of the market (Fama-French
  Mkt-RF plus RF), summed over the window.
* Before disclosure (not tradable): from the close of the first trading day on or after
  the transaction date to the close of the last trading day on or before the filing
  date. This is what members earned before anyone could copy them.
* After disclosure (tradable): 5, 20 and 60 trading days from entry. Purchases minus
  sales, with standard errors clustered by filing date.
* Calendar-time portfolio: each day, long every stock bought in the last 20 trading
  days after its entry and short every stock sold, equal weights; daily alpha against
  the market and against Fama-French three factors, Newey-West 5 lags.
* The original hypothesis of this repository: late filings (more than 45 days after
  the trade, the STOCK Act limit) carry more information. Tested as the difference in
  signed 20 day abnormal return (purchases positive, sales negative) between late and
  on-time filings.
* The six tradable tests are corrected together with Holm.

    python -m src.event_study build      # needs data/prices.parquet, writes data/snapshot/events.csv.gz
    python -m src.event_study analyze    # writes results/metrics.json from the snapshot
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
TX = ROOT / "data" / "house_ptr_transactions.csv"
PRICES = ROOT / "data" / "prices.parquet"
FF = ROOT / "data" / "ff3_daily.csv"
SNAP = ROOT / "data" / "snapshot" / "events.csv.gz"
METRICS = ROOT / "results" / "metrics.json"
WINDOWS = (5, 20, 60)
HOLD = 20
FIRST_YEAR = 2015  # the Clerk's yearly indexes list no transaction reports before 2015
LATE_DAYS = 45


def load_events() -> pd.DataFrame:
    tx = pd.read_csv(TX, parse_dates=["transaction_date", "notification_date", "filing_date"])
    n_all = len(tx)
    stock = (tx["asset_type"] == "ST") | (tx["asset_type"].isna() & ~tx["option"].astype(bool))
    ev = tx[stock & tx["ticker"].notna() & tx["transaction_type"].isin(["P", "S"])]
    ev = ev.dropna(subset=["transaction_date", "filing_date"])
    ev = ev[ev["filing_date"].dt.year >= FIRST_YEAR]
    lag = (ev["filing_date"] - ev["transaction_date"]).dt.days
    ev = ev[(lag >= 0) & (lag <= 365)].copy()
    ev["lag_days"] = (ev["filing_date"] - ev["transaction_date"]).dt.days
    ev.attrs["n_rows_parsed"] = n_all
    return ev


def build() -> None:
    ev = load_events()
    px = pd.read_parquet(PRICES).sort_index()
    px.index = pd.to_datetime(px.index)
    ff = pd.read_csv(FF, parse_dates=["date"]).set_index("date")
    days = px.index.intersection(ff.index)
    px, ff = px.loc[days], ff.loc[days]
    logp = np.log(px)
    lm = np.log1p(ff["mkt_rf"] + ff["rf"]).cumsum()  # cumulative log market
    ret = px.pct_change()
    rows = []
    for e in ev.itertuples(index=False):
        if e.ticker not in px.columns:
            continue
        s = logp[e.ticker]
        i_tx = days.searchsorted(e.transaction_date, "left")
        i_file = days.searchsorted(e.filing_date, "right") - 1
        i_in = days.searchsorted(e.filing_date, "right")
        if i_tx >= len(days) or i_in >= len(days) or i_file < i_tx:
            continue
        rec = {
            "doc_id": e.doc_id,
            "filer_name": e.filer_name,
            "ticker": e.ticker,
            "type": e.transaction_type,
            "amount_low": e.amount_low,
            "transaction_date": e.transaction_date.date(),
            "filing_date": e.filing_date.date(),
            "entry_date": days[i_in].date(),
            "lag_days": e.lag_days,
        }

        def car(a: int, b: int) -> float:
            if b >= len(days):
                return np.nan
            return float((s.iloc[b] - s.iloc[a]) - (lm.iloc[b] - lm.iloc[a]))

        rec["car_pre"] = car(i_tx, i_file)
        for h in WINDOWS:
            rec[f"car_{h}"] = car(i_in, i_in + h)
        path = ret[e.ticker].iloc[i_in + 1 : i_in + 1 + HOLD].to_numpy()
        for k in range(HOLD):
            rec[f"r{k + 1}"] = float(path[k]) if k < len(path) else np.nan
        if np.isfinite(rec["car_5"]):
            rows.append(rec)
    out = pd.DataFrame(rows)
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SNAP, index=False, float_format="%.6g")
    meta = {
        "rows_parsed": int(ev.attrs["n_rows_parsed"]),
        "stock_events": int(len(ev)),
        "events_with_prices": int(len(out)),
        "tickers": int(out["ticker"].nunique()),
    }
    (SNAP.parent / "build_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(meta)


def clustered(y: pd.Series, x: pd.DataFrame, groups: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(y, sm.add_constant(x)).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})


def rounded(x):
    """Six significant digits, so the file is identical across machines and library builds."""
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.6g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def holm(p: dict[str, float]) -> dict[str, float]:
    keys = sorted(p, key=p.get)
    m, run, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def calendar_portfolio(ev: pd.DataFrame, ff: pd.DataFrame) -> dict:
    days = ff.index
    long_, short_ = {}, {}
    for e in ev.itertuples(index=False):
        i = days.searchsorted(pd.Timestamp(e.entry_date), "left")
        book = long_ if e.type == "P" else short_
        for k in range(HOLD):
            r = getattr(e, f"r{k + 1}")
            if i + 1 + k < len(days) and np.isfinite(r):
                book.setdefault(days[i + 1 + k], []).append(r)
    lp = pd.Series({d: np.mean(v) for d, v in long_.items()}).sort_index()
    sp = pd.Series({d: np.mean(v) for d, v in short_.items()}).sort_index()
    df = pd.concat({"long": lp, "short": sp}, axis=1).dropna().join(ff, how="inner")
    df["ls"] = df["long"] - df["short"]
    df["long_ex"] = df["long"] - df["rf"]
    out = {"days": int(len(df)), "first_day": str(df.index.min().date()), "last_day": str(df.index.max().date())}
    for name, y in (("long_short", df["ls"]), ("long_only_excess", df["long_ex"])):
        for model, cols in (("capm", ["mkt_rf"]), ("ff3", ["mkt_rf", "smb", "hml"])):
            res = sm.OLS(y, sm.add_constant(df[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            out[f"{name}_{model}_alpha_annual"] = float(res.params["const"] * 252)
            out[f"{name}_{model}_t"] = float(res.tvalues["const"])
            out[f"{name}_{model}_p"] = float(res.pvalues["const"])
    out["long_short_sharpe"] = float(df["ls"].mean() / df["ls"].std() * np.sqrt(252))
    return out


def analyze() -> None:
    ev = pd.read_csv(SNAP, parse_dates=["transaction_date", "filing_date", "entry_date"])
    meta = json.loads((SNAP.parent / "build_meta.json").read_text())
    ff = pd.read_csv(FF, parse_dates=["date"]).set_index("date")
    ev["buy"] = (ev["type"] == "P").astype(float)
    ev["sign"] = np.where(ev["type"] == "P", 1.0, -1.0)
    m: dict = {"sample": meta | {
        "first_filing": str(ev["filing_date"].min().date()),
        "last_filing": str(ev["filing_date"].max().date()),
        "purchases": int((ev["type"] == "P").sum()),
        "sales": int((ev["type"] == "S").sum()),
        "members": int(ev["filer_name"].nunique()),
        "lag_median": float(ev["lag_days"].median()),
        "late_share": float((ev["lag_days"] > LATE_DAYS).mean()),
    }}
    windows = {"pre": "car_pre"} | {str(h): f"car_{h}" for h in WINDOWS}
    m["car"] = {}
    for name, col in windows.items():
        d = ev.dropna(subset=[col])
        r = clustered(d[col], d[["buy"]], d["filing_date"])
        m["car"][name] = {
            "n": int(len(d)),
            "buy_mean": float(d.loc[d["buy"] == 1, col].mean()),
            "sell_mean": float(d.loc[d["buy"] == 0, col].mean()),
            "diff": float(r.params["buy"]),
            "t": float(r.tvalues["buy"]),
            "p": float(r.pvalues["buy"]),
        }
    d = ev.dropna(subset=["car_20"]).copy()
    d["signed"] = d["sign"] * d["car_20"]
    d["late"] = (d["lag_days"] > LATE_DAYS).astype(float)
    r = clustered(d["signed"], d[["late"]], d["filing_date"])
    m["late"] = {
        "n_late": int(d["late"].sum()),
        "signed_on_time": float(d.loc[d["late"] == 0, "signed"].mean()),
        "signed_late": float(d.loc[d["late"] == 1, "signed"].mean()),
        "diff": float(r.params["late"]),
        "t": float(r.tvalues["late"]),
        "p": float(r.pvalues["late"]),
    }
    # robustness of the late filing result (reported, not part of the Holm family)
    lo, hi = d["signed"].quantile([0.01, 0.99])
    d["signed_w"] = d["signed"].clip(lo, hi)
    by_member = clustered(d["signed_w"], d[["late"]], d["filer_name"])
    top = d[d["late"] == 1].groupby("filer_name").size().nlargest(5).index
    rest = d[~d["filer_name"].isin(top)]
    without = clustered(rest["signed_w"], rest[["late"]], rest["filer_name"])
    m["late"]["robust"] = {
        "winsorized_member_cluster_diff": float(by_member.params["late"]),
        "winsorized_member_cluster_t": float(by_member.tvalues["late"]),
        "top5_late_filers_share": float(d[(d["late"] == 1) & d["filer_name"].isin(top)].shape[0] / d["late"].sum()),
        "without_top5_diff": float(without.params["late"]),
        "without_top5_t": float(without.tvalues["late"]),
    }
    # winsorized buy minus sell, to show the null does not hinge on extreme returns
    m["car_winsorized"] = {}
    for h in WINDOWS:
        dd = ev.dropna(subset=[f"car_{h}"]).copy()
        lo, hi = dd[f"car_{h}"].quantile([0.01, 0.99])
        r = clustered(dd[f"car_{h}"].clip(lo, hi), dd[["buy"]], dd["filing_date"])
        m["car_winsorized"][str(h)] = {"diff": float(r.params["buy"]), "t": float(r.tvalues["buy"])}
    m["calendar"] = calendar_portfolio(ev, ff)
    tests = {
        "buy minus sell, 5 days": m["car"]["5"]["p"],
        "buy minus sell, 20 days": m["car"]["20"]["p"],
        "buy minus sell, 60 days": m["car"]["60"]["p"],
        "long-short portfolio, CAPM alpha": m["calendar"]["long_short_capm_p"],
        "long-short portfolio, three-factor alpha": m["calendar"]["long_short_ff3_p"],
        "late minus on-time filings": m["late"]["p"],
    }
    m["holm"] = holm(tests)
    m["raw_p"] = tests
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(rounded(m), indent=1) + "\n")
    print(json.dumps(m, indent=1, default=float))


if __name__ == "__main__":
    {"build": build, "analyze": analyze}[sys.argv[1]]()
