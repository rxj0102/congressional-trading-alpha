"""
Senate periodic transaction reports from the Senate's eFD system (official source).

efdsearch.senate.gov serves reports only after its terms of use are accepted. They
restrict how the reports may be used (no commercial or credit purposes, under the
Ethics in Government Act). The repository owner accepted them for this research; the
script asks for that acceptance explicitly (--accept-terms) and never sends it
otherwise.

Electronic reports are HTML tables (ticker, asset type, transaction type, date,
amount range). Paper reports are scanned images and are skipped, which is counted.
The public date of a transaction is the date the report was received, as listed in
the search results.

    python -m src.senate_efd 2015 2026 --accept-terms   # writes data/senate_ptr_transactions.csv
"""

from __future__ import annotations

import re
import sys
import time
from html import unescape
from pathlib import Path

import pandas as pd
import requests

BASE = "https://efdsearch.senate.gov"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "senate_ptr_transactions.csv"
INDEX_OUT = ROOT / "data" / "senate_ptr_index.csv"
RAW = ROOT / "data" / "senate_raw"
HEADERS = {"User-Agent": "academic research (congressional-trading-alpha)"}
TYPES = {"Purchase": "P", "Sale (Full)": "S", "Sale (Partial)": "S", "Exchange": "E"}
AMOUNT = re.compile(r"\$([\d,]+)")


def session(accept_terms: bool) -> requests.Session:
    if not accept_terms:
        raise SystemExit("eFD requires accepting its terms of use; rerun with --accept-terms if you accept them")
    s = requests.Session()
    s.headers.update(HEADERS)
    r = s.get(f"{BASE}/search/home/", timeout=30)
    tok = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text).group(1)
    s.post(
        f"{BASE}/search/home/",
        data={"prohibition_agreement": "1", "csrfmiddlewaretoken": tok},
        headers={"Referer": f"{BASE}/search/home/"},
        timeout=30,
    )
    return s


def search(s: requests.Session, year: int) -> pd.DataFrame:
    """Every periodic transaction report received in one year."""
    rows, start = [], 0
    while True:
        csrf = s.cookies.get("csrftoken")
        r = s.post(
            f"{BASE}/search/report/data/",
            data={
                "start": str(start), "length": "100", "report_types": "[11]", "filer_types": "[]",
                "submitted_start_date": f"01/01/{year} 00:00:00",
                "submitted_end_date": f"12/31/{year} 23:59:59",
                "candidate_state": "", "senator_state": "", "office_id": "",
                "first_name": "", "last_name": "", "csrfmiddlewaretoken": csrf,
            },
            headers={"Referer": f"{BASE}/search/", "X-CSRFToken": csrf},
            timeout=60,
        )
        r.raise_for_status()
        j = r.json()
        for first, last, _, link, received in j["data"]:
            m = re.search(r'href="(/search/view/(ptr|paper)/([0-9a-f\-]+)/)"', link)
            if m:
                rows.append({
                    "doc_id": m.group(3), "kind": m.group(2), "path": m.group(1),
                    "filer_name": f"{first.strip()} {last.strip()}".strip(), "filing_date": received,
                })
        start += 100
        if start >= j["recordsTotal"]:
            break
        time.sleep(0.5)
    return pd.DataFrame(rows)


def cells(row_html: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", c))).strip()
        for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.S)
    ]


def parse(html: str) -> list[dict]:
    i = html.find("<table")
    if i < 0:
        return []
    body = html[i : html.find("</table>", i)]
    out = []
    for row in re.findall(r"<tr.*?</tr>", body, re.S)[1:]:
        c = cells(row)
        if len(c) < 8:
            continue
        _, tdate, _owner, ticker, name, atype, ttype, amount = c[:8]
        amt = AMOUNT.search(amount)
        out.append({
            "ticker": None if ticker in ("--", "") else ticker.upper(),
            "asset_type": "ST" if atype == "Stock" else atype,
            "transaction_type": TYPES.get(ttype, ttype),
            "partial": ttype == "Sale (Partial)",
            "transaction_date": tdate,
            "notification_date": None,
            "amount_low": float(amt.group(1).replace(",", "")) if amt else None,
            "option": "option" in atype.lower(),
        })
    return out


def build(first_year: int, last_year: int, accept_terms: bool) -> None:
    s = session(accept_terms)
    index = pd.concat([search(s, y) for y in range(first_year, last_year + 1)], ignore_index=True)
    index.to_csv(INDEX_OUT, index=False)
    out = []
    elec = index[index["kind"] == "ptr"]
    for n, rec in enumerate(elec.itertuples(index=False), 1):
        path = RAW / f"{rec.doc_id}.html"
        if not path.exists():
            r = s.get(BASE + rec.path, timeout=60)
            if r.status_code != 200 or "<table" not in r.text:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text)
            time.sleep(0.5)
        for row in parse(path.read_text()):
            out.append(row | {"doc_id": rec.doc_id, "filer_name": rec.filer_name, "state_dst": "", "filing_date": rec.filing_date})
        if n % 200 == 0:
            print(f"{n}/{len(elec)} reports, {len(out)} rows", flush=True)
    df = pd.DataFrame(out)
    for c in ("transaction_date", "filing_date"):
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce")
    df.to_csv(OUT, index=False)
    print(f"{len(index)} reports, {len(elec)} electronic, {len(df)} transactions")


if __name__ == "__main__":
    build(int(sys.argv[1]), int(sys.argv[2]), "--accept-terms" in sys.argv)
