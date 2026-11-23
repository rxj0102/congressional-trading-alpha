"""
House periodic transaction reports from the Clerk of the House (official source).

The yearly index ``{YEAR}FD.zip`` at disclosures-clerk.house.gov lists every
financial disclosure filing with its filing date and document id. Periodic
transaction reports (STOCK Act, filing type P) are PDFs at
``public_disc/ptr-pdfs/{YEAR}/{DocID}.pdf``. Electronically filed reports
(document ids starting with 2) carry a text layer that ``pdftotext -layout``
(poppler) turns into a table; paper filings (ids starting with 8) are scans and
are skipped, which is counted and reported.

The public date of a transaction is the report's filing date in the index: that
is when anyone outside Congress could first see it. The notification date in the
PDF is when the member was notified of the trade.

    python -m src.house_clerk 2013 2026     # downloads and parses, writes data/house_ptr_transactions.csv
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://disclosures-clerk.house.gov/public_disc"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "house_raw"
OUT = ROOT / "data" / "house_ptr_transactions.csv"
INDEX_OUT = ROOT / "data" / "house_ptr_index.csv"
HEADERS = {"User-Agent": "academic research (congressional-trading-alpha)"}

DATE = r"(\d{1,2}/\d{1,2}/\d{4})"
ROW = re.compile(
    rf"(?P<type>\b[SsPp] \(partial\)|\b[Pp]\b|\b[Ss]\b|\b[Ee]\b)\s+{DATE}\s+{DATE}\s+(?P<amount>\$[\d,]+(?: - \$?[\d,]*)?|Over \$[\d,]+)?",
)
# Reports filed before about 2018 come out of pdftotext with a broken font: letters in
# tickers and transaction types can be lower case ("(lYg)" is LYG), so matching ignores
# case and the ticker is upper-cased.
TICKER = re.compile(r"\(([A-Za-z][A-Za-z0-9.\-]{0,6})\)")
STATUS = re.compile(r"filing status", re.I)
ASSET_TYPE = re.compile(r"\[([A-Z]{2})\]")
AMOUNT_RANGE = re.compile(r"\$([\d,]+)")


def year_index(year: int) -> pd.DataFrame:
    """Periodic transaction reports listed in one year's index."""
    r = requests.get(f"{BASE}/financial-pdfs/{year}FD.zip", headers=HEADERS, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    xml = z.read(f"{year}FD.xml").decode("utf-8-sig")
    rows = []
    for m in re.findall(r"<Member>(.*?)</Member>", xml, re.S):
        get = lambda tag: (re.search(rf"<{tag}>(.*?)</{tag}>", m, re.S) or [None, ""])[1].strip()  # noqa: E731
        if get("FilingType") != "P":
            continue
        rows.append(
            {
                "doc_id": get("DocID"),
                "year": year,
                "last": get("Last"),
                "first": get("First"),
                "state_dst": get("StateDst"),
                "filing_date": get("FilingDate"),
            }
        )
    return pd.DataFrame(rows)


def pdf_text(year: int, doc_id: str) -> str | None:
    path = RAW / str(year) / f"{doc_id}.pdf"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"{BASE}/ptr-pdfs/{year}/{doc_id}.pdf", headers=HEADERS, timeout=60)
        if r.status_code != 200 or not r.content.startswith(b"%PDF"):
            return None
        path.write_bytes(r.content)
        time.sleep(0.2)
    out = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def lower_bound(amount: str | None) -> float | None:
    if not amount:
        return None
    m = AMOUNT_RANGE.search(amount)
    return float(m.group(1).replace(",", "")) if m else None


def parse(text: str) -> list[dict]:
    """Transaction rows of one report: ticker, asset type, type, dates, amount.

    The asset description is the text left of the transaction type on the row's line
    plus the same column on the following lines, up to the FILING STATUS line or the
    next row (the name, ticker and asset code often wrap several lines down).
    """
    lines = text.splitlines()
    rows = []
    for i, line in enumerate(lines):
        m = ROW.search(line)
        if not m:
            continue
        width = m.start()
        context = line[:width]
        for k in range(1, 9):
            if i + k >= len(lines):
                break
            nxt = lines[i + k]
            if ROW.search(nxt) or STATUS.search(nxt):
                break
            context += " " + nxt[: width + 12]
        tick = [t.upper() for t in TICKER.findall(context)]
        tick = [t for t in tick if t not in {"ST", "OP", "GS", "CS", "PS", "OT", "MF", "EF"}]
        atype = ASSET_TYPE.findall(context)
        rows.append(
            {
                "ticker": tick[-1] if tick else None,
                "asset_type": atype[0] if atype else None,
                "transaction_type": m.group("type").split()[0].upper(),
                "partial": "partial" in m.group("type"),
                "transaction_date": m.group(2),
                "notification_date": m.group(3),
                "amount_low": lower_bound(m.group("amount")),
                "option": bool(re.search(r"\boption|\bcall\b|\bput\b", context, re.I)),
            }
        )
    return rows


def reparse() -> None:
    """Parse again from the cached PDFs and the saved index, without downloading."""
    index = pd.read_csv(INDEX_OUT, dtype={"doc_id": str})
    out = []
    for rec in index[index["electronic"]].itertuples(index=False):
        path = RAW / str(rec.year) / f"{rec.doc_id}.pdf"
        if not path.exists():
            continue
        text = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True).stdout
        for row in parse(text):
            out.append(row | {"doc_id": rec.doc_id, "filer_name": f"{rec.first} {rec.last}".strip(),
                              "state_dst": rec.state_dst, "filing_date": rec.filing_date})
    df = pd.DataFrame(out)
    for c in ("transaction_date", "notification_date", "filing_date"):
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce")
    df.to_csv(OUT, index=False)
    print(f"{len(df)} transactions")


def build(first_year: int, last_year: int) -> None:
    index = pd.concat([year_index(y) for y in range(first_year, last_year + 1)], ignore_index=True)
    index["electronic"] = index["doc_id"].str.startswith("2")
    index.to_csv(INDEX_OUT, index=False)
    out = []
    elec = index[index["electronic"]]
    for n, rec in enumerate(elec.itertuples(index=False), 1):
        text = pdf_text(rec.year, rec.doc_id)
        if text is None:
            continue
        for row in parse(text):
            out.append(
                row
                | {
                    "doc_id": rec.doc_id,
                    "filer_name": f"{rec.first} {rec.last}".strip(),
                    "state_dst": rec.state_dst,
                    "filing_date": rec.filing_date,
                }
            )
        if n % 200 == 0:
            print(f"{n}/{len(elec)} reports, {len(out)} rows", flush=True)
    df = pd.DataFrame(out)
    for c in ("transaction_date", "notification_date", "filing_date"):
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce")
    df.to_csv(OUT, index=False)
    print(f"{len(index)} reports, {int(index['electronic'].sum())} electronic, {len(df)} transactions")


if __name__ == "__main__":
    if sys.argv[1] == "reparse":
        reparse()
    else:
        build(int(sys.argv[1]), int(sys.argv[2]))
