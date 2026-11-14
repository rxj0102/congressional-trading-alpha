# Congressional Trading Lag Arbitrage

[![CI](https://github.com/rxj0102/congressional-trading-alpha/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/congressional-trading-alpha/actions/workflows/ci.yml)

A quantitative research project that exploits the informational asymmetry embedded in the U.S. STOCK Act's mandatory 45-day congressional disclosure window.  Using publicly available trade filings from House and Senate members, this repository constructs a composite ranking signal, validates it with OLS regression, and backtests a weekly long/short equity strategy — all with reproducible, production-quality Python code.

---

## Overview

Members of Congress possess material non-public information by virtue of their legislative and oversight responsibilities.  The **STOCK Act (2012)** requires them to publicly disclose personal stock trades within 45 calendar days — creating a known, bounded window of informational advantage that can be systematically exploited after the disclosure becomes public.

This project operationalises that insight into a quantitative signal by combining three observable features:

| Feature | Rationale |
|---|---|
| **Disclosure lag** | Trades filed close to the deadline suggest the member delayed a revealing disclosure. |
| **Committee influence weight** | Members overseeing defence, intelligence, or finance sectors have higher regulatory information advantages. |
| **Analyst coverage (inverse)** | Low-coverage names carry the largest informational asymmetry between insiders and the public market. |

---

## Hypothesis

> *Congressional stock trades in low-analyst-coverage names, filed by members of high-influence committees close to the 45-day STOCK Act deadline, predict statistically significant positive forward returns over the subsequent 10 trading days, generating a positive alpha long/short spread.*

---

## Data Sources

| Source | URL | Description |
|---|---|---|
| House Stock Watcher | `housestockwatcher.com/api` | JSON API of House member disclosures |
| Senate Stock Watcher | `senatestockwatcher.com/api` | JSON API of Senate member disclosures |
| Yahoo Finance (via yfinance) | — | Adjusted daily close prices |
| Finviz | `finviz.com/quote.ashx` | News article count as coverage proxy |

All raw API responses are cached in `data/` as JSON files keyed by fetch date, so re-runs do not re-scrape.

---

## Methodology

### Composite Signal Formula

$$
\text{composite}_i = \underbrace{\text{direction}_i}_{\pm 1} \times \underbrace{\ln(1 + \text{lag\_days}_i)}_{\text{dampened lag}} \times \underbrace{w_{\text{committee}_i}}_{\text{influence}} \times \underbrace{\dfrac{1}{1 + \text{coverage\_score}_i}}_{\text{obscurity premium}}
$$

**Component interpretation:**

- $\text{direction} \in \{+1, -1\}$ — preserves the sign; buys are bullish signals, sells are bearish.
- $\ln(1 + \text{lag\_days})$ — log-dampened lag rewards disclosures filed late without letting extreme values dominate; bounded and slow-growing.
- $w_{\text{committee}}$ — scalar weight ∈ [0.5, 1.0] reflecting legislative oversight power over traded sectors.
- $\frac{1}{1+\text{coverage\_score}}$ — inverse Finviz article count; discounts well-followed mega-caps, amplifies obscure small-caps where the information edge is largest.

### Committee Influence Weights

| Committee | Weight |
|---|---|
| Armed Services | 1.00 |
| Intelligence | 1.00 |
| Appropriations | 0.85 |
| Finance | 0.75 |
| Energy | 0.70 |
| Other | 0.50 |

### Backtest Design

- **Frequency:** weekly position generation, entries at the Monday open following the signal week.
- **Holding period:** 10 trading days.
- **Book construction:** equal-weighted longs (top decile by composite score) and equal-weighted shorts (bottom decile).
- **PnL measure:** arithmetic long-short return (gross of transaction costs).

---

## Repository Structure

```
congressional-trading-alpha/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                          # Central config — all constants & paths
├── data/
│   ├── .gitkeep
│   ├── committee_assignments_example.csv  # Placeholder member→committee mapping
│   └── [auto-generated cache files]
├── outputs/
│   ├── .gitkeep
│   └── [charts and CSVs generated at runtime]
├── notebooks/
│   └── 01_exploratory_analysis.ipynb  # End-to-end research notebook
└── src/
    ├── __init__.py
    ├── scraper.py                     # API fetching & caching
    ├── features.py                    # Feature engineering pipeline
    ├── signal.py                      # OLS regression, ranking, positions
    ├── backtest.py                    # Event-driven backtest & metrics
    └── visualize.py                   # Chart generation
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Step 1 — Fetch and cache disclosure data
python -m src.scraper

# Step 2 — Compute signals and regression
python -m src.signal

# Step 3 — Run backtest and print metrics
python -m src.backtest
```

Each module can also be run in isolation; steps 2 and 3 include synthetic-data demos so they run without live network access.

### 3. Explore interactively

```bash
cd notebooks
jupyter notebook 01_exploratory_analysis.ipynb
```

### 4. Committee assignment enrichment (optional)

To activate committee-weighted signals, provide a CSV mapping member names to committees:

```bash
# See data/committee_assignments_example.csv for the expected format
# Download current assignments from clerk.house.gov or senate.gov
cp /path/to/your/assignments.csv data/committee_assignments_example.csv
```

---

## Results

> See `outputs/` after running the full pipeline.  Generated files include:
>
> | File | Description |
> |---|---|
> | `cumulative_pnl.png` | Weekly cumulative long/short PnL with drawdown shading |
> | `lag_distribution.png` | Histogram of disclosure lag by buy/sell |
> | `composite_score_vs_return.png` | Signal score vs. 10-day forward return scatter |
> | `coverage_score_distribution.png` | Coverage proxy distribution by low/normal flag |
> | `backtest_results.csv` | Week-by-week PnL table |
> | `weekly_positions.csv` | All generated long/short positions |
> | `performance_metrics.csv` | Sharpe, max drawdown, win rate |

---

## Reproducibility

### Data availability

All inputs come from free, public sources. No paid data or API keys are required unless noted.

- Yahoo Finance via the yfinance package (free, no key)
- Congressional trading disclosures (free public filings)
- Files shipped in the `data/` folder of this repository

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `uv.lock`.
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
pip install uv     # once
uv sync            # installs the exact pinned environment
uv run pytest -q   # runs the test suite
```

### Runtime

The test suite (11 tests) finishes in about 25 seconds on the machine above, including any data downloads it triggers.

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.


### Signal documentation

| Field | Value |
|---|---|
| Signal | composite = direction x ln(1 + lag_days) x committee_weight x 1 / (1 + coverage_score) |
| Expected sign | Positive: late-disclosed buys by influential committee members in thinly covered stocks outperform |
| Book | Weekly, top decile long and bottom decile short (`TOP_DECILE_THRESHOLD` = 0.9), held 10 trading days, entry at the Monday open after the signal week |
| Inputs | STOCK Act periodic transaction reports, committee assignments, Finviz news counts as a coverage proxy, prices (yfinance) |
| Code | `src/scraper.py`, `src/features.py`, `src/signal.py`, `src/backtest.py` |
| Status | Not run. The House and Senate Stock Watcher feeds returned HTTP 403 from the environment used for this review. Two design points to resolve before a run: `coverage_score` comes from a current Finviz snapshot, so applying it to past trades uses information from after the trade (the code docstring already flags this), `data/committee_assignments_example.csv` is an example file rather than sourced assignments, and `COMMITTEE_WEIGHTS` in `config.py` are judgment calls |

## Academic References

- **Karadas, Salim, Jahangir Sultan, and Omer Faruk Erol (2017).** "Do Members of U.S. Congress Trade on Non-Public Information?" *North American Journal of Economics and Finance* 42: 143–158.

- **Eggers, Andrew C., and Jens Hainmueller (2013).** "Capitol Losses: The Mediocre Performance of Congressional Stock Portfolios." *Journal of Politics* 75(2): 535–551.

- **Ziobrowski, Alan J., Ping Cheng, James W. Boyd, and Brigitte J. Ziobrowski (2004).** "Abnormal Returns from the Common Stock Investments of the United States Senate." *Journal of Financial and Quantitative Analysis* 39(4): 661–676.

---

## Disclaimer

> **This repository is for academic and educational research purposes only.**  
> Nothing in this codebase constitutes investment advice, financial advice, or a solicitation to buy or sell any security.  
> Past performance of any backtested strategy is not indicative of future results.  
> The authors make no representations about the accuracy or completeness of the data sourced from third-party APIs.  
> Trading on material non-public information is illegal; this project exclusively analyses information that is already in the public domain (STOCK Act disclosures).  
> Always consult a qualified financial professional before making investment decisions.

## Known limitations

- **Full-sample ranking.** `rank_signals` computes decile/LONG/SHORT cutoffs from the whole disclosure history at once, so the thresholds applied to early weeks use information from later weeks. A point-in-time version would use an expanding (or trailing) window of past scores only.
- **Coverage score is current, not historical.** `compute_coverage_score` counts today's Finviz headlines, which is not what the market could see at the disclosure date.
