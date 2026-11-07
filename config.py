"""
config.py
=========
Central configuration for the Congressional Trading Lag Arbitrage project.

All constants, paths, and strategy parameters live here so that no hardcoded
values appear anywhere else in the codebase.
"""

from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Ensure output directories exist at import time
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Data source URLs ──────────────────────────────────────────────────────────
HOUSE_URL: str = "https://housestockwatcher.com/api"
SENATE_URL: str = "https://senatestockwatcher.com/api"

# ── HTTP settings ─────────────────────────────────────────────────────────────
REQUEST_TIMEOUT: int = 10        # seconds
REQUEST_SLEEP: float = 1.0       # seconds between successive requests

# ── Strategy parameters ───────────────────────────────────────────────────────
LOOKBACK_DAYS: int = 90          # window (days) for computing coverage score
POST_DISCLOSURE_WINDOW: int = 10 # trading days to measure forward return
MAX_LEGAL_LAG: int = 45          # STOCK Act maximum allowed disclosure lag (days)
TOP_DECILE_THRESHOLD: float = 0.9  # composite score percentile cut for LONG

# ── Committee influence weights ───────────────────────────────────────────────
COMMITTEE_LIST: list[str] = [
    "Armed Services",
    "Finance",
    "Intelligence",
    "Appropriations",
    "Energy",
]

COMMITTEE_WEIGHTS: dict[str, float] = {
    "Armed Services":  1.00,
    "Intelligence":    1.00,
    "Appropriations":  0.85,
    "Finance":         0.75,
    "Energy":          0.70,
    "other":           0.50,
}

# ── Static data files ─────────────────────────────────────────────────────────
COMMITTEE_CSV: Path = DATA_DIR / "committee_assignments_example.csv"

# ── Output file paths ─────────────────────────────────────────────────────────
# Raw cache files
HOUSE_CACHE: Path = DATA_DIR / "house_disclosures_{date}.json"
SENATE_CACHE: Path = DATA_DIR / "senate_disclosures_{date}.json"
PRICE_CACHE: Path = DATA_DIR / "prices_{date}.json"

# Charts
CHART_CUMULATIVE_PNL: Path = OUTPUTS_DIR / "cumulative_pnl.png"
CHART_LAG_DISTRIBUTION: Path = OUTPUTS_DIR / "lag_distribution.png"
CHART_SCORE_VS_RETURN: Path = OUTPUTS_DIR / "composite_score_vs_return.png"
CHART_COVERAGE_DISTRIBUTION: Path = OUTPUTS_DIR / "coverage_score_distribution.png"

# Tabular results
RESULTS_CSV: Path = OUTPUTS_DIR / "backtest_results.csv"
POSITIONS_CSV: Path = OUTPUTS_DIR / "weekly_positions.csv"
METRICS_CSV: Path = OUTPUTS_DIR / "performance_metrics.csv"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
