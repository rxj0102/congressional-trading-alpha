"""
src/visualize.py
================
Publication-quality charts for the Congressional Trading Lag Arbitrage strategy.

All figures are saved to ``outputs/`` at 300 DPI in PNG format.  Functions
are stateless: they receive data and return the ``matplotlib.figure.Figure``
object while also persisting to disk.

Requires ``matplotlib >= 3.8`` and ``seaborn >= 0.13``.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

import config

logger = logging.getLogger(__name__)

# ── Style defaults ────────────────────────────────────────────────────────────
_STYLE = "seaborn-v0_8-whitegrid"
_DPI = 300
_FIGSIZE_WIDE = (12, 5)
_FIGSIZE_SQUARE = (8, 6)
_PALETTE = {"buy": "#2563eb", "sell": "#dc2626", True: "#f59e0b", False: "#6b7280"}
_BUY_COLOR = "#2563eb"
_SELL_COLOR = "#dc2626"
_GREEN = "#16a34a"
_RED = "#dc2626"
_NEUTRAL = "#6b7280"


def _apply_style() -> None:
    """Apply the project's standard Matplotlib style."""
    try:
        plt.style.use(_STYLE)
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


# ── 1. Cumulative PnL ─────────────────────────────────────────────────────────

def plot_cumulative_pnl(results_df: pd.DataFrame) -> plt.Figure:
    """
    Plot cumulative long-short PnL with shaded drawdown periods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of :func:`src.backtest.run_backtest`.  Must contain columns:
        ``week`` (str), ``cumulative_pnl`` (float), ``ls_return`` (float).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure (also saved to ``config.CHART_CUMULATIVE_PNL``).

    Notes
    -----
    Drawdown periods are shaded in translucent red.  The drawdown is measured
    against the rolling maximum of the arithmetic cumulative PnL series.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

    x = range(len(results_df))
    pnl = results_df["cumulative_pnl"].values
    roll_max = np.maximum.accumulate(pnl)

    # Main PnL line
    ax.plot(x, pnl, color=_GREEN, linewidth=2, label="Cumulative L/S PnL", zorder=3)

    # Zero reference
    ax.axhline(0, color=_NEUTRAL, linewidth=1, linestyle="--", alpha=0.7)

    # Drawdown shading
    ax.fill_between(
        x,
        pnl,
        roll_max,
        where=(pnl < roll_max),
        interpolate=True,
        color=_RED,
        alpha=0.18,
        label="Drawdown",
    )

    # X-axis ticks: show every nth week label to avoid crowding
    n_labels = min(12, len(results_df))
    step = max(1, len(results_df) // n_labels)
    tick_positions = list(range(0, len(results_df), step))
    tick_labels = [results_df["week"].iloc[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    ax.set_title("Congressional Trading: Cumulative Long/Short PnL", fontweight="bold")
    ax.set_xlabel("ISO Week")
    ax.set_ylabel("Cumulative Return (arithmetic)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(config.CHART_CUMULATIVE_PNL, dpi=_DPI, bbox_inches="tight")
    logger.info("Saved cumulative PnL chart → %s", config.CHART_CUMULATIVE_PNL)
    return fig


# ── 2. Lag distribution ───────────────────────────────────────────────────────

def plot_lag_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Histogram of ``lag_days`` split by buy vs. sell transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched disclosures DataFrame with ``lag_days`` and
        ``transaction_type`` columns.

    Returns
    -------
    matplotlib.figure.Figure
        Saved to ``config.CHART_LAG_DISTRIBUTION``.

    Notes
    -----
    A vertical dashed line marks the 45-day legal maximum under the STOCK
    Act.  Distributions are overlaid semi-transparently; a rug strip is added
    at the bottom for sample density.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

    bins = np.arange(0, config.MAX_LEGAL_LAG + 3, 2)

    for t_type, color, label in [
        ("buy", _BUY_COLOR, "Buy"),
        ("sell", _SELL_COLOR, "Sell"),
    ]:
        sub = df[df["transaction_type"] == t_type]["lag_days"].dropna()
        if len(sub) == 0:
            continue
        ax.hist(
            sub,
            bins=bins,
            color=color,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.5,
            label=f"{label} (n={len(sub):,})",
        )
        # Rug
        ax.plot(
            sub,
            np.full(len(sub), -0.5),
            "|",
            color=color,
            alpha=0.15,
            markersize=4,
        )

    ax.axvline(
        config.MAX_LEGAL_LAG,
        color="#854d0e",
        linestyle="--",
        linewidth=1.8,
        label=f"{config.MAX_LEGAL_LAG}-day STOCK Act limit",
    )

    ax.set_title("Disclosure Lag Distribution by Transaction Type", fontweight="bold")
    ax.set_xlabel("Disclosure Lag (calendar days)")
    ax.set_ylabel("Count")
    ax.legend()

    fig.tight_layout()
    fig.savefig(config.CHART_LAG_DISTRIBUTION, dpi=_DPI, bbox_inches="tight")
    logger.info("Saved lag distribution chart → %s", config.CHART_LAG_DISTRIBUTION)
    return fig


# ── 3. Composite score vs. forward return ─────────────────────────────────────

def plot_composite_score_vs_return(df: pd.DataFrame) -> plt.Figure:
    """
    Scatter plot of ``composite_score`` vs ``forward_return`` with OLS trend.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched disclosures DataFrame with ``composite_score``,
        ``forward_return``, and ``position`` columns.

    Returns
    -------
    matplotlib.figure.Figure
        Saved to ``config.CHART_SCORE_VS_RETURN``.

    Notes
    -----
    Points are coloured by ``position`` (LONG = blue, SHORT = red,
    NEUTRAL = grey).  An OLS trend line is overlaid using
    ``numpy.polyfit``.  A Pearson correlation coefficient is annotated in the
    upper-left corner.
    """
    _apply_style()
    plot_df = df.dropna(subset=["composite_score", "forward_return"]).copy()

    if len(plot_df) < 5:
        logger.warning("Too few points for composite_score scatter — skipping.")
        fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)

    color_map = {"LONG": _BUY_COLOR, "SHORT": _SELL_COLOR, "NEUTRAL": _NEUTRAL}
    for pos, color in color_map.items():
        sub = plot_df[plot_df.get("position", "NEUTRAL") == pos] if "position" in plot_df.columns else plot_df
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["composite_score"],
            sub["forward_return"],
            color=color,
            alpha=0.45,
            s=20,
            edgecolors="none",
            label=pos,
        )

    # OLS trend line
    x = plot_df["composite_score"].values
    y = plot_df["forward_return"].values
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = np.polyval(coeffs, x_line)
    ax.plot(x_line, y_line, color="#1e293b", linewidth=1.8, label="OLS fit")

    # Pearson r annotation
    corr = float(np.corrcoef(x, y)[0, 1])
    ax.text(
        0.04, 0.94,
        f"Pearson r = {corr:+.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    ax.axhline(0, color=_NEUTRAL, linewidth=0.8, linestyle="--")
    ax.axvline(0, color=_NEUTRAL, linewidth=0.8, linestyle="--")

    ax.set_title("Composite Signal Score vs. 10-Day Forward Return", fontweight="bold")
    ax.set_xlabel("Composite Score\n(+ = stronger LONG signal, − = stronger SHORT signal)")
    ax.set_ylabel(f"{config.POST_DISCLOSURE_WINDOW}-Day Forward Return")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(config.CHART_SCORE_VS_RETURN, dpi=_DPI, bbox_inches="tight")
    logger.info("Saved score vs. return chart → %s", config.CHART_SCORE_VS_RETURN)
    return fig


# ── 4. Coverage score distribution ────────────────────────────────────────────

def plot_coverage_score_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Histogram of ``coverage_score`` split by ``low_coverage`` flag.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched disclosures DataFrame with ``coverage_score`` and
        ``low_coverage`` columns.

    Returns
    -------
    matplotlib.figure.Figure
        Saved to ``config.CHART_COVERAGE_DISTRIBUTION``.

    Notes
    -----
    The vertical dashed line marks the 10th-percentile threshold used to
    define ``low_coverage = True``.  Low-coverage names (orange) represent
    the highest informational-asymmetry opportunity set.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

    plot_df = df.dropna(subset=["coverage_score"]).copy()
    p10 = plot_df["coverage_score"].quantile(0.10)

    for flag, color, label in [
        (True, "#f59e0b", f"Low coverage (≤ p10 = {p10:.0f} articles)"),
        (False, "#6b7280", "Normal coverage"),
    ]:
        sub = plot_df[plot_df["low_coverage"] == flag]["coverage_score"]
        if len(sub) == 0:
            continue
        ax.hist(
            sub,
            bins=40,
            color=color,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.4,
            label=f"{label} (n={len(sub):,})",
        )

    ax.axvline(
        p10,
        color="#92400e",
        linestyle="--",
        linewidth=1.8,
        label=f"p10 threshold ({p10:.0f})",
    )

    ax.set_title("Finviz News Article Count Distribution (Coverage Score)", fontweight="bold")
    ax.set_xlabel("Coverage Score (number of Finviz news articles)")
    ax.set_ylabel("Count")
    ax.legend()

    fig.tight_layout()
    fig.savefig(config.CHART_COVERAGE_DISTRIBUTION, dpi=_DPI, bbox_inches="tight")
    logger.info("Saved coverage distribution chart → %s", config.CHART_COVERAGE_DISTRIBUTION)
    return fig
