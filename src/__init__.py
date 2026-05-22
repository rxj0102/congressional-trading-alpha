"""
src
===
Congressional Trading Lag Arbitrage — source package.

Modules
-------
scraper    : Pull and cache congressional disclosure data from public APIs.
features   : Compute derived features (lag, coverage, committee weight, signal).
signal     : OLS regression, signal ranking, and weekly position generation.
backtest   : Event-driven long/short backtest and performance metrics.
visualize  : Matplotlib/Seaborn charts saved to outputs/.
"""
