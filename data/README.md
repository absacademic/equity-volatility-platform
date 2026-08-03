# Data

- `raw/` : immutable source extracts
- `interim/` : normalized intermediate datasets
- `processed/options/clean/`: accepted option quotes in partitioned Parquet
- `processed/options/rejected/`: rejected option quotes with reasons and scores
- `processed/reference/`: normalized underlying-price and rate data
- `processed/events/`: normalized event data
- `processed/metadata/`: ingestion manifests with hashes and date ranges
- `processed/volatility.duckdb`: analytical DuckDB views

Market data files to be excluded from Git, notwithstanding small `sample_spy_*` files to test pipeline
- `processed/surfaces/`: arbitrage diagnostics, standardized points, daily feature tables, reports, and charts

Small `sample_spy_*` files are retained for adapter and command demonstrations. Market-sized source data remains excluded from Git.

Event-study inputs and outputs:

- `interim/week5-demo/`: synthetic event, underlying-price, and surface-feature inputs
- `processed/event-studies/`: point-in-time events, event windows, models, backtests, attribution, plots, and conclusions

## Deterministic panel

`vol-platform synthetic-week6` creates a 2018-2025 sample under `data/interim/week6-demo`. Generated files include events, underlying prices, surface features, contract-level option quotes, and chronological strategy signals for AAPL, SPY, and XSP.
