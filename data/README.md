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
