# Smile, surface, and feature workflow

The pipeline starts from clean Parquet option quotes. It performs exact expiration timing, rate interpolation, dividend-aware forward estimation, bid/mid/ask IV inversion, smile fitting, arbitrage controls, standardized delta interpolation, and point-in-time feature construction.

## Main command

```bash
vol-platform surface PATH_TO_CLEAN_PARQUET \
  --rates PATH_TO_RATE_CSV \
  --dividends PATH_TO_DIVIDEND_CSV \
  --events PATH_TO_EVENT_CSV \
  --underlying-history PATH_TO_UNDERLYING_HISTORY_CSV \
  --date 2026-07-01
```

A directory containing partitioned clean Parquet files can be supplied instead of one file. Optional reference files may also be omitted.

## Built-in synthetic demonstration

```bash
vol-platform synthetic-chain --output-dir data/interim/week4-demo

vol-platform surface data/interim/week4-demo/synthetic-clean-chain.parquet \
  --rates data/interim/week4-demo/synthetic-rates.csv \
  --dividends data/interim/week4-demo/synthetic-dividends.csv \
  --events data/interim/week4-demo/synthetic-events.csv \
  --underlying-history data/interim/week4-demo/synthetic-underlying-history.csv \
  --output-dir data/processed/surfaces/demo
```

Run the commands directly on Windows. `make demo-surface` is only a convenience target for systems with Make installed.

## Week 4 controls

The diagnostic layer tests midpoint prices, executable bid-ask quotes, and fitted surfaces separately. It checks strike monotonicity, butterfly convexity, negative total variance, calendar consistency, and unreasonable fitted-wing extrapolation.

Small numerical negative variances can be floored. Material fitted-surface violations are rejected. Every floor or rejection is written to `surface-adjustments.csv`; raw diagnostic margins remain in `arbitrage-diagnostics.csv`.

American-style contracts receive discrete-dividend present-value fields and a conservative early-exercise risk flag. Flagged observations are excluded from smile fitting by default, while remaining in the analytical dataset for review.

## Standardized and historical features

Each accepted fit is sampled at 10-delta put, 25-delta put, ATM, 25-delta call, and 10-delta call points. These points create ATM volatility, downside skew, 25-delta risk reversal, butterfly, wing curvature, term-structure slopes, IV bid-ask width, and fit-residual features.

Realized volatility uses underlying prices strictly before the option quote date. Historical changes, rolling and expanding z-scores, expanding percentiles, and same-date cross-sectional ranks are therefore point-in-time. Scheduled events are linked only when their timestamps and expected status were available by the feature timestamp.

## Outputs

The output directory contains:

- `implied-volatility.parquet`: bid, mid, and ask IVs plus moneyness, delta, vega, and total variance
- `forward-estimates.csv`: parity and dividend-adjusted forward comparisons
- `forward-pairs.csv`: near-ATM call-put pairs used in each parity estimate
- `smile-fit-details.csv`: per-expiration diagnostics for every model and weighting
- `model-comparison.csv`: aggregate fit errors, coverage, stability, and failure rates
- `arbitrage-diagnostics.csv`: market and fitted-surface no-arbitrage checks
- `surface-adjustments.csv`: all variance floors and fitted-surface rejections
- `arbitrage-report.md`: compact violation and adjustment summary
- `standardized-delta-points.csv`: five standardized points per valid expiration
- `daily-volatility-features.parquet` and `.csv`: one row per symbol, date, and expiration
- `surface.duckdb`: queryable copies of all analytical tables
- `surface-report.md`: current-date run summary
- `plots/`: current smile, residual, surface, bid-ask, and term-structure figures
- `plots/historical/`: historical ATM volatility, skew, and VRP figures

## Conventions

- Expiration time defaults to 4:00 p.m. America/New_York
- Time to expiry uses exact elapsed seconds divided by the configured day-count basis
- Rates are continuously compounded zero rates and are linearly interpolated by maturity
- Forward moneyness is `log(K / F)`
- Standardized points use absolute Black-76 discounted forward delta
- Spline and SVI models fit total variance and convert predictions back to implied volatility
- A feature row is retained even when invalid; `chain_valid` indicates whether all material controls and standardized-point requirements passed
