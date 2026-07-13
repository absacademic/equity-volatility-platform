# Smile and surface workflow

The pipeline starts from the clean Parquet quotes. It performs exact expiration timing, zero-rate interpolation, parity-forward estimation, bid/mid/ask IV inversion, smile fitting, diagnostics, and plotting.

## Main command

```bash
vol-platform surface PATH_TO_CLEAN_PARQUET \
  --rates PATH_TO_RATE_CSV \
  --date 2026-07-01
```

One can supply a directory containing partitioned clean Parquet files instead of one file

## Built-in synthetic demonstration

```bash
vol-platform synthetic-chain --output-dir data/interim/week3-demo

vol-platform surface data/interim/week3-demo/synthetic-clean-chain.parquet \
  --rates data/interim/week3-demo/synthetic-rates.csv \
  --output-dir data/processed/surfaces/demo
```

Run the two `vol-platform` commands directly. `make demo-surface` is only a convenience target for systems with Make installed.

## Outputs

The output directory contains:
- `implied-volatility.parquet`: bid, mid, and ask IVs plus moneyness, delta, vega, and total variance
- `forward-estimates.csv`: one parity-forward estimate per expiration with dispersion and reliability
- `forward-pairs.csv`: the near-ATM call-put pairs used in each estimate
- `smile-fit-details.csv`: per-expiration errors for every model and weighting combination
- `model-comparison.csv`: aggregate RMSE, maximum residual, coverage, stability, and failed-fit rates
- `surface.duckdb`: queryable copies of all Week 3 analytical tables
- `surface-report.md`: a compact run summary
- `plots/`: smile, residual, surface, bid-ask-band, and ATM-term-structure figures

## Conventions

- Expiration time defaults to 4:00 p.m. America/New_York
- Time to expiry uses exact elapsed seconds divided by the configured day-count basis
- Rates are continuously compounded zero rates and are linearly interpolated by maturity
- Forward moneyness is `log(K / F)`
- Spline and SVI models fit total variance, then convert predictions back to implied volatility
- Black-76 treats the contracts as European. For American-style SPY options, early-exercise and dividend effects can appear as parity dispersion or IV noise, especially for deep in-the-money contracts
- Stability is a zero-to-one smoothness score based on positive finite predictions and normalized curve curvature