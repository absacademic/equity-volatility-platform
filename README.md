# Equity Volatility Research Platform

A Python-based research platform for equity-option pricing, implied-volatility estimation, volatility-surface analysis, point-in-time event studies, strategy backtesting, and options P&L attribution.

The initial research universe is **SPY**, chosen because it is a highly liquid ETF.

## Startup

```bash
python -m venv .venv
```

You can now activate the environment.

## Windows PowerShell
```bash
.venv\Scripts\Activate.ps1
```

Install the package and development tools:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run all checks:

```bash
ruff check .
ruff format --check .
pytest --cov=vol_platform --cov-report=term-missing
```

## Linux Setup

To use this project on Linux-based systems, the following command-structure may be run:

```bash
git clone https://github.com/absacademic/equity-volatility-platform.git
cd equity-volatility-platform

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"

ruff check .
ruff format --check .
pytest --cov=vol_platform --cov-report=term-missing
```

Note that a Docker image may require installing `tzdata`:

```bash
sudo apt-get update
sudo apt-get install -y tzdata
```

The repository can also be run in Docker:

```bash
docker build -t equity-volatility-platform:0.6.0 .
docker run --rm equity-volatility-platform:0.6.0 vol-platform --help
```

Run complete deterministic workflow with:

```bash
make reproduce
```

Detailed installation, architecture, strategy, real-data, reproduction, and disclaimer documentation is available in the `docs/` directory.


## CLI examples 

Price a Black-Scholes call and return its Greeks:

```bash
vol-platform price \
  --underlying 100 \
  --strike 105 \
  --time 0.50 \
  --rate 0.04 \
  --vol 0.25 \
  --dividend-yield 0.01 \
  --type call
```

Recover an implied volatility:

```bash
vol-platform implied-vol \
  --price 6.20 \
  --underlying 100 \
  --strike 105 \
  --time 0.50 \
  --rate 0.04 \
  --dividend-yield 0.01 \
  --type call
```

Select Black-76 with:

```bash
--model black_76
```

Note that the `--underlying` argument means spot under Black-Scholes and forward under Black-76.

Price a discretely monitored up-and-out barrier call:

```bash
vol-platform monte-carlo-barrier \ 
  --spot 100 \ 
  --strike 100 \ 
  --barrier 125 \ 
  --time 1 \ 
  --rate 0.04 \ 
  --vol 0.25 \ 
  --type call \ 
  --barrier-type up_and_out \ 
  --paths 100000 \ 
  --steps 252 \ 
  --seed 7
```
The result includes the simulated price, standard error, 95% confidence interval, path count, monitoring steps, and barrier-hit probability.

## Some numerical conventions used

- Rates and divided yields are continuously compounded annual rate
- Time to expiry in years
- Black-Scholes uses a spot and continuous divided yield
- Black-76 uses a forward and discounts the expected payoff at risk-free rate
- Vega reported per `1.00` absolute volatility change; divisibl by `100` for sensitivity to a single volatility point
- Theta is annual time decay; Black-76 theta holds the forward as fixed
- `intrinsic-value` returns the immediate-exercise payoff w/o discounting (can be negative for European option)
- Implied-volatility solver validates European no-arbitrage bounds prior to attempting inversion

## Implied-volatility solver defined behavior

The solver returns an `ImpliedVolResult` rather than raising exceptions for expected quote-quality failures.

It can have statuses of:
- `success` : A finite implied volatility was recovered
- `at_lower_bound` : The price is consistent with zero volatility
- `at_upper_bound` : The price is at the model upper bound; no finite IV exists
- `price_below_lower_bound` : The input price violates the lower arbitrage bound
- `price_above_upper_bound` : The input price violates the upper arbitrage bound
- `expired` : Volatility is not identifiable at expiry
- `invalid_input` : A scalar input or solver setting is invalid.
- `non_convergence` : The configured volatility bracket did not produce a root

## Data schemas 

`src/vol_platform/schemas.py` defines immutable Pydantic records for:

- `OptionQuote`
- `UnderlyingPrice`
- `RateCurvePoint`
- `DividendRecord`
- `EventRecord`
- `ImpliedVolatilityRecord`

The corresponding DuckDB tables are defined in `sql/schema.sql`

Default settings live in `configs/base.yml`. The main environment variables are documented in `.env.example`:

```text
VOL_PLATFORM_DATA_DIR=./data
VOL_PLATFORM_CONFIG_PATH=./configs/base.yml
VOL_PLATFORM_LOG_LEVEL=INFO

## Repository layout

```text
equity-volatility-platform/
├── .github/workflows/ci.yml
├── configs/base.yml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/00_pricing_engine_smoke_test.ipynb
├── reports/
├── sql/schema.sql
├── src/vol_platform/
│   ├── cli.py
│   ├── config.py
│   ├── schemas.py
│   └── pricing/
│       ├── black_scholes.py
│       ├── black76.py
│       ├── bounds.py
│       ├── greeks.py
│       ├── implied_vol.py
│       ├── parity.py
│       └── values.py
├── tests/
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

## Data Ingestion
The initial data universe is SPY.

A single CSV can contain both option quotes and embedded underlying-price fields:

```bash
vol-platform ingest data/raw/sample_spy_quotes.csv
```

Standalone underlying, rate, and event files can also be supplied:

```bash
vol-platform ingest data/raw/sample_spy_quotes.csv \
  --underlying data/raw/sample_spy_underlying.csv \
  --rates data/raw/sample_spy_rates.csv \
  --events data/raw/sample_spy_events.csv
```

The same sample pipeline can be run with:

```bash
make demo-ingest
```

Each input quote is retained and receives:

  is_valid
  rejection_reason
  quote_quality_score

The current checks cover missing or invalid fields, expired contracts, nonpositive and crossed quotes, duplicates, wide spreads, low liquidity, implausible moneyness, and stale or missing underlying-price alignments.

Cleaning thresholds are configured in `configs/base.yml`.

The clean Parquet quotes are the start of the ingestion pipeline.

For one quote date:

```bash
vol-platform surface data/processed/options/clean --rates data/raw/sample_spy_rates.csv --date 2026-07-01
```

Generate and analyze a synthetic Week 4 dataset:

```bash
vol-platform synthetic-chain --output-dir data/interim/week4-demo

vol-platform surface data/interim/week4-demo/synthetic-clean-chain.parquet \
  --rates data/interim/week4-demo/synthetic-rates.csv \
  --dividends data/interim/week4-demo/synthetic-dividends.csv \
  --events data/interim/week4-demo/synthetic-events.csv \
  --underlying-history data/interim/week4-demo/synthetic-underlying-history.csv \
  --output-dir data/processed/surfaces/demo
```

On systems with Make installed, the same workflow can be run with:
```bash
make demo-surface
```

For each expiration, the forward estimator:

- Matches calls and puts with the same strike
- Selects several near-ATM pairs
- Calculates a parity-implied forward for each pair
- Weights pairs using quote quality and bid-ask spread
- Reports the weighted forward, dispersion, pair count, and reliability

The implied-volatility dataset contains:

- Bid, midpoint, and ask implied volatility
- Spot and forward log moneyness
- Delta and vega
- Total variance
- Solver status and residual
- Forward reliability and dispersion
- An OTM-option and smile-fit eligibility indicator

The smile models are:

- Weighted cubic smoothing spline
- Raw SVI fitted to total variance

Both models are evaluated using equal, vega, inverse-spread, and quote-quality weighting.

Generate and analyze a synthetic event-study dataset:

```bash
vol-platform synthetic-event-study --output-dir data/interim/week5-demo 

vol-platform event-study data/interim/week5-demo/synthetic-week5-surface-features.csv \ 
  --events data/interim/week5-demo/synthetic-week5-events.csv \ --underlying data/interim/week5-demo/synthetic-week5-underlying.csv \ --output-dir data/processed/event-studies/demo
```

On systems with Make installed, the same workflow can be run with:
```bash
make demo-event-study
```

On Windows PowerShell, run:

```bash
vol-platform synthetic-event-study --output-dir data/interim/week5-demo 

vol-platform event-study data/interim/week5-demo/synthetic-week5-surface-features.csv ` 
  --events data/interim/week5-demo/synthetic-week5-events.csv ` --underlying data/interim/week5-demo/synthetic-week5-underlying.csv ` --output-dir data/processed/event-studies/demo
```

To perform real event-study, replace the synthetic files with:
- Daily volatility-feature CSV or Parquet file
- Point-in-time event CSV
- Daily underlying-price CSV

The event file should contain:

event_id
event_type
event_timestamp
known_timestamp
title
symbols
source
expected

Only surface observations timestamped before an event are used. Events known only after their event timestamp are marked invalid and excluded from modeling.

Generate deterministic 2018-2025 sample:

```bash
vol-platform synthetic-week6 --output-dir data/interim/week6-demo
```

Run the exact contract-level strategy backtest:

```bash
vol-platform strategy-backtest data/interim/week6-demo/synthetic-week6-signals.csv \ 
  --option-quotes data/interim/week6-demo/synthetic-week6-option-quotes.csv \ 
  --underlying data/interim/week6-demo/synthetic-week6-underlying.csv \ 
  --config configs/week6-example.yml \ 
  --output-dir data/processed/strategies/week6-demo
```

The sample compares:

- AAPL as an individual equity
- SPY as an ETF
- XSP as an index product
- CPI, FOMC, earnings, and large-market-move events
- Multiple years and volatility regimes

On systems with Make installed:

```bash
make demo-week6
```

On Windows PowerShell:

```bash
vol-platform synthetic-week6 --output-dir data/interim/week6-demo 

vol-platform strategy-backtest data/interim/week6-demo/synthetic-week6-signals.csv ` 
  --option-quotes data/interim/week6-demo/synthetic-week6-option-quotes.csv ` 
  --underlying data/interim/week6-demo/synthetic-week6-underlying.csv ` 
  --config configs/week6-example.yml ` 
  --output-dir data/processed/strategies/week6-demo
```

## Data outputs

Clean and rejected option quotes are stored as partitioned Parquet files:

```bash
data/processed/options/clean/underlying_symbol=SPY/quote_date=YYYY-MM-DD/ data/processed/options/rejected/underlying_symbol=SPY/quote_date=YYYY-MM-DD/
```

Each ingestion run also creates:

```bash
data/processed/reference/
data/processed/events/
data/processed/metadata/
data/processed/volatility.duckdb
reports/generated/data-quality-<run_id>.md
reports/generated/rejection-summary-<run_id>.csv
```

The DuckDB database contains these analytical views:

  clean_quotes
  rejected_quotes
  daily_summaries
  expiration_chains

## Surface outputs

Each surface run creates:
implied-volatility.parquet 
forward-estimates.csv 
forward-pairs.csv 
smile-fit-details.csv 
model-comparison.csv 
surface.duckdb 
surface-report.md 
surface-manifest.json 
plots/smile.png 
plots/residuals.png
plots/surface.png 
plots/bid_ask_band.png 
plots/atm_term_structure.png
arbitrage-diagnostics.csv
surface-adjustments.csv
arbitrage-report.md
standardized-delta-points.csv
daily-volatility-features.parquet
daily-volatility-features.csv
plots/historical_atm_volatility.png
plots/historical_skew.png
plots/historical_vrp.png

The arbitrage diagnostic table separates violations found in midpoint prices, executable bid-ask prices, and fitted surfaces. The adjustment table records any variance floor, refitting control, or surface rejection.

The standardized-delta table contains:

10-delta put
25-delta put
ATM
25-delta call
10-delta call

The daily volatility-feature table contains one row per symbol, quote date, and expiration. It includes ATM volatility, downside skew, risk reversal, butterfly, curvature, term-structure slopes, IV bid-ask width, fit residuals, realized volatility, volatility-risk-premium measures, event features, historical changes, z-scores, percentiles, and cross-sectional ranks where applicable.

The DuckDB database contains:

implied_volatility_dataset
forward_estimates
forward_pairs
smile_fit_details
model_comparison
arbitrage_diagnostics
surface_adjustments
standardized_delta_points
daily_volatility_features
underlying_history

## Event-study outputs

Each event-study output creates:
point-in-time-events.parquet
point-in-time-events.csv
event-windows.parquet
event-windows.csv
event-study-dataset.parquet
event-study-dataset.csv
summary-analysis.csv
regime-comparison.csv
model-coefficients.csv
model-performance.csv
coefficient-stability.csv
walk-forward-results.parquet
walk-forward-results.csv
walk-forward-performance.csv
strategy-backtest.parquet
strategy-backtest.csv
strategy-summary.csv
pnl-attribution.parquet
pnl-attribution.csv
research-conclusion.md
event-study-report.json
event-study.duckdb
plots/expected_vs_realized.png
plots/surface_dislocation.png
plots/linear_coefficients.png
plots/strategy_backtest.png

The baseline linear model predicts the difference between the expected and realized event move. The logistic model predicts whether the expected move exceeds the realized move.

The strategy shorts the event straddle when the linear model predicts overestimation and buys it when the model predicts underestimation. Results are reported before and after estimated fixed and spread-based transaction costs and are compared with an always-short-straddle baseline.

The current straddle P&L is a daily approximation. Exact contract-level P&L would require actual option entry and exit quotes, contract multipliers, Greeks, hedge transactions, and execution timestamps.

## Some numerical conventions
- Expiration defaults to 4:00 p.m. America/New_York
- Time to expiration uses exact elapsed seconds divided by the configured day-count basis
- Rates are continuously compounded zero rates
- Rates are linearly interpolated by maturity and held flat outside the available curve
- Forward moneyness is log(K / F)
- Smile models fit total variance
- Implied volatilities are calculated using Black-76 and the parity-estimated forward
- Spline stability is evaluated from finite positive predictions and normalized curve curvature
- SPY options are American-style, while Black-76 is a European model. Early-exercise and dividend effects can therefore appear as parity dispersion or implied-volatility noise
- Dividend adjustments use only dividend information known by the quote timestamp
- Event features use only events known by the quote timestamp
- Realized-volatility windows use underlying prices strictly before the feature date
- Historical rolling and expanding statistics use only the current and previous feature rows
- Standardized deltas use the Black-76 discounted-forward delta convention
- Fitted smiles are controlled before they are selected for standardized interpolation
- Small numerical negative variances may be floored, but material arbitrage violations cause surface rejection
- Early-exercise-risk contracts are excluded from parity estimation and smile fitting by default
- Invalid chains remain in the feature table with `chain_valid = false`

## DISCLAIMER

The bundled datasets are small synthetic or sample datasets intended to validate the software and demonstrate reproducible workflows. They are NOT evidence of a profitable trading strategy.

Real-market research requires appropriately licensed option quotes, underlying prices, rates, dividend information, corporate-action records, and point-in-time event calendars. Historical midpoint prices should not be interpreted as executable fills. Live results may differ because of bid-ask spreads, slippage, commissions, margin requirements, financing, taxes, assignment, early exercise, market impact, timestamp quality, and data-vendor corrections.

Some documentation and testing was AI-assisted to ensure cross-platform compatibility and complete records for how this project operates. The author does not make any claim for the commercial vailidity nor efficacy of this platform. 