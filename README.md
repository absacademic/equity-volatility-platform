# Equity Volatility Research Platform

A Python-based research platform that performs equity-option pricing, implied-volatility estimation, surface features, event studies, strategy backtesting, and options P&L attribution (most features in progress).

The initial research universe is **SPY**, chosen since it is a highly liquid ETF.

## Current Status:

Inital phase: 
Repo being set-up along with environment and basic pricing models and implied-volatility solver. Typer CLI interfaced with ordinary typed parameters included along with DuckDB table definitions, RUFF linting/formattting, and a number of unit and integration tests.

Phase 2:
Added a reproducible data pipeline for option quotes, underlying prices, interest rates, and event data. Raw quotes are normalized, aligned with underlying prices, checked for data-quality problems, and stored as clean and rejected Parquet datasets. The pipeline also creates analytical DuckDB views, metadata manifests, and data-quality reports.

Phase 3:
Added an implied-volatility smile and surface pipeline. The pipeline calculates exact time to expiration, interpolates continuously compounded zero rates, estimates forwards from several near-ATM call-put pairs, and calculates bid, midpoint, and ask implied volatilities.

Cubic-spline and SVI smiles are fitted using equal, vega, spread, and quote-quality weighting. The models are compared using RMSE, maximum residual, coverage, stability, and failed-fit rates. The pipeline also produces smile, residual, surface, bid-ask-band, and ATM term-structure plots.

Phase 4:

Added no-arbitrage diagnostics, dividend and early-exercise adjustments, standardized delta-point interpolation, event-linked volatility features, and historical surface comparisons.

The pipeline now checks midpoint prices, executable bid-ask prices, and fitted surfaces separately for strike-monotonicity, butterfly-convexity, negative-total-variance, calendar-consistency, and extrapolation violations. Small numerical errors can be adjusted, while materially invalid surfaces are rejected. Every adjustment is recorded.

The platform interpolates 10-delta put, 25-delta put, ATM, 25-delta call, and 10-delta call volatility points. It produces one point-in-time feature row per symbol, date, and expiration containing volatility, skew, risk-reversal, butterfly, curvature, term-structure, residual, realized-volatility, volatility-risk-premium, event, and historical-comparison features.


## Startup

```bash
python -m venv .venv
```

You can now activate the environment.

## Windows PowerShell
.venv\Scripts\Activate.ps1

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

On Windows PowerShell, run:

```bash
vol-platform synthetic-chain --output-dir data/interim/week4-demo

vol-platform surface data/interim/week4-demo/synthetic-clean-chain.parquet `
  --rates data/interim/week4-demo/synthetic-rates.csv `
  --dividends data/interim/week4-demo/synthetic-dividends.csv `
  --events data/interim/week4-demo/synthetic-events.csv `
  --underlying-history data/interim/week4-demo/synthetic-underlying-history.csv `
  --output-dir data/processed/surfaces/demo
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

## Completion criterion

`tests/test_week4_pipeline.py` verifies that one surface workflow:

- Reads clean option quotes and all optional point-in-time reference inputs
- Applies dividend and early-exercise adjustments
- Diagnoses midpoint, executable, and fitted-surface arbitrage violations separately
- Records surface controls and rejections
- Interpolates five standardized delta points per expiration
- Calculates volatility, skew, butterfly, curvature, and term-structure features
- Calculates realized-volatility and volatility-risk-premium measures
- Links known events without look-ahead bias
- Calculates rolling z-scores, expanding z-scores, percentiles, and applicable cross-sectional ranks
- Creates historical volatility, skew, and VRP charts
- Writes all diagnostic, standardized-point, feature, report, manifest, and DuckDB outputs
- Produces exactly one quality-controlled feature row per symbol, date, and expiration
- Retains invalid chains as explicitly flagged feature rows

## TODO

Next add event-study analysis, trading-strategy backtests, transaction-cost modeling, and options P&L attribution