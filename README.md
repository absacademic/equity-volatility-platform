# Equity Volatility Research Platform

A Python-based research platform that performs equity-option pricing, implied-volatility estimation, surface features, event studies, strategy backtesting, and options P&L attribution (most features in progress).

The initial research universe is **SPY**, chosen since it is a highly liquid ETF.

## Current Status:

Repo being set-up along with environment and basic pricing models and implied-volatility solver. Typer CLI interfaced with ordinary typed parameters included along with DuckDB table definitions, RUff linting/formattting, and a number of unit and integration tests.

## Startup

```bash
python -m venv .venv
```

Activate the environment:

# Windows PowerShell
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

- Ratesand divided yields are continuously compounded annual rate
- Time to expiry in years
- Black-Scholes uses a spot and continuous divided yield
- Black-76 uses a forward and discounts the expected payoff at risk-free rate
- Vega reported per `1.00` absolute volatility change; divisibl by `100` for sensitivity to a single volatility point
- Theta is annual time decay; Black-76 theta holds the forward as fixed
- `intrinsic-value` returns the immediate-exercise payoff w/o discounting (can be negative for European option)
- Implied-volatility solver validates European no-arbitrage bounds prior to attempting inversion

## Implied-volatility solver defined behavior

The solver returns an `ImpliedVolResult` rather than raising for expected quote-quality failures.

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

## Completion criterion

`tests/test_acceptance.py` verifies the Week 1 acceptance path:

1. Generate a Black-Scholes option price.
2. Recover the original volatility.
3. Compare analytic delta and vega with central finite differences.

## TODO

Next add ingestion adapters, quote cleaning, forward estimation from option pairs, rate/divided alignment, Parquet/DuckDB storage for SPY option chains