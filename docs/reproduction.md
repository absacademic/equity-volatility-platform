# Reproduction commands

## Full deterministic workflow

Generate the 2018-2025 multi-asset sample.

```bash
vol-platform synthetic-week6 --output-dir data/interim/week6-demo
```

Run the exact strategy backtest.

```bash
vol-platform strategy-backtest data/interim/week6-demo/synthetic-week6-signals.csv \
  --option-quotes data/interim/week6-demo/synthetic-week6-option-quotes.csv \
  --underlying data/interim/week6-demo/synthetic-week6-underlying.csv \
  --config configs/week6-example.yml \
  --output-dir data/processed/strategies/week6-demo
```

Run the event study separately for each product.

```bash
vol-platform event-study data/interim/week6-demo/synthetic-week6-surface-features.csv \
  --events data/interim/week6-demo/synthetic-week6-events.csv \
  --underlying data/interim/week6-demo/synthetic-week6-underlying.csv \
  --symbol SPY \
  --config configs/week6-example.yml \
  --output-dir data/processed/event-studies/week6-spy
```

Repeat the command with `--symbol AAPL` and `--symbol XSP`.

On systems with Make installed:

```bash
make demo-week6
```

## Windows PowerShell

```powershell
vol-platform synthetic-week6 --output-dir data/interim/week6-demo

vol-platform strategy-backtest data/interim/week6-demo/synthetic-week6-signals.csv `
  --option-quotes data/interim/week6-demo/synthetic-week6-option-quotes.csv `
  --underlying data/interim/week6-demo/synthetic-week6-underlying.csv `
  --config configs/week6-example.yml `
  --output-dir data/processed/strategies/week6-demo
```

## Barrier-option Monte Carlo

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

The command returns the simulated price, standard error, 95% confidence interval, path count, monitoring steps, and barrier-hit probability.

## All checks

```bash
ruff check .
ruff format --check .
pytest --cov=vol_platform --cov-report=term-missing
```
