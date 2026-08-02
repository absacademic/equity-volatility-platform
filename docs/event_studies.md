# Event studies and event-strategy backtests

Event-studies test whether a surface observed before a scheduled market event helps explain whether the option market overestimated or underestimated the subsequent move.

The pipeline is designed for earnings events on individual equities and for CPI releases, Federal Reserve meetings, or other dated market events on index products. The default project universe remains SPY, so the synthetic demonstration uses CPI and FOMC events.

## Inputs

The `event-study` command requires three tables:

- A Week 4 daily surface-feature table in CSV or Parquet format.
- An event CSV containing `event_id`, `event_type`, `event_timestamp`, `known_timestamp`, `title`, `symbols`, `source`, and `expected`.
- An underlying-price CSV containing `timestamp`, `symbol`, and `last` or another supported price column.

Each event is classified as `pre_market`, `regular_session`, `after_hours`, or `market_closed`. After-hours and closed-market events use the next weekday as the reaction date. A surface observation is eligible only when its quote timestamp is strictly earlier than the event timestamp.

## Synthetic demonstration

Generate deterministic CPI and FOMC inputs, then run the full Week 5 workflow:

```bash
vol-platform synthetic-event-study --output-dir data/interim/week5-demo

vol-platform event-study data/interim/week5-demo/synthetic-week5-surface-features.csv \
  --events data/interim/week5-demo/synthetic-week5-events.csv \
  --underlying data/interim/week5-demo/synthetic-week5-underlying.csv \
  --output-dir data/processed/event-studies/demo
```

On systems with Make installed, run `make demo-event-study`. On Windows PowerShell, run the two `vol-platform` commands directly and replace each trailing backslash with a backtick.

## Event windows and features

The default window runs from 20 trading days before through 5 trading days after the reaction date. The selected pre-event surface is the valid expiration closest to 30 calendar days to expiry.

The baseline model uses:

- ATM implied volatility
- 25-delta downside skew
- ATM term-structure slope
- option-volume change
- open-interest change
- point-in-time ATM IV percentile
- a surface-dislocation score built from available rolling z-scores

The expected one-day absolute move is approximated by

```text
ATM IV × sqrt(2 / pi) / sqrt(252)
```

This converts annualized ATM volatility into the expected absolute return under a one-day normal-return approximation.

## Outcomes

The output dataset includes:

- signed and absolute event returns
- expected minus realized move
- an overestimate indicator
- post-event ATM IV collapse
- changes in ATM volatility and skew
- gross and net daily ATM-straddle approximations
- realized-move, implied-move, IV-change, and transaction-cost P&L components

The return measurement is session-aware but uses daily underlying closes. The straddle calculation is a transparent research approximation with a configurable vega scale. Exact contract-level P&L requires actual option entry and exit quotes, Greeks, hedge trades, contract multipliers, and execution timestamps.

## Models and validation

The pipeline fits a standardized linear regression for the expected-minus-realized move and a logistic regression for the overestimate indicator. Missing feature values are imputed using training-sample medians, and scaling parameters are estimated from the training period only.

Events are split chronologically into training, validation, and test periods. The pipeline reports coefficients, approximate 95% confidence intervals, validation and test metrics, coefficient stability after adding the validation period, and expanding walk-forward predictions.

## Backtest

The baseline strategy shorts a straddle when the linear model predicts overestimation and buys a straddle when it predicts underestimation. Estimated transaction costs include a fixed round-trip cost and a spread-based component. Results are compared with an always-short-straddle baseline.

The research conclusion is generated from held-out model performance and cost-adjusted test results. A negative conclusion is preferred when apparent predictability disappears after costs or lacks stability.

## Main outputs

The output directory contains point-in-time events, event windows, the modeling dataset, summaries, regime comparisons, model coefficients and performance, walk-forward results, strategy results, P&L attribution, plots, a DuckDB database, a JSON report, and `research-conclusion.md`.
