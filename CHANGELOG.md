# Changelog

## 0.6.0 - Project completion (relative)

### Event research

- Added a combined CPI, FOMC, earnings, and derived large-market-move event-history builder
- Added a stability gate that permits the nonlinear polynomial-ridge comparison only after the chronological linear baseline passes test, coefficient-sign, and walk-forward checks
- Preserved the W5 approximation as a signal-research diagnostic while directing final execution conclusions to the exact contract-level strategy pipeline.

### Final strategy

- Added an event-conditioned, delta-hedged at-the-money straddle strategy tied to the Week 5 expected-minus-realized-move prediction
- Added exact call-put contract selection, bid/ask execution, configurable slippage, commissions, hedge costs, financing, capital limits, overlap controls, and auditable trade rejections
- Added initial and periodic delta hedging
- Added option, hedge, transaction-cost, financing, gross, net, and midpoint-upper-bound P&L
- Added delta, gamma, vega, theta, and residual option-P&L attribution.
- Added return, volatility, Sharpe ratio, drawdown, win rate, turnover, cost drag, tail loss, expected shortfall, and midpoint-optimism metrics
- Added sensitivity tests for signal thresholds, holding periods, hedge frequencies, slippage, liquidity, spread limits, years, and volatility regimes
- Added comparisons by symbol, asset type, event type, chronological period, year, and regime

### Reproducibility

- Added deterministic 2018-2025 AAPL, SPY, and XSP sample-data generation
- Added a Week 6 configuration, Dockerfile, Docker ignore file, CI workflow, architecture diagram, installation guide, reproduction guide, strategy guide, real-history data guide, and data disclaimer
- Added a complete research report and updated generated-report documentation
- Added Make targets for the Week 6 demo, barrier-option demo, Docker build, and complete reproduction workflow
- Added the missing sample SPY option-quote file used by the ingestion demo
- Added automated Week 6 event, nonlinear-model, strategy, pipeline, and Monte Carlo tests

### Exotic pricing

- Added a CLI-accessible, discretely monitored barrier-option Monte Carlo pricer with up/down and in/out variants, rebates, antithetic sampling, confidence intervals, standard errors, and barrier-hit probabilities
