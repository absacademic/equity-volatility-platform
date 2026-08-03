## Strategy choice

The final strategy is an event-conditioned, delta-hedged at-the-money straddle. It is tied directly to the Week 5 target:

```text
expected move minus realized move
```

A positive prediction indicates that the option market is expected to overstate the event move, so the strategy sells the straddle. A negative prediction buys the straddle. Predictions inside the configured threshold do not create a trade.

## Trade definition

- Entry: one trading day before the event reaction date.
- Exit: one trading day after the reaction date by default.
- Option selection: a call and put with the same strike and expiration, closest to the underlying price and the target DTE.
- Sizing: fixed requested contracts, capped by maximum contracts, per-trade capital, and aggregate portfolio capital.
- Hedging: an initial delta hedge followed by periodic rebalancing at the configured trading-day frequency.
- Overlap: a symbol cannot open a new event position before its prior position exits.

Large-market-move events are included in the research calendar but are not traded by default because the event is known only after the move occurs. They can be used for descriptive event studies or for a separately defined post-event strategy.

## Execution

The engine does not treat midpoint fills as executable.

- Option purchases execute at the ask plus configured slippage.
- Option sales execute at the bid minus configured slippage.
- Illiquid contracts are rejected using volume, open-interest, spread, DTE, and quote-completeness filters.
- Commissions are charged per contract and per transaction.
- Underlying hedge turnover is charged in basis points.
- Financing is accrued on the option-and-hedge cash balance.
- Midpoint results are retained only as an upper bound.

## P&L

Each trade reports:

- Option P&L measured from midpoint marks, plus executable option P&L for audit.
- Hedge P&L from the actual hedge position across underlying price changes.
- Transaction costs, including option execution cost, commissions, and hedge costs.
- Financing P&L.
- Net P&L and return on estimated capital at risk.

Option P&L is attributed to delta, gamma, vega, theta, and residual error. Hedge P&L remains separate so the effect of delta neutralization is visible.

## Risk limits

The sample configuration limits contracts, capital per trade, aggregate capital across open positions, short-option margin, relative spread, DTE, volume, and open interest. It also rejects multiplier mismatches, incomplete quotes, and overlapping positions in the same symbol. These controls are research approximations and do not replace broker margin or live risk systems.

## Reported metrics

The pipeline reports portfolio total return from cumulative net P&L, event-level annualized return and volatility, Sharpe ratio, portfolio-equity drawdown, win rate, turnover, cost drag, fifth-percentile loss, expected shortfall, and midpoint optimism. It also retains a compounded per-trade return for comparison. Results are grouped by symbol, asset type, event type, chronological period, year, and volatility regime.
