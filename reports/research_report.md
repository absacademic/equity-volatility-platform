# Equity volatility research report

## Research objective

The project studies whether point-in-time equity-option surface features help explain when option-implied event moves overstate or understate subsequent realized moves. It then translates the chronological signal into one explicit strategy: an event-conditioned, delta-hedged at-the-money straddle.

## Research design

The event calendar supports CPI, FOMC, earnings, and large-market-move events. Surface features are selected strictly before each event. Chronological train, validation, test, and walk-forward periods prevent random future-to-past leakage. Linear and logistic models remain the primary baselines. A degree-two polynomial ridge model is evaluated only when the baseline passes minimum out-of-sample directional-accuracy, coefficient-sign-stability, and walk-forward checks.

## Strategy implementation

The final backtest selects an exact call-put pair rather than approximating a daily straddle return. It defines entry and exit dates, target DTE, ATM selection, position size, capital limits, tradable event types, liquidity filters, and overlap rules. Option purchases use ask-side execution and sales use bid-side execution with additional slippage. The engine includes commissions, hedge turnover costs, and financing.

Delta is hedged at entry and periodically through exit. Each trade separates option P&L, hedge P&L, costs, financing, and net P&L. Option changes are attributed to delta, gamma, vega, theta, and a residual.

## Robustness analysis

The sensitivity grid changes signal thresholds, holding periods, hedge frequencies, spread and slippage assumptions, open-interest filters, year ranges, and volatility regimes. Summary tables compare equities, ETFs, and index products as well as event types and chronological periods.

## Reproducible evidence

The repository includes a deterministic 2018-2025 sample for AAPL, SPY, and XSP. It covers CPI, FOMC, earnings, and derived large-market-move events and includes exact option-contract quote histories. This sample validates the complete pipeline and allows automated testing.

The sample is synthetic. A real empirical conclusion requires legally obtained historical option bid/ask data, correct event-known timestamps, corporate-action handling, settlement and exercise-style adjustments, and a defensible index hedge.

## Final conclusion

The repository is complete as a reproducible research platform: another user can install it, run automated checks, generate sample data, reproduce a surface and event study, run the exact strategy backtest, inspect transaction-cost and Greek attribution, and price a barrier option through Monte Carlo from the CLI. The software separates workflow validation from empirical claims.
