# Architecture

```mermaid
flowchart LR
    A[Raw option, underlying, rate, dividend, and event files] --> B[Adapters and point-in-time cleaning]
    B --> C[Parquet partitions and DuckDB views]
    C --> D[Forward and implied-volatility engine]
    D --> E[Smile and surface models]
    E --> F[No-arbitrage controls and standardized features]
    F --> G[Chronological event-study baseline]
    G --> H{Baseline stable?}
    H -- No --> I[Keep linear and logistic results only]
    H -- Yes --> J[Gated polynomial ridge comparison]
    G --> K[Event-conditioned straddle signal]
    K --> L[Exact contract selection]
    L --> M[Bid/ask execution and periodic delta hedging]
    M --> N[P&L, Greeks, costs, financing, and risk metrics]
    N --> O[Sensitivity by threshold, holding period, costs, liquidity, year, regime, and asset type]
    P[Barrier-option Monte Carlo CLI] --> Q[Price, standard error, confidence interval, and knock probability]
```

The package is separated into five main areas:

- `vol_platform.data`: adapters, validation, alignment, storage, and data-quality reports.
- `vol_platform.pricing`: Black-Scholes, Black-76, Greeks, implied volatility, parity, and Monte Carlo pricing.
- `vol_platform.surface`: forwards, dividends, arbitrage diagnostics, smile models, standardized deltas, and surface features.
- `vol_platform.event_study`: point-in-time event windows, chronological models, stability checks, and research diagnostics.
- `vol_platform.strategy`: exact option-contract execution, delta hedging, P&L attribution, risk metrics, and sensitivity analysis.

Generated analytical tables are written to Parquet and CSV. DuckDB files provide local SQL access without requiring a server.
