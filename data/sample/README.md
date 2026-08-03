# Sample data

Run `vol-platform synthetic-week6 --output-dir data/interim/week6-demo` to create the deterministic sample files used by the final event-study and strategy workflows.

The generator creates a 2018-2025 panel for AAPL, SPY, and XSP with CPI, FOMC, earnings, and derived large-market-move events. It also creates daily surface features, underlying prices, exact call-put quote histories, and chronological strategy signals.

The generated values are synthetic. They are intended for tests and workflow reproduction ONLY.
