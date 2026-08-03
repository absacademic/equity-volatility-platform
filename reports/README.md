# Reports

Generated option outputs are written to `reports/generated/`:

- `data-quality-<run_id>.md` : acceptance, rejection, and alignment summary
- `rejection-summary-<run_id>.csv` : rejection reason counts

Reports are reproducible from the ingestion command.

Event-study reports are written inside the selected `processed/event-studies/` output directory:

- `event-study-report.json`: run settings, counts, methods, and conclusion
- `research-conclusion.md`: held-out statistical and economic conclusion
- `summary-analysis.csv` and `regime-comparison.csv`: exploratory summaries
- `model-performance.csv`, `walk-forward-performance.csv`, and `strategy-summary.csv`: out-of-sample diagnostics

The finalized research design is documented in `reports/research_report.md`. Generated strategy reports are written to `data/processed/strategies/<run>/strategy-report.md` unless another output directory is supplied.
