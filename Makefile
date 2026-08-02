.PHONY: install lint format test check demo-ingest demo-surface demo-event-study

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff check . --fix
	ruff format .

test:
	pytest --cov=vol_platform --cov-report=term-missing

check:
	ruff check .
	ruff format --check .
	pytest --cov=vol_platform --cov-report=term-missing

demo-ingest:
	vol-platform ingest data/raw/sample_spy_quotes.csv \
		--underlying data/raw/sample_spy_underlying.csv \
		--rates data/raw/sample_spy_rates.csv \
		--events data/raw/sample_spy_events.csv

demo-surface:
	vol-platform synthetic-chain --output-dir data/interim/week4-demo
	vol-platform surface data/interim/week4-demo/synthetic-clean-chain.parquet \
		--rates data/interim/week4-demo/synthetic-rates.csv \
		--dividends data/interim/week4-demo/synthetic-dividends.csv \
		--events data/interim/week4-demo/synthetic-events.csv \
		--underlying-history data/interim/week4-demo/synthetic-underlying-history.csv \
		--output-dir data/processed/surfaces/demo

demo-event-study:
	vol-platform synthetic-event-study --output-dir data/interim/week5-demo
	vol-platform event-study data/interim/week5-demo/synthetic-week5-surface-features.csv \
		--events data/interim/week5-demo/synthetic-week5-events.csv \
		--underlying data/interim/week5-demo/synthetic-week5-underlying.csv \
		--output-dir data/processed/event-studies/demo
