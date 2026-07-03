.PHONY: install lint format test check

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