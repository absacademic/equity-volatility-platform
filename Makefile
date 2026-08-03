.PHONY: install lint format test check demo-ingest demo-surface demo-event-study demo-week6 demo-barrier reproduce docker-build

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


demo-week6:
	vol-platform synthetic-week6 --output-dir data/interim/week6-demo
	vol-platform strategy-backtest data/interim/week6-demo/synthetic-week6-signals.csv \
		--option-quotes data/interim/week6-demo/synthetic-week6-option-quotes.csv \
		--underlying data/interim/week6-demo/synthetic-week6-underlying.csv \
		--config configs/week6-example.yml \
		--output-dir data/processed/strategies/week6-demo


demo-barrier:
	vol-platform monte-carlo-barrier --spot 100 --strike 100 --barrier 125 \
		--time 1 --rate 0.04 --vol 0.25 --type call \
		--barrier-type up_and_out --paths 100000 --steps 252 --seed 7


reproduce: check demo-ingest demo-surface demo-event-study demo-week6 demo-barrier


docker-build:
	docker build -t equity-volatility-platform:0.6.0 .
