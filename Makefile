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