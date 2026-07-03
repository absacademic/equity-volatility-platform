from __future__ import annotations

import json
from pathlib import Path

import duckdb
import polars as pl
from typer.testing import CliRunner

from vol_platform.cli import app
from vol_platform.data.pipeline import run_ingestion

SAMPLE = (
    "quote_timestamp,option_symbol,underlying_symbol,expiration,strike,option_type,"
    "bid,ask,volume,open_interest,underlying_timestamp,underlying_last\n"
    "2026-07-01T14:30:00Z,A,SPY,2026-07-17,600,call,4.0,4.2,10,20,"
    "2026-07-01T14:29:59Z,600\n"
    "2026-07-01T14:30:01Z,B,SPY,2026-07-17,610,call,0.0,0.3,0,0,"
    "2026-07-01T14:29:59Z,600\n"
)


def _config(tmp_path: Path) -> Path:
    config = tmp_path / "config.yml"
    config.write_text(
        f"""project:
  name: test
universe:
  symbols: [SPY]
pricing:
  default_rate: 0.04
  default_dividend_yield: 0.0
  iv: {{}}
paths:
  raw_data: {tmp_path / "raw"}
  interim_data: {tmp_path / "interim"}
  processed_data: {tmp_path / "processed"}
  reports: {tmp_path / "reports"}
data:
  schema_version: '1.0.0'
  alignment:
    max_underlying_staleness_seconds: 300
  cleaning:
    max_relative_spread: 0.50
    min_volume: 1
    min_open_interest: 1
    min_moneyness: 0.50
    max_moneyness: 1.50
""",
        encoding="utf-8",
    )
    return config


def test_week_two_completion_criterion(tmp_path: Path) -> None:
    raw = tmp_path / "quotes.csv"
    raw.write_text(SAMPLE, encoding="utf-8")
    config = _config(tmp_path)

    result = run_ingestion(raw, config_path=config)

    assert result.input_rows == 2
    assert result.clean_rows == 1
    assert result.rejected_rows == 1
    assert result.report.exists()
    assert result.manifest.exists()
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["inputs"][0]["sha256"]
    assert manifest["inputs"][0]["row_count"] == 2
    assert manifest["inputs"][0]["schema_version"] == "1.0.0"

    clean_files = list((result.output_dir / "options" / "clean").rglob("*.parquet"))
    rejected_files = list((result.output_dir / "options" / "rejected").rglob("*.parquet"))
    assert clean_files and rejected_files
    assert pl.read_parquet(clean_files[0])["is_valid"][0]
    assert not pl.read_parquet(rejected_files[0])["is_valid"][0]

    connection = duckdb.connect(str(result.database), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM clean_quotes").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM rejected_quotes").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM daily_summaries").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM expiration_chains").fetchone()[0] == 1
    finally:
        connection.close()


def test_ingest_cli_runs_one_command(tmp_path: Path) -> None:
    raw = tmp_path / "quotes.csv"
    raw.write_text(SAMPLE, encoding="utf-8")
    config = _config(tmp_path)
    result = CliRunner().invoke(app, ["ingest", str(raw), "--config", str(config)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["clean_rows"] == 1
    assert payload["rejected_rows"] == 1
