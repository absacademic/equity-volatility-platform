from pathlib import Path

import duckdb
import polars as pl
from typer.testing import CliRunner

from vol_platform.cli import app
from vol_platform.surface.pipeline import run_surface_analysis
from vol_platform.surface.synthetic import write_synthetic_inputs


def _config(tmp_path: Path) -> Path:
    config = tmp_path / "config.yml"
    config.write_text(
        f"""project:
  name: test
  timezone: America/New_York
  day_count_basis: 365.0
universe:
  symbols: [SPY]
pricing:
  default_rate: 0.04
  default_dividend_yield: 0.012
  iv: {{}}
paths:
  raw_data: {tmp_path / "raw"}
  interim_data: {tmp_path / "interim"}
  processed_data: {tmp_path / "processed"}
  reports: {tmp_path / "reports"}
surface:
  spline_smoothing: 1.0e-7
  forward:
    max_pairs: 5
    maximum_atm_distance: 0.10
""",
        encoding="utf-8",
    )
    return config


def test_week_three_completion_criterion(tmp_path: Path) -> None:
    chain, rates = write_synthetic_inputs(tmp_path / "inputs")
    result = run_surface_analysis(
        chain,
        rates_file=rates,
        config_path=_config(tmp_path),
        output_dir=tmp_path / "surface-output",
    )

    assert result.input_rows == 54
    assert result.iv_rows == 54
    assert result.successful_fits == 24
    assert result.failed_fits == 0
    assert result.implied_volatility_dataset.exists()
    assert result.forward_summary.exists()
    assert result.model_comparison.exists()
    assert result.report.exists()
    assert result.database.exists()
    assert {path.name for path in result.plots} == {
        "smile.png",
        "residuals.png",
        "surface.png",
        "bid_ask_band.png",
        "atm_term_structure.png",
    }

    iv_data = pl.read_parquet(result.implied_volatility_dataset)
    comparison = pl.read_csv(result.model_comparison)
    assert iv_data["mid_implied_volatility"].null_count() == 0
    assert comparison.height == 8

    connection = duckdb.connect(str(result.database), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM forward_estimates").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM model_comparison").fetchone()[0] == 8
    finally:
        connection.close()


def test_surface_cli_runs_one_command(tmp_path: Path) -> None:
    chain, rates = write_synthetic_inputs(tmp_path / "inputs")
    result = CliRunner().invoke(
        app,
        [
            "surface",
            str(chain),
            "--rates",
            str(rates),
            "--config",
            str(_config(tmp_path)),
            "--output-dir",
            str(tmp_path / "cli-output"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"successful_fits": 24' in result.output
