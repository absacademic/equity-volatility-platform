from pathlib import Path

import duckdb
import polars as pl

from vol_platform.event_study.pipeline import run_event_study
from vol_platform.event_study.synthetic import write_synthetic_event_study_inputs


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "week5-config.yml"
    path.write_text(
        f"""project:
  name: week5-test
  timezone: America/New_York
universe:
  symbols: [SPY]
pricing:
  default_model: black_scholes
  default_rate: 0.04
  default_dividend_yield: 0.012
  iv: {{}}
paths:
  raw_data: {tmp_path / 'raw'}
  interim_data: {tmp_path / 'interim'}
  processed_data: {tmp_path / 'processed'}
  reports: {tmp_path / 'reports'}
event_study:
  pre_event_days: 20
  post_event_days: 5
  target_dte_days: 30
  train_fraction: 0.60
  validation_fraction: 0.20
  minimum_walk_forward_train: 12
  fixed_cost_bps: 8.0
  spread_cost_multiplier: 0.50
""",
        encoding="utf-8",
    )
    return path


def test_week_five_completion_criterion(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    events, underlying, features = write_synthetic_event_study_inputs(inputs)
    result = run_event_study(
        features,
        events,
        underlying,
        config_path=_config(tmp_path),
        output_dir=tmp_path / "week5-output",
    )

    for path in (
        result.point_in_time_events,
        result.event_windows,
        result.event_dataset,
        result.summary_analysis,
        result.regime_comparison,
        result.model_coefficients,
        result.model_performance,
        result.coefficient_stability,
        result.walk_forward_results,
        result.walk_forward_performance,
        result.strategy_backtest,
        result.strategy_summary,
        result.pnl_attribution,
        result.conclusion,
        result.report,
        result.database,
    ):
        assert path.exists()

    dataset = pl.read_parquet(result.event_dataset)
    performance = pl.read_csv(result.model_performance)
    assert dataset.height >= 40
    assert {"train", "validation", "test"}.issubset(set(dataset["period"].to_list()))
    assert {"linear", "logistic"} == set(performance["model"].to_list())
    assert len(result.plots) == 4
    assert "Synthetic demonstration conclusion" in result.conclusion.read_text(
        encoding="utf-8"
    )

    connection = duckdb.connect(str(result.database), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] > 0
        assert connection.execute("SELECT COUNT(*) FROM dataset").fetchone()[0] == dataset.height
        attribution_count = connection.execute(
            "SELECT COUNT(*) FROM pnl_attribution"
        ).fetchone()[0]
        assert attribution_count == dataset.height
    finally:
        connection.close()
