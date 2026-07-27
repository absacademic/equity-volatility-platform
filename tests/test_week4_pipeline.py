from pathlib import Path

import duckdb
import polars as pl

from vol_platform.surface.pipeline import run_surface_analysis
from vol_platform.surface.synthetic import synthetic_clean_chain, write_synthetic_inputs


def _config(tmp_path: Path) -> Path:
    config = tmp_path / "week4-config.yml"
    config.write_text(
        f"""project:
  name: week4-test
  timezone: America/New_York
  day_count_basis: 365.0
universe:
  symbols: [SPY]
pricing:
  default_rate: 0.04
  default_dividend_yield: 0.012
  iv: {{}}
paths:
  raw_data: {tmp_path / 'raw'}
  interim_data: {tmp_path / 'interim'}
  processed_data: {tmp_path / 'processed'}
  reports: {tmp_path / 'reports'}
surface:
  spline_smoothing: 1.0e-7
  exercise_style: american
  dividends:
    use_forward_fallback: true
    exclude_early_exercise_risk: true
  standardized:
    extrapolation_limit: 0.15
  historical:
    realized_windows: [5, 20, 60]
    rolling_window: 20
""",
        encoding="utf-8",
    )
    return config


def test_week_four_completion_criterion(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    chain, rates = write_synthetic_inputs(inputs)
    result = run_surface_analysis(
        chain,
        rates_file=rates,
        dividends_file=inputs / "synthetic-dividends.csv",
        events_file=inputs / "synthetic-events.csv",
        underlying_history_file=inputs / "synthetic-underlying-history.csv",
        config_path=_config(tmp_path),
        output_dir=tmp_path / "week4-output",
    )

    assert result.arbitrage_report.exists()
    assert result.arbitrage_diagnostics.exists()
    assert result.surface_adjustments.exists()
    assert result.standardized_delta_points.exists()
    assert result.daily_feature_table.exists()
    assert {path.name for path in result.historical_plots} == {
        "historical_atm_iv.png",
        "historical_skew.png",
        "historical_vrp.png",
    }

    points = pl.read_csv(result.standardized_delta_points)
    features = pl.read_parquet(result.daily_feature_table)
    assert points.height == 15
    assert features.height == 3
    assert features.select(["quote_date", "symbol", "expiration"]).unique().height == 3
    assert features["realized_volatility_20d"].null_count() == 0
    assert features["event_count_to_expiry"].min() >= 1

    connection = duckdb.connect(str(result.database), read_only=True)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM standardized_delta_points"
        ).fetchone()[0] == 15
        assert connection.execute(
            "SELECT COUNT(*) FROM daily_volatility_features"
        ).fetchone()[0] == 3
    finally:
        connection.close()


def test_invalid_chain_still_produces_flagged_feature_row(tmp_path: Path) -> None:
    inputs = tmp_path / "invalid-inputs"
    _, rates = write_synthetic_inputs(inputs)
    chain = synthetic_clean_chain()
    first_expiration = chain["expiration"].min()
    invalid = chain.with_columns(
        pl.when(
            (pl.col("expiration") == first_expiration)
            & (pl.col("option_type") == "call")
            & (pl.col("strike") == 600.0)
        )
        .then(pl.lit(99.9))
        .otherwise(pl.col("bid"))
        .alias("bid"),
        pl.when(
            (pl.col("expiration") == first_expiration)
            & (pl.col("option_type") == "call")
            & (pl.col("strike") == 600.0)
        )
        .then(pl.lit(100.1))
        .otherwise(pl.col("ask"))
        .alias("ask"),
        pl.when(
            (pl.col("expiration") == first_expiration)
            & (pl.col("option_type") == "call")
            & (pl.col("strike") == 600.0)
        )
        .then(pl.lit(100.0))
        .otherwise(pl.col("mid"))
        .alias("mid"),
    )
    invalid_path = inputs / "invalid-chain.parquet"
    invalid.write_parquet(invalid_path)

    result = run_surface_analysis(
        invalid_path,
        rates_file=rates,
        config_path=_config(tmp_path),
        output_dir=tmp_path / "invalid-output",
    )
    features = pl.read_parquet(result.daily_feature_table)
    affected = features.filter(pl.col("expiration") == first_expiration)

    assert affected.height == 1
    assert not affected["chain_valid"][0]
    assert affected["material_arbitrage_violation_count"][0] > 0
