# End-to-end event study, baseline modeling, and strategy backtest pipeline

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from vol_platform.config import load_config
from vol_platform.data.adapters import EventCSVAdapter, UnderlyingPriceCSVAdapter
from vol_platform.event_study.analysis import (
    add_regimes,
    build_research_conclusion,
    build_summary_tables,
)
from vol_platform.event_study.backtest import run_event_strategy_backtest
from vol_platform.event_study.datasets import (
    build_event_windows,
    normalize_point_in_time_events,
    read_table,
    select_pre_event_surface_features,
)
from vol_platform.event_study.models import DEFAULT_FEATURES, run_baseline_models
from vol_platform.event_study.outcomes import calculate_event_outcomes
from vol_platform.event_study.plots import create_event_study_plots


@dataclass(frozen=True, slots=True)
class EventStudyResult:
    output_dir: Path
    point_in_time_events: Path
    event_windows: Path
    event_dataset: Path
    summary_analysis: Path
    regime_comparison: Path
    model_coefficients: Path
    model_performance: Path
    coefficient_stability: Path
    walk_forward_results: Path
    walk_forward_performance: Path
    strategy_backtest: Path
    strategy_summary: Path
    pnl_attribution: Path
    conclusion: Path
    report: Path
    database: Path
    plots: tuple[Path, ...]


def _settings(config: object) -> dict[str, Any]:
    value = getattr(config, "event_study", {})
    return value if isinstance(value, dict) else {}


def _write_pair(frame: pl.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    frame.write_parquet(parquet_path)
    frame.write_csv(csv_path)


def _enrich_windows(
    windows: pl.DataFrame,
    underlying: pl.DataFrame,
    daily_features: pl.DataFrame,
    *,
    symbol: str,
) -> pl.DataFrame:
    price_by_date: dict[object, float] = {}
    for row in underlying.sort("timestamp").iter_rows(named=True):
        if str(row.get("symbol", "")).upper() != symbol.upper():
            continue
        price = row.get("underlying_price", row.get("last"))
        if row.get("timestamp") is not None and price is not None:
            price_by_date[row["timestamp"].date()] = float(price)
    feature_by_date: dict[object, dict[str, Any]] = {}
    for row in daily_features.sort("time_to_expiry").iter_rows(named=True):
        if str(row.get("symbol", "")).upper() != symbol.upper():
            continue
        current = feature_by_date.get(row.get("quote_date"))
        if current is None or abs(float(row.get("time_to_expiry", 0.0)) - 30.0 / 365.0) < abs(
            float(current.get("time_to_expiry", 0.0)) - 30.0 / 365.0
        ):
            feature_by_date[row.get("quote_date")] = row
    rows = []
    for row in windows.iter_rows(named=True):
        date_value = row["window_date"]
        feature = feature_by_date.get(date_value, {})
        row["underlying_price"] = price_by_date.get(date_value)
        row["atm_implied_volatility"] = feature.get("atm_implied_volatility")
        row["downside_skew_25"] = feature.get("downside_skew_25")
        rows.append(row)
    return pl.DataFrame(rows) if rows else windows


def _create_database(root: Path, tables: dict[str, Path]) -> Path:
    database = root / "event-study.duckdb"
    connection = duckdb.connect(str(database))
    try:
        for name, path in tables.items():
            escaped = str(path).replace("'", "''")
            connection.execute(
                f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{escaped}')"
            )
    finally:
        connection.close()
    return database


def run_event_study(
    daily_features_file: str | Path,
    events_file: str | Path,
    underlying_file: str | Path,
    *,
    symbol: str = "SPY",
    config_path: str | Path = "configs/base.yml",
    output_dir: str | Path | None = None,
) -> EventStudyResult:
    config = load_config(config_path)
    settings = _settings(config)
    timezone = str(config.project.get("timezone", "America/New_York"))
    daily_features = read_table(daily_features_file)
    events_raw = EventCSVAdapter().read(events_file)
    underlying = UnderlyingPriceCSVAdapter().read(underlying_file)

    events = normalize_point_in_time_events(events_raw, timezone=timezone)
    trading_dates = [
        row["timestamp"].date()
        for row in underlying.iter_rows(named=True)
        if row.get("timestamp") is not None
    ]
    windows = build_event_windows(
        events,
        trading_dates,
        pre_days=int(settings.get("pre_event_days", 20)),
        post_days=int(settings.get("post_event_days", 5)),
    )
    windows = _enrich_windows(windows, underlying, daily_features, symbol=symbol)
    event_features = select_pre_event_surface_features(
        events,
        daily_features,
        symbol=symbol,
        target_dte_days=int(settings.get("target_dte_days", 30)),
        annualization_days=int(settings.get("annualization_days", 252)),
    )
    outcomes = calculate_event_outcomes(
        event_features,
        windows,
        underlying,
        daily_features,
        annualization_days=int(settings.get("annualization_days", 252)),
        fixed_cost_bps=float(settings.get("fixed_cost_bps", 8.0)),
        spread_cost_multiplier=float(settings.get("spread_cost_multiplier", 0.5)),
        vega_scale=float(settings.get("vega_scale", 0.25)),
    )
    outcomes = add_regimes(outcomes)
    minimum_walk_forward_train = int(settings.get("minimum_walk_forward_train", 12))
    minimum_events = max(5, minimum_walk_forward_train + 1)
    if outcomes.height < minimum_events:
        raise ValueError(
            "event study requires at least "
            f"{minimum_events} complete events; found {outcomes.height}"
        )
    model_outputs = run_baseline_models(
        outcomes,
        features=tuple(settings.get("features", DEFAULT_FEATURES)),
        train_fraction=float(settings.get("train_fraction", 0.60)),
        validation_fraction=float(settings.get("validation_fraction", 0.20)),
        ridge=float(settings.get("ridge", 1.0e-6)),
        minimum_walk_forward_train=minimum_walk_forward_train,
    )
    backtest, attribution, strategy_summary = run_event_strategy_backtest(
        model_outputs.dataset
    )
    summary, regimes = build_summary_tables(backtest)

    default_root = Path(config.paths["processed_data"]) / "event-studies" / symbol.lower()
    root = Path(output_dir or default_root)
    root.mkdir(parents=True, exist_ok=True)
    plots_dir = root / "plots"

    paths = {
        "events": root / "point-in-time-events.parquet",
        "windows": root / "event-windows.parquet",
        "dataset": root / "event-study-dataset.parquet",
        "walk_forward_results": root / "walk-forward-results.parquet",
        "strategy_backtest": root / "strategy-backtest.parquet",
        "pnl_attribution": root / "pnl-attribution.parquet",
    }
    _write_pair(events, paths["events"], root / "point-in-time-events.csv")
    _write_pair(windows, paths["windows"], root / "event-windows.csv")
    _write_pair(backtest, paths["dataset"], root / "event-study-dataset.csv")
    _write_pair(
        model_outputs.walk_forward_predictions,
        paths["walk_forward_results"],
        root / "walk-forward-results.csv",
    )
    _write_pair(
        backtest, paths["strategy_backtest"], root / "strategy-backtest.csv"
    )
    _write_pair(
        attribution, paths["pnl_attribution"], root / "pnl-attribution.csv"
    )

    summary_path = root / "summary-analysis.csv"
    regimes_path = root / "regime-comparison.csv"
    coefficients_path = root / "model-coefficients.csv"
    performance_path = root / "model-performance.csv"
    stability_path = root / "coefficient-stability.csv"
    walk_performance_path = root / "walk-forward-performance.csv"
    strategy_summary_path = root / "strategy-summary.csv"
    summary.write_csv(summary_path)
    regimes.write_csv(regimes_path)
    model_outputs.coefficients.write_csv(coefficients_path)
    model_outputs.performance.write_csv(performance_path)
    model_outputs.stability.write_csv(stability_path)
    model_outputs.walk_forward_performance.write_csv(walk_performance_path)
    strategy_summary.write_csv(strategy_summary_path)

    synthetic = any(
        "synthetic" in str(value).lower()
        for value in events["source"].drop_nulls().to_list()
    )
    conclusion_text = build_research_conclusion(
        model_outputs.performance,
        strategy_summary,
        synthetic=synthetic,
    )
    conclusion_path = root / "research-conclusion.md"
    conclusion_path.write_text(conclusion_text, encoding="utf-8")

    plot_paths = create_event_study_plots(
        backtest,
        model_outputs.coefficients,
        backtest,
        plots_dir,
    )
    database = _create_database(root, paths)
    report_path = root / "event-study-report.json"
    report_path.write_text(
        json.dumps(
            {
                "symbol": symbol.upper(),
                "event_count": events.height,
                "modeled_event_count": backtest.height,
                "point_in_time_invalid_count": events.filter(
                    ~pl.col("point_in_time_valid")
                ).height,
                "window": {
                    "pre_event_days": int(settings.get("pre_event_days", 20)),
                    "post_event_days": int(settings.get("post_event_days", 5)),
                },
                "features": list(settings.get("features", DEFAULT_FEATURES)),
                "pnl_method": "daily_atm_straddle_approximation",
                "synthetic_inputs": synthetic,
                "conclusion": conclusion_text.strip(),
                "plots": [str(path) for path in plot_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return EventStudyResult(
        output_dir=root,
        point_in_time_events=paths["events"],
        event_windows=paths["windows"],
        event_dataset=paths["dataset"],
        summary_analysis=summary_path,
        regime_comparison=regimes_path,
        model_coefficients=coefficients_path,
        model_performance=performance_path,
        coefficient_stability=stability_path,
        walk_forward_results=paths["walk_forward_results"],
        walk_forward_performance=walk_performance_path,
        strategy_backtest=paths["strategy_backtest"],
        strategy_summary=strategy_summary_path,
        pnl_attribution=paths["pnl_attribution"],
        conclusion=conclusion_path,
        report=report_path,
        database=database,
        plots=tuple(plot_paths),
    )
