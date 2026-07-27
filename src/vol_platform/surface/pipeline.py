# IV smile, arbitrage, standardized-feature, and historical pipeline

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from vol_platform.config import load_config
from vol_platform.data.adapters import (
    DividendCSVAdapter,
    EventCSVAdapter,
    RateCSVAdapter,
    UnderlyingPriceCSVAdapter,
)
from vol_platform.surface.arbitrage import (
    apply_surface_controls,
    build_arbitrage_diagnostics,
    build_arbitrage_report,
    mark_resolved_diagnostics,
)
from vol_platform.surface.dividends import (
    add_dividend_and_exercise_features,
    apply_dividend_forward_adjustments,
)
from vol_platform.surface.evaluation import evaluate_fits
from vol_platform.surface.expiration import exact_time_to_expiry
from vol_platform.surface.features import build_implied_volatility_dataset
from vol_platform.surface.forward import ForwardSettings, estimate_forwards
from vol_platform.surface.history import (
    add_event_linked_features,
    add_historical_comparisons,
    add_realized_volatility_features,
    build_underlying_history,
    create_historical_plots,
)
from vol_platform.surface.models import fit_all_smiles
from vol_platform.surface.plots import create_surface_plots
from vol_platform.surface.rates import discount_factor, interpolate_rate
from vol_platform.surface.standardized import (
    build_daily_volatility_features,
    interpolate_standardized_delta_points,
)


@dataclass(frozen=True, slots=True)
class SurfaceAnalysisResult:
    quote_date: date
    input_rows: int
    iv_rows: int
    successful_fits: int
    failed_fits: int
    output_dir: Path
    implied_volatility_dataset: Path
    forward_summary: Path
    model_comparison: Path
    report: Path
    arbitrage_report: Path
    arbitrage_diagnostics: Path
    surface_adjustments: Path
    standardized_delta_points: Path
    daily_feature_table: Path
    database: Path
    plots: tuple[Path, ...]
    historical_plots: tuple[Path, ...]


def _read_quotes(path: Path) -> pl.DataFrame:
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no Parquet files found under {path}")
        return pl.concat([pl.read_parquet(file) for file in files], how="diagonal_relaxed")
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    raise ValueError("surface analysis expects a clean Parquet file or directory")


def _read_adapter(path: str | Path | None, adapter: Any) -> pl.DataFrame | None:
    if path is None:
        return None
    selected = Path(path)
    if selected.suffix.lower() == ".parquet":
        return pl.read_parquet(selected)
    return adapter.read(selected, "surface_input")


def _surface_settings(config: Any) -> dict[str, Any]:
    extras = config.model_extra or {}
    return dict(extras.get("surface", {}))


def _add_rate_and_expiration_features(
    quotes: pl.DataFrame,
    rates: pl.DataFrame | None,
    *,
    timezone: str,
    day_count_basis: float,
    default_rate: float,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in quotes.iter_rows(named=True):
        time_to_expiry = exact_time_to_expiry(
            row["quote_timestamp"],
            row["expiration"],
            timezone=timezone,
            day_count_basis=day_count_basis,
        )
        if time_to_expiry <= 0.0:
            continue
        interpolated = interpolate_rate(
            rates,
            quote_date=row["quote_timestamp"].date(),
            expiration=row["expiration"],
            default_rate=float(row.get("risk_free_rate") or default_rate),
            day_count_basis=day_count_basis,
            currency=row.get("currency") or "USD",
        )
        enriched = dict(row)
        enriched.update(
            {
                "time_to_expiry": time_to_expiry,
                "interpolated_rate": interpolated.rate,
                "discount_factor": discount_factor(interpolated.rate, time_to_expiry),
                "interpolated_rate_as_of_date": interpolated.source_as_of_date,
                "used_default_rate": interpolated.used_default,
            }
        )
        rows.append(enriched)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _write_database(path: Path, tables: dict[str, pl.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    try:
        for name, frame in tables.items():
            connection.register(f"_{name}", frame.to_arrow())
            connection.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _{name}")
            connection.unregister(f"_{name}")
    finally:
        connection.close()


def _write_report(
    path: Path,
    quote_date: date,
    forwards: pl.DataFrame,
    comparison: pl.DataFrame,
    features: pl.DataFrame,
    plot_paths: list[Path],
    historical_plot_paths: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reliable = forwards.filter(pl.col("reliability") != "unreliable").height
    valid_chains = features.filter(pl.col("chain_valid")).height if not features.is_empty() else 0
    lines = [
        f"# Surface analysis: {quote_date}",
        "",
        "## Forward estimation",
        "",
        f"- Expirations estimated: {forwards.height}",
        f"- Expirations with usable reliability: {reliable}",
        "- Discrete dividends are present-valued before the theoretical forward comparison.",
        "- Missing parity forwards may use the configured dividend-adjusted spot fallback.",
        "",
        "## Quality-controlled feature rows",
        "",
        f"- Rows produced for this date: {features.height}",
        f"- Rows passing all material controls: {valid_chains}",
        "",
        "## Model comparison",
        "",
    ]
    if comparison.is_empty():
        lines.append("No smile model fit succeeded.")
    else:
        lines.extend(
            [
                "| Model | Weighting | Failed-fit rate | Average RMSE | "
                "Maximum residual | Coverage | Stability |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )

        def metric(value: object, digits: int) -> str:
            return f"{float(value):.{digits}f}" if value is not None else "n/a"

        for row in comparison.iter_rows(named=True):
            lines.append(
                f"| {row['model']} | {row['weighting']} | "
                f"{metric(row['failed_fit_rate'], 3)} | "
                f"{metric(row['average_rmse'], 6)} | "
                f"{metric(row['maximum_residual'], 6)} | "
                f"{metric(row['average_coverage'], 3)} | "
                f"{metric(row['average_stability'], 3)} |"
            )
    lines.extend(["", "## Current-date visualizations", ""])
    lines.extend(f"- `{plot.name}`" for plot in plot_paths)
    lines.extend(["", "## Historical visualizations", ""])
    lines.extend(f"- `{plot.name}`" for plot in historical_plot_paths)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _settings(defaults: dict[str, Any], name: str) -> dict[str, Any]:
    value = defaults.get(name, {})
    return dict(value) if isinstance(value, dict) else {}


def _analyze_one_date(
    quotes: pl.DataFrame,
    rates: pl.DataFrame | None,
    dividends: pl.DataFrame | None,
    *,
    project: dict[str, Any],
    pricing: Any,
    settings: dict[str, Any],
) -> dict[str, Any]:
    timezone = str(project.get("timezone", "America/New_York"))
    day_count_basis = float(project.get("day_count_basis", 365.0))
    exercise_style = str(settings.get("exercise_style", "american"))
    enriched = _add_rate_and_expiration_features(
        quotes,
        rates,
        timezone=timezone,
        day_count_basis=day_count_basis,
        default_rate=pricing.default_rate,
    )
    enriched = add_dividend_and_exercise_features(
        enriched,
        dividends,
        day_count_basis=day_count_basis,
        exercise_style=exercise_style,
    )

    forward_config = _settings(settings, "forward")
    forward_settings = ForwardSettings(
        max_pairs=int(forward_config.get("max_pairs", 5)),
        maximum_atm_distance=float(forward_config.get("maximum_atm_distance", 0.10)),
        reliable_relative_dispersion=float(
            forward_config.get("reliable_relative_dispersion", 0.005)
        ),
        minimum_average_quality=float(forward_config.get("minimum_average_quality", 0.50)),
    )
    forward_quotes = enriched.filter(~pl.col("early_exercise_risk"))
    forwards, pairs = estimate_forwards(forward_quotes, forward_settings)
    dividend_config = _settings(settings, "dividends")
    forwards = apply_dividend_forward_adjustments(
        forwards,
        enriched,
        use_fallback=bool(dividend_config.get("use_forward_fallback", True)),
    )

    solver_options = {
        "initial_volatility": pricing.iv.initial_volatility,
        "minimum_volatility": pricing.iv.minimum_volatility,
        "maximum_volatility": pricing.iv.maximum_volatility,
        "price_tolerance": pricing.iv.price_tolerance,
        "volatility_tolerance": pricing.iv.volatility_tolerance,
        "max_iterations": pricing.iv.max_iterations,
    }
    iv_data = build_implied_volatility_dataset(
        enriched,
        forwards,
        solver_options=solver_options,
        exclude_early_exercise_risk=bool(
            dividend_config.get("exclude_early_exercise_risk", True)
        ),
    )
    raw_fits = fit_all_smiles(
        iv_data,
        smoothing=float(settings.get("spline_smoothing", 1e-7)),
    )
    raw_details, _ = evaluate_fits(iv_data, raw_fits)
    arbitrage = _settings(settings, "arbitrage")
    diagnostics = build_arbitrage_diagnostics(
        iv_data,
        raw_details,
        raw_fits,
        price_tolerance=float(arbitrage.get("price_tolerance", 1e-6)),
        variance_tolerance=float(arbitrage.get("variance_tolerance", 1e-10)),
        calendar_tolerance=float(arbitrage.get("calendar_tolerance", 1e-6)),
        extrapolation_padding=float(arbitrage.get("extrapolation_padding", 0.15)),
        maximum_extrapolated_iv=float(arbitrage.get("maximum_extrapolated_iv", 3.0)),
        maximum_variance_multiple=float(arbitrage.get("maximum_variance_multiple", 8.0)),
    )
    fits, adjustments = apply_surface_controls(
        iv_data,
        raw_fits,
        variance_tolerance=float(arbitrage.get("variance_tolerance", 1e-10)),
        price_tolerance=float(arbitrage.get("price_tolerance", 1e-6)),
        calendar_tolerance=float(arbitrage.get("calendar_tolerance", 1e-6)),
        extrapolation_padding=float(arbitrage.get("extrapolation_padding", 0.15)),
        maximum_extrapolated_iv=float(arbitrage.get("maximum_extrapolated_iv", 3.0)),
        maximum_variance_multiple=float(arbitrage.get("maximum_variance_multiple", 8.0)),
    )
    diagnostics = mark_resolved_diagnostics(diagnostics, adjustments)
    details, comparison = evaluate_fits(iv_data, fits)
    standardized = _settings(settings, "standardized")
    points = interpolate_standardized_delta_points(
        iv_data,
        details,
        fits,
        extrapolation_limit=float(standardized.get("extrapolation_limit", 0.15)),
    )
    features = build_daily_volatility_features(points, iv_data, details, diagnostics)
    return {
        "enriched": enriched,
        "forwards": forwards,
        "pairs": pairs,
        "iv_data": iv_data,
        "fits": fits,
        "details": details,
        "comparison": comparison,
        "diagnostics": diagnostics,
        "adjustments": adjustments,
        "points": points,
        "features": features,
    }


def run_surface_analysis(
    clean_quotes: str | Path,
    *,
    rates_file: str | Path | None = None,
    dividends_file: str | Path | None = None,
    events_file: str | Path | None = None,
    underlying_history_file: str | Path | None = None,
    quote_date: date | str | None = None,
    config_path: str | Path = "configs/base.yml",
    output_dir: str | Path | None = None,
) -> SurfaceAnalysisResult:
    """Produce current-date surface outputs and point-in-time historical feature rows."""

    config = load_config(config_path)
    settings = _surface_settings(config)
    all_quotes = _read_quotes(Path(clean_quotes)).filter(pl.col("is_valid"))
    available_dates = all_quotes["quote_date"].unique().sort()
    if available_dates.is_empty():
        raise ValueError("no valid quote dates were found")
    selected_date = date.fromisoformat(quote_date) if isinstance(quote_date, str) else quote_date
    selected_date = selected_date or available_dates[-1]
    if selected_date not in available_dates.to_list():
        raise ValueError(f"no clean quotes found for {selected_date}")

    rates = _read_adapter(rates_file, RateCSVAdapter())
    dividends = _read_adapter(dividends_file, DividendCSVAdapter())
    events = _read_adapter(events_file, EventCSVAdapter())
    external_history = _read_adapter(underlying_history_file, UnderlyingPriceCSVAdapter())

    analyses: dict[date, dict[str, Any]] = {}
    for current_date in available_dates.to_list():
        if current_date > selected_date:
            continue
        current_quotes = all_quotes.filter(pl.col("quote_date") == current_date)
        analyses[current_date] = _analyze_one_date(
            current_quotes,
            rates,
            dividends,
            project=config.project,
            pricing=config.pricing,
            settings=settings,
        )

    selected = analyses[selected_date]
    all_points = pl.concat(
        [analysis["points"] for analysis in analyses.values()], how="diagonal_relaxed"
    )
    all_diagnostics = pl.concat(
        [analysis["diagnostics"] for analysis in analyses.values()], how="diagonal_relaxed"
    )
    all_adjustments = pl.concat(
        [analysis["adjustments"] for analysis in analyses.values()], how="diagonal_relaxed"
    )
    all_features = pl.concat(
        [analysis["features"] for analysis in analyses.values()], how="diagonal_relaxed"
    )
    underlying_history = build_underlying_history(all_quotes, external_history)
    historical = _settings(settings, "historical")
    all_features = add_realized_volatility_features(
        all_features,
        underlying_history,
        annualization_days=int(historical.get("annualization_days", 252)),
        windows=tuple(
            int(value)
            for value in historical.get("realized_windows", [5, 20, 60])
        ),
    )
    all_features = add_event_linked_features(
        all_features,
        events,
        timezone=str(config.project.get("timezone", "America/New_York")),
    )
    all_features = add_historical_comparisons(
        all_features,
        rolling_window=int(historical.get("rolling_window", 20)),
    )

    default_root = Path(config.paths["processed_data"]) / "surfaces" / str(selected_date)
    root = Path(output_dir or default_root)
    root.mkdir(parents=True, exist_ok=True)
    iv_path = root / "implied-volatility.parquet"
    forward_path = root / "forward-estimates.csv"
    pair_path = root / "forward-pairs.csv"
    details_path = root / "smile-fit-details.csv"
    comparison_path = root / "model-comparison.csv"
    diagnostics_path = root / "arbitrage-diagnostics.csv"
    adjustments_path = root / "surface-adjustments.csv"
    points_path = root / "standardized-delta-points.csv"
    features_path = root / "daily-volatility-features.parquet"
    features_csv_path = root / "daily-volatility-features.csv"

    selected["iv_data"].write_parquet(iv_path)
    selected["forwards"].write_csv(forward_path)
    selected["pairs"].write_csv(pair_path)
    selected["details"].write_csv(details_path)
    selected["comparison"].write_csv(comparison_path)
    all_diagnostics.write_csv(diagnostics_path)
    all_adjustments.write_csv(adjustments_path)
    all_points.write_csv(points_path)
    all_features.write_parquet(features_path)
    all_features.write_csv(features_csv_path)

    plots = create_surface_plots(
        selected["iv_data"], selected["details"], selected["fits"], root / "plots"
    )
    historical_plots = create_historical_plots(all_features, root / "plots" / "historical")
    database = root / "surface.duckdb"
    _write_database(
        database,
        {
            "implied_volatility_dataset": selected["iv_data"],
            "forward_estimates": selected["forwards"],
            "forward_pairs": selected["pairs"],
            "smile_fit_details": selected["details"],
            "model_comparison": selected["comparison"],
            "arbitrage_diagnostics": all_diagnostics,
            "surface_adjustments": all_adjustments,
            "standardized_delta_points": all_points,
            "daily_volatility_features": all_features,
            "underlying_history": underlying_history,
        },
    )
    arbitrage_report = root / "arbitrage-report.md"
    build_arbitrage_report(all_diagnostics, all_adjustments, arbitrage_report)
    report = root / "surface-report.md"
    selected_features = all_features.filter(pl.col("quote_date") == selected_date)
    _write_report(
        report,
        selected_date,
        selected["forwards"],
        selected["comparison"],
        selected_features,
        plots,
        historical_plots,
    )

    successful_fits = sum(fit.success for fit in selected["fits"])
    failed_fits = sum(not fit.success for fit in selected["fits"])
    manifest = {
        "quote_date": str(selected_date),
        "input_rows": all_quotes.filter(pl.col("quote_date") == selected_date).height,
        "iv_rows": selected["iv_data"].height,
        "successful_fits": successful_fits,
        "failed_fits": failed_fits,
        "historical_dates": len(analyses),
        "feature_rows": all_features.height,
        "valid_feature_rows": all_features.filter(pl.col("chain_valid")).height,
        "outputs": {
            "implied_volatility_dataset": str(iv_path),
            "forward_summary": str(forward_path),
            "forward_pairs": str(pair_path),
            "model_fit_details": str(details_path),
            "model_comparison": str(comparison_path),
            "arbitrage_diagnostics": str(diagnostics_path),
            "surface_adjustments": str(adjustments_path),
            "standardized_delta_points": str(points_path),
            "daily_feature_table": str(features_path),
            "database": str(database),
            "report": str(report),
            "arbitrage_report": str(arbitrage_report),
            "plots": [str(path) for path in plots],
            "historical_plots": [str(path) for path in historical_plots],
        },
    }
    (root / "surface-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return SurfaceAnalysisResult(
        quote_date=selected_date,
        input_rows=manifest["input_rows"],
        iv_rows=selected["iv_data"].height,
        successful_fits=successful_fits,
        failed_fits=failed_fits,
        output_dir=root,
        implied_volatility_dataset=iv_path,
        forward_summary=forward_path,
        model_comparison=comparison_path,
        report=report,
        arbitrage_report=arbitrage_report,
        arbitrage_diagnostics=diagnostics_path,
        surface_adjustments=adjustments_path,
        standardized_delta_points=points_path,
        daily_feature_table=features_path,
        database=database,
        plots=tuple(plots),
        historical_plots=tuple(historical_plots),
    )
