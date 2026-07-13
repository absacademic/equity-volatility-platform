# IV smile and surface pipeline

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from vol_platform.config import load_config
from vol_platform.data.adapters import RateCSVAdapter
from vol_platform.surface.evaluation import evaluate_fits
from vol_platform.surface.expiration import exact_time_to_expiry
from vol_platform.surface.features import build_implied_volatility_dataset
from vol_platform.surface.forward import ForwardSettings, estimate_forwards
from vol_platform.surface.models import fit_all_smiles
from vol_platform.surface.plots import create_surface_plots
from vol_platform.surface.rates import discount_factor, interpolate_rate


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
    database: Path
    plots: tuple[Path, ...]


def _read_quotes(path: Path) -> pl.DataFrame:
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no Parquet files found under {path}")
        return pl.concat([pl.read_parquet(file) for file in files], how="diagonal_relaxed")
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    raise ValueError("surface analysis expects a clean Parquet file or directory")


def _read_rates(path: str | Path | None) -> pl.DataFrame | None:
    if path is None:
        return None
    selected = Path(path)
    if selected.suffix.lower() == ".parquet":
        return pl.read_parquet(selected)
    return RateCSVAdapter().read(selected, "surface_input")


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


def _write_database(
    path: Path,
    iv_data: pl.DataFrame,
    forwards: pl.DataFrame,
    pairs: pl.DataFrame,
    details: pl.DataFrame,
    comparison: pl.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    try:
        for name, frame in {
            "implied_volatility_dataset": iv_data,
            "forward_estimates": forwards,
            "forward_pairs": pairs,
            "smile_fit_details": details,
            "model_comparison": comparison,
        }.items():
            print(f"{name}: rows={frame.height}, columns={frame.columns}")

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
    plot_paths: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reliable = forwards.filter(pl.col("reliability") != "unreliable").height
    lines = [
        f"# Surface analysis: {quote_date}",
        "",
        "## Forward estimation",
        "",
        f"- Expirations estimated: {forwards.height}",
        f"- Expirations with usable reliability: {reliable}",
        "- Forward dispersion is the weighted standard deviation across selected near-ATM pairs.",
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
        for row in comparison.iter_rows(named=True):
            lines.append(
                "| {model} | {weighting} | {failed_fit_rate:.3f} | {average_rmse:.6f} | "
                "{maximum_residual:.6f} | {average_coverage:.3f} | "
                "{average_stability:.3f} |".format(**row)
            )
    lines.extend(["", "## Visualizations", ""])
    lines.extend(f"- `{plot.name}`" for plot in plot_paths)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_surface_analysis(
    clean_quotes: str | Path,
    *,
    rates_file: str | Path | None = None,
    quote_date: date | str | None = None,
    config_path: str | Path = "configs/base.yml",
    output_dir: str | Path | None = None,
) -> SurfaceAnalysisResult:
    """Estimate forwards, calculate IVs, fit smiles, and write diagnostics for one date."""

    config = load_config(config_path)
    settings = _surface_settings(config)
    quotes = _read_quotes(Path(clean_quotes)).filter(pl.col("is_valid"))
    available_dates = quotes["quote_date"].unique().sort()
    if available_dates.is_empty():
        raise ValueError("no valid quote dates were found")
    selected_date = date.fromisoformat(quote_date) if isinstance(quote_date, str) else quote_date
    selected_date = selected_date or available_dates[-1]
    quotes = quotes.filter(pl.col("quote_date") == selected_date)
    if quotes.is_empty():
        raise ValueError(f"no clean quotes found for {selected_date}")

    rates = _read_rates(rates_file)
    project = config.project
    enriched = _add_rate_and_expiration_features(
        quotes,
        rates,
        timezone=str(project.get("timezone", "America/New_York")),
        day_count_basis=float(project.get("day_count_basis", 365.0)),
        default_rate=config.pricing.default_rate,
    )
    forward_config = settings.get("forward", {})
    forward_settings = ForwardSettings(
        max_pairs=int(forward_config.get("max_pairs", 5)),
        maximum_atm_distance=float(forward_config.get("maximum_atm_distance", 0.10)),
        reliable_relative_dispersion=float(
            forward_config.get("reliable_relative_dispersion", 0.005)
        ),
        minimum_average_quality=float(forward_config.get("minimum_average_quality", 0.50)),
    )
    forwards, pairs = estimate_forwards(enriched, forward_settings)
    solver_options = {
        "initial_volatility": config.pricing.iv.initial_volatility,
        "minimum_volatility": config.pricing.iv.minimum_volatility,
        "maximum_volatility": config.pricing.iv.maximum_volatility,
        "price_tolerance": config.pricing.iv.price_tolerance,
        "volatility_tolerance": config.pricing.iv.volatility_tolerance,
        "max_iterations": config.pricing.iv.max_iterations,
    }
    iv_data = build_implied_volatility_dataset(enriched, forwards, solver_options=solver_options)
    fits = fit_all_smiles(
        iv_data,
        smoothing=float(settings.get("spline_smoothing", 1e-7)),
    )
    details, comparison = evaluate_fits(iv_data, fits)

    default_root = Path(config.paths["processed_data"]) / "surfaces" / str(selected_date)
    root = Path(output_dir or default_root)
    root.mkdir(parents=True, exist_ok=True)
    iv_path = root / "implied-volatility.parquet"
    forward_path = root / "forward-estimates.csv"
    pair_path = root / "forward-pairs.csv"
    details_path = root / "smile-fit-details.csv"
    comparison_path = root / "model-comparison.csv"
    iv_data.write_parquet(iv_path)
    forwards.write_csv(forward_path)
    pairs.write_csv(pair_path)
    details.write_csv(details_path)
    comparison.write_csv(comparison_path)

    plots = create_surface_plots(iv_data, details, fits, root / "plots")
    database = root / "surface.duckdb"
    _write_database(database, iv_data, forwards, pairs, details, comparison)
    report = root / "surface-report.md"
    _write_report(report, selected_date, forwards, comparison, plots)
    manifest = {
        "quote_date": str(selected_date),
        "input_rows": quotes.height,
        "iv_rows": iv_data.height,
        "successful_fits": sum(fit.success for fit in fits),
        "failed_fits": sum(not fit.success for fit in fits),
        "outputs": {
            "implied_volatility_dataset": str(iv_path),
            "forward_summary": str(forward_path),
            "forward_pairs": str(pair_path),
            "model_fit_details": str(details_path),
            "model_comparison": str(comparison_path),
            "database": str(database),
            "report": str(report),
            "plots": [str(path) for path in plots],
        },
    }
    (root / "surface-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return SurfaceAnalysisResult(
        quote_date=selected_date,
        input_rows=quotes.height,
        iv_rows=iv_data.height,
        successful_fits=manifest["successful_fits"],
        failed_fits=manifest["failed_fits"],
        output_dir=root,
        implied_volatility_dataset=iv_path,
        forward_summary=forward_path,
        model_comparison=comparison_path,
        report=report,
        database=database,
        plots=tuple(plots),
    )
