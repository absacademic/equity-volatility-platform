# Combines the full process into one command

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from vol_platform.config import load_config
from vol_platform.data.adapters import (
    EventCSVAdapter,
    OptionQuoteCSVAdapter,
    RateCSVAdapter,
    UnderlyingPriceCSVAdapter,
)
from vol_platform.data.alignment import align_rates, align_underlying_prices
from vol_platform.data.cleaning import CleaningRules, clean_option_quotes
from vol_platform.data.metadata import file_metadata
from vol_platform.data.report import build_rejection_summary, write_quality_report
from vol_platform.data.storage import (
    create_duckdb_views,
    write_manifest,
    write_partitioned,
    write_table,
)


@dataclass(frozen=True)
class IngestionResult:
    run_id: str
    input_rows: int
    clean_rows: int
    rejected_rows: int
    output_dir: Path
    database: Path
    report: Path
    manifest: Path


def run_ingestion(
    input_file: str | Path,
    *,
    underlying_file: str | Path | None = None,
    rates_file: str | Path | None = None,
    events_file: str | Path | None = None,
    config_path: str | Path = "configs/base.yml",
    output_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    source: str = "local_csv",
) -> IngestionResult:
    config = load_config(config_path)
    data_config: dict[str, Any] = config.model_extra.get("data", {}) if config.model_extra else {}
    schema_version = str(data_config.get("schema_version", "1.0.0"))
    cleaning = data_config.get("cleaning", {})
    alignment = data_config.get("alignment", {})
    rules = CleaningRules(**cleaning)
    max_staleness = float(alignment.get("max_underlying_staleness_seconds", 300))

    input_path = Path(input_file)
    output = Path(output_dir or config.paths["processed_data"])
    database = Path(database_path or output / "volatility.duckdb")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")

    quotes = OptionQuoteCSVAdapter().read(input_path, source)
    embedded = UnderlyingPriceCSVAdapter().read_embedded(input_path)
    underlying = embedded
    metadata = [file_metadata(input_path, quotes, source, schema_version, "quote_timestamp")]

    if underlying_file is not None:
        external = UnderlyingPriceCSVAdapter().read(underlying_file, source)
        underlying = pl.concat([embedded, external], how="diagonal_relaxed").unique(
            subset=["timestamp", "symbol"], keep="last"
        )
        metadata.append(
            file_metadata(underlying_file, external, source, schema_version, "timestamp")
        )

    rates = RateCSVAdapter().read(rates_file, source) if rates_file is not None else None
    events = EventCSVAdapter().read(events_file, source) if events_file is not None else None
    if rates_file is not None and rates is not None:
        metadata.append(file_metadata(rates_file, rates, source, schema_version, "as_of_date"))
    if events_file is not None and events is not None:
        metadata.append(
            file_metadata(events_file, events, source, schema_version, "event_timestamp")
        )

    aligned = align_underlying_prices(quotes, underlying, max_staleness)
    aligned = align_rates(aligned, rates, config.pricing.default_rate)
    normalized = clean_option_quotes(aligned, rules)
    clean = normalized.filter(pl.col("is_valid"))
    rejected = normalized.filter(~pl.col("is_valid"))
    summary = build_rejection_summary(rejected)

    clean_root = output / "options" / "clean"
    rejected_root = output / "options" / "rejected"
    write_partitioned(clean, clean_root, run_id)
    write_partitioned(rejected, rejected_root, run_id)
    write_table(underlying, output / "reference" / f"underlying-{run_id}.parquet")
    if rates is not None:
        write_table(rates, output / "reference" / f"rates-{run_id}.parquet")
    if events is not None:
        write_table(events, output / "events" / f"events-{run_id}.parquet")

    report_dir = Path(config.paths["reports"])
    report = report_dir / f"data-quality-{run_id}.md"
    summary_path = report_dir / f"rejection-summary-{run_id}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(summary_path)
    write_quality_report(report, str(input_path), quotes.height, clean, rejected, summary)

    manifest = output / "metadata" / f"ingestion-{run_id}.json"
    write_manifest(
        {
            "run_id": run_id,
            "schema_version": schema_version,
            "inputs": metadata,
            "outputs": {
                "clean_rows": clean.height,
                "rejected_rows": rejected.height,
                "clean_path": str(clean_root),
                "rejected_path": str(rejected_root),
                "database": str(database),
                "report": str(report),
                "rejection_summary": str(summary_path),
            },
        },
        manifest,
    )
    create_duckdb_views(database, clean_root, rejected_root)
    return IngestionResult(
        run_id=run_id,
        input_rows=quotes.height,
        clean_rows=clean.height,
        rejected_rows=rejected.height,
        output_dir=output,
        database=database,
        report=report,
        manifest=manifest,
    )
