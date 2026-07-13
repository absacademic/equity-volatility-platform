# Writes Parquet paritions and creates DuckDB views

import json
from pathlib import Path
from typing import Any

import duckdb
import polars as pl


def write_partitioned(frame: pl.DataFrame, root: Path, run_id: str) -> None:
    # Write symbol/date partitions (Hive-style) with one file per partition

    root.mkdir(parents=True, exist_ok=True)
    if frame.is_empty():
        frame.write_parquet(root / f"empty-{run_id}.parquet")
        return
    for partition in frame.partition_by(["underlying_symbol", "quote_date"], maintain_order=True):
        symbol = partition["underlying_symbol"][0] or "UNKNOWN"
        quote_date = partition["quote_date"][0] or "UNKNOWN"
        target = root / f"underlying_symbol={symbol}" / f"quote_date={quote_date}"
        target.mkdir(parents=True, exist_ok=True)
        partition.write_parquet(target / f"part-{run_id}.parquet")


def write_table(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def create_duckdb_views(database: Path, clean_root: Path, rejected_root: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    clean_glob = (clean_root / "**" / "*.parquet").as_posix().replace("'", "''")
    rejected_glob = (rejected_root / "**" / "*.parquet").as_posix().replace("'", "''")
    connection = duckdb.connect(str(database))
    try:
        connection.execute(
            f"""
            CREATE OR REPLACE VIEW clean_quotes AS
            SELECT *
            FROM read_parquet(
                '{clean_glob}', hive_partitioning = true, union_by_name = true
            );

            CREATE OR REPLACE VIEW rejected_quotes AS
            SELECT *
            FROM read_parquet(
                '{rejected_glob}', hive_partitioning = true, union_by_name = true
            );

            CREATE OR REPLACE VIEW daily_summaries AS
            SELECT
                quote_date,
                underlying_symbol,
                COUNT(*) AS quote_count,
                COUNT(DISTINCT expiration) AS expiration_count,
                AVG(mid) AS average_mid,
                AVG(relative_spread) AS average_relative_spread,
                AVG(quote_quality_score) AS average_quality_score,
                MEDIAN(alignment_delay_seconds) AS median_alignment_delay_seconds
            FROM clean_quotes
            GROUP BY quote_date, underlying_symbol;

            CREATE OR REPLACE VIEW expiration_chains AS
            SELECT
                *,
                CONCAT(
                    underlying_symbol, ':', CAST(expiration AS VARCHAR), ':', option_type
                ) AS chain_id,
                ROW_NUMBER() OVER (
                    PARTITION BY quote_date, underlying_symbol, expiration, option_type
                    ORDER BY strike
                ) AS chain_position
            FROM clean_quotes;
            """
        )
    finally:
        connection.close()
