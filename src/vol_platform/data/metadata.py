"""Records source, file modification time as the available download-time proxy,
ingestion time, SHA-256 hash, row count, date range, and schema version"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


def file_metadata(
    path: str | Path,
    frame: pl.DataFrame,
    source: str,
    schema_version: str,
    date_column: str,
) -> dict[str, Any]:
    file_path = Path(path)
    dates = (
        frame.get_column(date_column).drop_nulls() if date_column in frame.columns else pl.Series()
    )
    minimum = dates.min() if not dates.is_empty() else None
    maximum = dates.max() if not dates.is_empty() else None
    return {
        "source": source,
        "file": str(file_path),
        "download_time_utc": datetime.fromtimestamp(file_path.stat().st_mtime, UTC).isoformat(),
        "ingested_at_utc": datetime.now(UTC).isoformat(),
        "sha256": _sha256(file_path),
        "row_count": frame.height,
        "date_range": {
            "start": minimum.isoformat() if minimum is not None else None,
            "end": maximum.isoformat() if maximum is not None else None,
        },
        "schema_version": schema_version,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
