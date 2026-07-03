# Creates rejection summary and Markdown data-quality report

from __future__ import annotations

from pathlib import Path

import polars as pl


def build_rejection_summary(rejected: pl.DataFrame) -> pl.DataFrame:
    if rejected.is_empty():
        return pl.DataFrame({"rejection_reason": [], "row_count": []})
    reasons: list[str] = []
    for value in rejected["rejection_reason"].drop_nulls().to_list():
        reasons.extend(value.split(";"))
    return (
        pl.DataFrame({"rejection_reason": reasons})
        .group_by("rejection_reason")
        .len(name="row_count")
        .sort("row_count", descending=True)
    )


def write_quality_report(
    path: Path,
    source_file: str,
    total_rows: int,
    clean: pl.DataFrame,
    rejected: pl.DataFrame,
    summary: pl.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    delays = (
        clean["alignment_delay_seconds"].drop_nulls()
        if "alignment_delay_seconds" in clean
        else None
    )
    median_delay = delays.median() if delays is not None and not delays.is_empty() else None
    rejection_lines = ["| Reason | Rows |", "|---|---:|"]
    rejection_lines.extend(
        f"| {row['rejection_reason']} | {row['row_count']} |"
        for row in summary.iter_rows(named=True)
    )
    if summary.is_empty():
        rejection_lines.append("| None | 0 |")

    contents = f"""# Data-quality report

- Source file: `{source_file}`
- Input rows: {total_rows}
- Clean rows: {clean.height}
- Rejected rows: {rejected.height}
- Acceptance rate: {(clean.height / total_rows if total_rows else 0):.1%}
- Median underlying alignment delay: {median_delay if median_delay is not None else "n/a"} seconds

## Rejection summary

{chr(10).join(rejection_lines)}
"""
    path.write_text(contents, encoding="utf-8")