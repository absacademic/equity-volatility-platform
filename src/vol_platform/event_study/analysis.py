# Summary analysis, regimes, and evidence-based conclusion text

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def add_regimes(dataset: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for row in dataset.iter_rows(named=True):
        percentile = row.get("iv_percentile")
        dislocation = row.get("surface_dislocation")
        row["iv_regime"] = (
            "high_iv" if percentile is not None and float(percentile) >= 0.50 else "low_iv"
        )
        row["surface_regime"] = (
            "dislocated" if dislocation is not None and float(dislocation) >= 1.0 else "normal"
        )
        rows.append(row)
    return pl.DataFrame(rows) if rows else dataset


def _group_summary(rows: list[dict[str, Any]], label: str, value: str) -> dict[str, Any]:
    gaps = np.asarray([float(row["expected_minus_realized_move"]) for row in rows])
    returns = np.asarray([float(row["absolute_return"]) for row in rows])
    expected = np.asarray([float(row["expected_move"]) for row in rows])
    return {
        "group": label,
        "value": value,
        "observation_count": len(rows),
        "mean_expected_move": float(np.mean(expected)),
        "mean_absolute_return": float(np.mean(returns)),
        "mean_expected_minus_realized": float(np.mean(gaps)),
        "overestimate_rate": float(np.mean(gaps > 0.0)),
        "mean_iv_collapse": float(
            np.mean(
                [
                    float(row["post_event_iv_collapse"])
                    for row in rows
                    if row.get("post_event_iv_collapse") is not None
                ]
            )
        )
        if any(row.get("post_event_iv_collapse") is not None for row in rows)
        else None,
    }


def build_summary_tables(dataset: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = list(dataset.iter_rows(named=True))
    summary = [_group_summary(rows, "all", "all")] if rows else []
    regime = []
    for column in ("event_type", "iv_regime", "surface_regime", "period"):
        values = sorted({str(row.get(column)) for row in rows})
        for value in values:
            selected = [row for row in rows if str(row.get(column)) == value]
            result = _group_summary(selected, column, value)
            (summary if column in {"event_type", "period"} else regime).append(result)
    return pl.DataFrame(summary), pl.DataFrame(regime)


def build_research_conclusion(
    model_performance: pl.DataFrame,
    strategy_summary: pl.DataFrame,
    *,
    synthetic: bool = False,
) -> str:
    test_model = model_performance.filter(
        (pl.col("model") == "linear") & (pl.col("period") == "test")
    )
    test_strategy = strategy_summary.filter(pl.col("period") == "test")
    r_squared = test_model["r_squared"][0] if test_model.height else None
    directional = test_model["directional_accuracy"][0] if test_model.height else None
    net = test_strategy["mean_net_return"][0] if test_strategy.height else None
    r_squared = float(r_squared) if r_squared is not None and np.isfinite(r_squared) else None
    directional = (
        float(directional) if directional is not None and np.isfinite(directional) else None
    )
    net = float(net) if net is not None and np.isfinite(net) else None
    prefix = "Synthetic demonstration conclusion" if synthetic else "Research conclusion"
    if net is None or directional is None:
        finding = "The available sample is too small to support an out-of-sample conclusion."
    elif net <= 0.0:
        finding = (
            "Pre-event surface features do not support a tradable prediction after estimated "
            "transaction costs. Any apparent gross predictability is economically weak."
        )
    elif directional < 0.55 or r_squared is None or r_squared <= 0.0:
        finding = (
            "The models show limited statistical predictability and the positive backtest result "
            "is not stable enough to treat as evidence of an exploitable signal."
        )
    else:
        finding = (
            "The chronological test sample shows modest evidence that pre-event surface features "
            "predict over- versus underestimation, with positive cost-adjusted strategy results."
        )
    qualification = (
        " This conclusion describes generated data and validates the workflow, not "
        "real-market evidence."
        if synthetic
        else " Results remain conditional on data quality, execution assumptions, and "
        "the event sample."
    )
    metrics = (
        f" Test directional accuracy: {directional:.1%}; test R-squared: {r_squared:.3f}; "
        f"mean test net return: {net:.4%}."
        if directional is not None and r_squared is not None and net is not None
        else ""
    )
    return f"# {prefix}\n\n{finding}{qualification}{metrics}\n"
