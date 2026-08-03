# Checks stability of linear baseline
# Runs degree-two polynomial ridge if baseline passes
# Requires min test directional accuracy, coefficient-sign stability, stable walk-forward test

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class NonlinearOutputs:
    predictions: pl.DataFrame
    performance: pl.DataFrame
    status: pl.DataFrame


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def baseline_stability_status(
    performance: pl.DataFrame,
    stability: pl.DataFrame,
    walk_forward_performance: pl.DataFrame,
    *,
    minimum_directional_accuracy: float = 0.50,
    minimum_sign_stability: float = 0.60,
) -> tuple[bool, str]:
    linear_test = performance.filter((pl.col("model") == "linear") & (pl.col("period") == "test"))
    if linear_test.is_empty():
        return False, "missing chronological test performance"
    accuracy = _finite(linear_test["directional_accuracy"][0])
    if accuracy is None or accuracy < minimum_directional_accuracy:
        return False, "test directional accuracy below stability threshold"
    linear_stability = stability.filter(
        (pl.col("model") == "linear") & (pl.col("feature") != "intercept")
    )
    if linear_stability.is_empty():
        return False, "missing coefficient stability diagnostics"
    sign_share = float(linear_stability["sign_stable"].cast(pl.Float64).mean())
    if sign_share < minimum_sign_stability:
        return False, "too many coefficient sign changes"
    walk_test = walk_forward_performance.filter(pl.col("period") == "test")
    if walk_test.is_empty():
        return False, "missing walk-forward test performance"
    walk_accuracy = _finite(walk_test["linear_directional_accuracy"][0])
    if walk_accuracy is None or walk_accuracy < minimum_directional_accuracy:
        return False, "walk-forward directional accuracy below stability threshold"
    return True, "chronological baseline passed stability gate"


def _base_matrix(
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
    *,
    medians: np.ndarray | None = None,
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(
        [
            [
                value if (value := _finite(row.get(name))) is not None else np.nan
                for name in features
            ]
            for row in rows
        ],
        dtype=float,
    )
    if raw.size == 0:
        raw = np.empty((0, len(features)), dtype=float)
    if medians is None:
        medians = np.nanmedian(raw, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(raw), raw, medians)
    if means is None:
        means = filled.mean(axis=0) if len(filled) else np.zeros(len(features))
    if scales is None:
        scales = filled.std(axis=0, ddof=1) if len(filled) > 1 else np.ones(len(features))
        scales = np.where(np.isfinite(scales) & (scales > 1.0e-12), scales, 1.0)
    return (filled - means) / scales, medians, means, scales


def _polynomial_design(base: np.ndarray) -> np.ndarray:
    columns = [np.ones(len(base))]
    columns.extend(base[:, index] for index in range(base.shape[1]))
    columns.extend(base[:, index] ** 2 for index in range(base.shape[1]))
    columns.extend(
        base[:, first] * base[:, second] for first, second in combinations(range(base.shape[1]), 2)
    )
    return np.column_stack(columns)


def run_gated_polynomial_model(
    dataset: pl.DataFrame,
    baseline_performance: pl.DataFrame,
    coefficient_stability: pl.DataFrame,
    walk_forward_performance: pl.DataFrame,
    *,
    features: tuple[str, ...],
    ridge: float = 0.01,
    enabled: bool = True,
    minimum_directional_accuracy: float = 0.50,
    minimum_sign_stability: float = 0.60,
) -> NonlinearOutputs:
    stable, reason = baseline_stability_status(
        baseline_performance,
        coefficient_stability,
        walk_forward_performance,
        minimum_directional_accuracy=minimum_directional_accuracy,
        minimum_sign_stability=minimum_sign_stability,
    )
    should_run = enabled and stable
    status = pl.DataFrame(
        [
            {
                "model": "polynomial_ridge_degree_2",
                "enabled": enabled,
                "baseline_stable": stable,
                "status": "run" if should_run else "skipped",
                "reason": reason if enabled else "nonlinear model disabled by configuration",
            }
        ]
    )
    if not should_run:
        return NonlinearOutputs(
            predictions=dataset.with_columns(pl.lit(None).alias("nonlinear_prediction")),
            performance=pl.DataFrame(
                [
                    {
                        "model": "polynomial_ridge_degree_2",
                        "period": "not_run",
                        "observation_count": 0,
                        "rmse": None,
                        "mae": None,
                        "r_squared": None,
                        "directional_accuracy": None,
                    }
                ]
            ),
            status=status,
        )

    rows = list(dataset.sort("event_timestamp").iter_rows(named=True))
    train = [row for row in rows if row.get("period") == "train"]
    train_base, medians, means, scales = _base_matrix(train, features)
    design = _polynomial_design(train_base)
    target = np.asarray([float(row["expected_minus_realized_move"]) for row in train])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ target

    all_base, _, _, _ = _base_matrix(
        rows,
        features,
        medians=medians,
        means=means,
        scales=scales,
    )
    predictions = _polynomial_design(all_base) @ beta
    for row, prediction in zip(rows, predictions, strict=True):
        row["nonlinear_prediction"] = float(prediction)

    performance_rows: list[dict[str, Any]] = []
    for period in ("train", "validation", "test"):
        selected = [row for row in rows if row.get("period") == period]
        if not selected:
            continue
        actual = np.asarray([float(row["expected_minus_realized_move"]) for row in selected])
        predicted = np.asarray([float(row["nonlinear_prediction"]) for row in selected])
        residual = actual - predicted
        denominator = float(np.sum((actual - actual.mean()) ** 2))
        performance_rows.append(
            {
                "model": "polynomial_ridge_degree_2",
                "period": period,
                "observation_count": len(selected),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "r_squared": (
                    1.0 - float(np.sum(residual**2)) / denominator if denominator > 0.0 else None
                ),
                "directional_accuracy": float(np.mean((predicted > 0.0) == (actual > 0.0))),
            }
        )
    return NonlinearOutputs(
        predictions=pl.DataFrame(rows).sort("event_timestamp"),
        performance=pl.DataFrame(performance_rows),
        status=status,
    )
