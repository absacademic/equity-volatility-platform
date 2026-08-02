# Chronological baseline regressions and walk-forward evaluation

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

DEFAULT_FEATURES = (
    "atm_volatility",
    "skew",
    "term_structure",
    "volume_change",
    "open_interest_change",
    "iv_percentile",
    "surface_dislocation",
)


@dataclass(frozen=True, slots=True)
class ModelOutputs:
    dataset: pl.DataFrame
    coefficients: pl.DataFrame
    performance: pl.DataFrame
    stability: pl.DataFrame
    walk_forward_predictions: pl.DataFrame
    walk_forward_performance: pl.DataFrame


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def assign_chronological_periods(
    frame: pl.DataFrame,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> pl.DataFrame:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if validation_fraction < 0.0 or train_fraction + validation_fraction >= 1.0:
        raise ValueError("training and validation fractions must leave a test period")
    ordered = frame.sort("event_timestamp")
    count = ordered.height
    train_end = max(1, min(count - 2, int(count * train_fraction))) if count >= 3 else count
    validation_end = (
        max(train_end + 1, min(count - 1, int(count * (train_fraction + validation_fraction))))
        if count >= 3
        else count
    )
    periods = [
        "train" if index < train_end else "validation" if index < validation_end else "test"
        for index in range(count)
    ]
    return ordered.with_columns(pl.Series("period", periods))


def _matrix(
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
    *,
    medians: np.ndarray | None = None,
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_rows = []
    for row in rows:
        values = []
        for name in features:
            value = _finite(row.get(name))
            values.append(value if value is not None else np.nan)
        raw_rows.append(values)
    raw = np.asarray(raw_rows, dtype=float)
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
    standardized = (filled - means) / scales
    design = np.column_stack([np.ones(len(standardized)), standardized])
    return design, medians, means, scales


def _linear_fit(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    inverse = np.linalg.pinv(design.T @ design + penalty)
    beta = inverse @ design.T @ target
    residual = target - design @ beta
    degrees = max(len(target) - design.shape[1], 1)
    variance = float(residual @ residual / degrees)
    standard_errors = np.sqrt(np.maximum(np.diag(inverse) * variance, 0.0))
    return beta, standard_errors


def _logistic_fit(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
    *,
    iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    if len(np.unique(target)) < 2:
        probability = float(np.clip(target.mean() if len(target) else 0.5, 1.0e-6, 1 - 1.0e-6))
        beta = np.zeros(design.shape[1])
        beta[0] = math.log(probability / (1.0 - probability))
        return beta, np.full(design.shape[1], np.nan)
    beta = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        linear = np.clip(design @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(probability * (1.0 - probability), 1.0e-6, None)
        working = linear + (target - probability) / weights
        hessian = design.T @ (weights[:, None] * design) + penalty
        updated = np.linalg.pinv(hessian) @ design.T @ (weights * working)
        if np.max(np.abs(updated - beta)) < 1.0e-9:
            beta = updated
            break
        beta = updated
    probability = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30.0, 30.0)))
    weights = np.clip(probability * (1.0 - probability), 1.0e-6, None)
    covariance = np.linalg.pinv(design.T @ (weights[:, None] * design) + penalty)
    return beta, np.sqrt(np.maximum(np.diag(covariance), 0.0))


def _auc(target: np.ndarray, probability: np.ndarray) -> float | None:
    positives = int(target.sum())
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probability)
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1, dtype=float)
    numerator = ranks[target == 1].sum() - positives * (positives + 1) / 2
    return float(numerator / (positives * negatives))


def _performance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for period in ("train", "validation", "test"):
        selected = [row for row in rows if row["period"] == period]
        if not selected:
            continue
        actual = np.asarray([float(row["expected_minus_realized_move"]) for row in selected])
        predicted = np.asarray([float(row["linear_prediction"]) for row in selected])
        binary = np.asarray([int(bool(row["market_overestimated"])) for row in selected])
        probability = np.clip(
            np.asarray([float(row["logistic_probability"]) for row in selected]),
            1.0e-8,
            1.0 - 1.0e-8,
        )
        residual = actual - predicted
        denominator = float(np.sum((actual - actual.mean()) ** 2))
        output.extend(
            [
                {
                    "model": "linear",
                    "period": period,
                    "observation_count": len(selected),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "mae": float(np.mean(np.abs(residual))),
                    "r_squared": (
                        1.0 - float(np.sum(residual**2)) / denominator
                        if denominator > 0
                        else None
                    ),
                    "directional_accuracy": float(np.mean((predicted > 0) == (actual > 0))),
                    "accuracy": None,
                    "brier_score": None,
                    "log_loss": None,
                    "auc": None,
                },
                {
                    "model": "logistic",
                    "period": period,
                    "observation_count": len(selected),
                    "rmse": None,
                    "mae": None,
                    "r_squared": None,
                    "directional_accuracy": None,
                    "accuracy": float(np.mean((probability >= 0.5) == binary)),
                    "brier_score": float(np.mean((probability - binary) ** 2)),
                    "log_loss": float(
                        -np.mean(
                            binary * np.log(probability)
                            + (1 - binary) * np.log(1 - probability)
                        )
                    ),
                    "auc": _auc(binary, probability),
                },
            ]
        )
    return output


def _coefficient_rows(
    model: str,
    names: tuple[str, ...],
    beta: np.ndarray,
    standard_errors: np.ndarray,
    fit_sample: str,
) -> list[dict[str, Any]]:
    output = []
    for index, name in enumerate(("intercept", *names)):
        error = float(standard_errors[index]) if np.isfinite(standard_errors[index]) else None
        coefficient = float(beta[index])
        output.append(
            {
                "model": model,
                "fit_sample": fit_sample,
                "feature": name,
                "coefficient": coefficient,
                "standard_error": error,
                "ci_lower_95": coefficient - 1.96 * error if error is not None else None,
                "ci_upper_95": coefficient + 1.96 * error if error is not None else None,
            }
        )
    return output


def _fit_and_predict(
    train: list[dict[str, Any]],
    predict: list[dict[str, Any]],
    features: tuple[str, ...],
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    design, medians, means, scales = _matrix(train, features)
    linear_target = np.asarray([float(row["expected_minus_realized_move"]) for row in train])
    binary_target = np.asarray(
        [int(bool(row["market_overestimated"])) for row in train],
        dtype=float,
    )
    linear_beta, linear_error = _linear_fit(design, linear_target, ridge)
    logistic_beta, logistic_error = _logistic_fit(design, binary_target, ridge)
    predict_design, _, _, _ = _matrix(
        predict,
        features,
        medians=medians,
        means=means,
        scales=scales,
    )
    linear_prediction = predict_design @ linear_beta
    linear_score = np.clip(predict_design @ logistic_beta, -30.0, 30.0)
    logistic_probability = 1.0 / (1.0 + np.exp(-linear_score))
    return (
        linear_prediction,
        logistic_probability,
        linear_beta,
        linear_error,
        logistic_beta,
        logistic_error,
    )


def run_baseline_models(
    dataset: pl.DataFrame,
    *,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    ridge: float = 1.0e-6,
    minimum_walk_forward_train: int = 12,
) -> ModelOutputs:
    if dataset.height < 3:
        raise ValueError("baseline models require at least three chronological events")
    if not features:
        raise ValueError("at least one model feature is required")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    with_period = assign_chronological_periods(
        dataset,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    rows = list(with_period.iter_rows(named=True))
    train = [row for row in rows if row["period"] == "train"]
    (
        predictions,
        probabilities,
        linear_beta,
        linear_error,
        logistic_beta,
        logistic_error,
    ) = _fit_and_predict(train, rows, features, ridge)
    for row, prediction, probability in zip(rows, predictions, probabilities, strict=True):
        row["linear_prediction"] = float(prediction)
        row["logistic_probability"] = float(probability)
        row["predicted_overestimate"] = bool(probability >= 0.5)

    coefficient_rows = _coefficient_rows("linear", features, linear_beta, linear_error, "train")
    coefficient_rows += _coefficient_rows(
        "logistic", features, logistic_beta, logistic_error, "train"
    )

    train_validation = [row for row in rows if row["period"] in {"train", "validation"}]
    _, _, linear_beta_tv, _, logistic_beta_tv, _ = _fit_and_predict(
        train_validation,
        train_validation,
        features,
        ridge,
    )
    stability_rows = []
    for model, first, second in (
        ("linear", linear_beta, linear_beta_tv),
        ("logistic", logistic_beta, logistic_beta_tv),
    ):
        for index, name in enumerate(("intercept", *features)):
            stability_rows.append(
                {
                    "model": model,
                    "feature": name,
                    "train_coefficient": float(first[index]),
                    "train_validation_coefficient": float(second[index]),
                    "absolute_change": float(abs(second[index] - first[index])),
                    "sign_stable": bool(np.sign(first[index]) == np.sign(second[index])),
                }
            )

    walk_rows: list[dict[str, Any]] = []
    start = min(max(minimum_walk_forward_train, 2), max(len(rows) - 1, 2))
    for index in range(start, len(rows)):
        prior = rows[:index]
        current = [rows[index]]
        linear_pred, logistic_pred, *_ = _fit_and_predict(prior, current, features, ridge)
        walk_rows.append(
            {
                "event_id": current[0]["event_id"],
                "event_timestamp": current[0]["event_timestamp"],
                "period": current[0]["period"],
                "training_observation_count": len(prior),
                "actual_move_gap": current[0]["expected_minus_realized_move"],
                "actual_overestimate": current[0]["market_overestimated"],
                "linear_prediction": float(linear_pred[0]),
                "logistic_probability": float(logistic_pred[0]),
            }
        )
    walk_performance = []
    for period in ("validation", "test"):
        selected = [row for row in walk_rows if row["period"] == period]
        if not selected:
            continue
        actual = np.asarray([float(row["actual_move_gap"]) for row in selected])
        predicted = np.asarray([float(row["linear_prediction"]) for row in selected])
        binary = np.asarray([int(bool(row["actual_overestimate"])) for row in selected])
        probability = np.asarray([float(row["logistic_probability"]) for row in selected])
        walk_performance.append(
            {
                "period": period,
                "observation_count": len(selected),
                "linear_rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
                "linear_directional_accuracy": float(np.mean((actual > 0) == (predicted > 0))),
                "logistic_accuracy": float(np.mean((probability >= 0.5) == binary)),
                "logistic_brier_score": float(np.mean((probability - binary) ** 2)),
            }
        )

    return ModelOutputs(
        dataset=pl.DataFrame(rows).sort("event_timestamp"),
        coefficients=pl.DataFrame(coefficient_rows),
        performance=pl.DataFrame(_performance_rows(rows)),
        stability=pl.DataFrame(stability_rows),
        walk_forward_predictions=pl.DataFrame(walk_rows),
        walk_forward_performance=pl.DataFrame(walk_performance),
    )
