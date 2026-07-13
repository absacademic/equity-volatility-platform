# Smile-fit diagnostics and model-comparison tables

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from vol_platform.surface.models import SmileFit

SMILE_FIT_DETAIL_SCHEMA = {
    "expiration": pl.Date,
    "model": pl.String,
    "weighting": pl.String,
    "fit_success": pl.Boolean,
    "message": pl.String,
    "observation_count": pl.Int64,
    "rmse": pl.Float64,
    "maximum_residual": pl.Float64,
    "coverage": pl.Float64,
    "stability": pl.Float64,
}


MODEL_COMPARISON_SCHEMA = {
    "model": pl.String,
    "weighting": pl.String,
    "fit_count": pl.Int64,
    "successful_fit_count": pl.Int64,
    "failed_fit_count": pl.Int64,
    "failed_fit_rate": pl.Float64,
    "average_rmse": pl.Float64,
    "maximum_residual": pl.Float64,
    "average_coverage": pl.Float64,
    "average_stability": pl.Float64,
}


def _fit_metrics(fit: SmileFit, frame: pl.DataFrame) -> dict[str, Any]:
    base = {
        "expiration": fit.expiration,
        "model": fit.model,
        "weighting": fit.weighting,
        "fit_success": fit.success,
        "message": fit.message,
        "observation_count": frame.height,
    }

    if not fit.success or frame.is_empty():
        return {
            **base,
            "rmse": None,
            "maximum_residual": None,
            "coverage": 0.0,
            "stability": 0.0,
        }

    x = np.asarray(frame["forward_moneyness"], dtype=float)
    observed_iv = np.asarray(frame["mid_implied_volatility"], dtype=float)
    time_to_expiry = float(frame["time_to_expiry"].median())

    finite_observations = np.isfinite(x) & np.isfinite(observed_iv)

    if time_to_expiry <= 0.0 or not finite_observations.any():
        return {
            **base,
            "rmse": None,
            "maximum_residual": None,
            "coverage": 0.0,
            "stability": 0.0,
        }

    predicted_variance = fit.predict_total_variance(x)

    valid = finite_observations & np.isfinite(predicted_variance) & (predicted_variance > 0.0)

    predicted_iv = np.full_like(predicted_variance, np.nan)
    predicted_iv[valid] = np.sqrt(predicted_variance[valid] / time_to_expiry)

    residuals = predicted_iv[valid] - observed_iv[valid]
    coverage = float(valid.mean())

    if residuals.size > 0:
        rmse = float(np.sqrt(np.mean(residuals**2)))
        maximum_residual = float(np.max(np.abs(residuals)))
    else:
        rmse = None
        maximum_residual = None

    finite_x = x[finite_observations]

    if finite_x.size < 2 or np.ptp(finite_x) <= 0.0:
        stability = 0.0
    else:
        grid = np.linspace(
            float(finite_x.min()),
            float(finite_x.max()),
            101,
        )
        grid_variance = fit.predict_total_variance(grid)
        stable = np.isfinite(grid_variance) & (grid_variance > 0.0)

        if stable.all():
            first_derivative = np.gradient(grid_variance, grid)
            curvature = np.gradient(first_derivative, grid)
            scale = max(float(np.mean(grid_variance)), 1e-8)
            roughness = float(np.mean(np.abs(curvature)) / scale)
            stability = 1.0 / (1.0 + roughness)
        else:
            stability = float(stable.mean())

    return {
        **base,
        "rmse": rmse,
        "maximum_residual": maximum_residual,
        "coverage": coverage,
        "stability": stability,
    }


def evaluate_fits(
    iv_data: pl.DataFrame,
    fits: list[SmileFit],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return per-expiration diagnostics and aggregate comparisons."""

    eligible = iv_data.filter(pl.col("fit_eligible"))

    rows: list[dict[str, Any]] = []

    for fit in fits:
        frame = eligible.filter(pl.col("expiration") == fit.expiration)
        rows.append(_fit_metrics(fit, frame))

    if rows:
        details = pl.DataFrame(
            rows,
            schema=SMILE_FIT_DETAIL_SCHEMA,
            strict=False,
        )
    else:
        details = pl.DataFrame(
            schema=SMILE_FIT_DETAIL_SCHEMA,
        )

    if details.is_empty():
        comparison = pl.DataFrame(
            schema=MODEL_COMPARISON_SCHEMA,
        )
        return details, comparison

    comparison = (
        details.group_by(["model", "weighting"])
        .agg(
            pl.len().cast(pl.Int64).alias("fit_count"),
            pl.col("fit_success").cast(pl.Int64).sum().alias("successful_fit_count"),
            (1 - pl.col("fit_success").cast(pl.Int64)).sum().alias("failed_fit_count"),
            (1.0 - pl.col("fit_success").cast(pl.Float64).mean()).alias("failed_fit_rate"),
            pl.col("rmse").mean().alias("average_rmse"),
            pl.col("maximum_residual").max().alias("maximum_residual"),
            pl.col("coverage").mean().alias("average_coverage"),
            pl.col("stability").mean().alias("average_stability"),
        )
        .select(list(MODEL_COMPARISON_SCHEMA))
        .cast(MODEL_COMPARISON_SCHEMA, strict=False)
        .sort(
            ["failed_fit_rate", "average_rmse"],
            nulls_last=True,
        )
    )

    return details, comparison


def best_fits_by_expiration_and_model(
    details: pl.DataFrame, fits: list[SmileFit]
) -> dict[tuple[object, str], SmileFit]:
    # Select the successful weighting with the lowest RMSE for each model and expiry

    selected: dict[tuple[object, str], SmileFit] = {}
    if details.is_empty():
        return selected
    successful = details.filter(pl.col("fit_success") & pl.col("rmse").is_not_null())
    for group in successful.partition_by(["expiration", "model"], maintain_order=True):
        best = group.sort("rmse").row(0, named=True)
        for fit in fits:
            if (
                fit.expiration == best["expiration"]
                and fit.model == best["model"]
                and fit.weighting == best["weighting"]
            ):
                selected[(fit.expiration, fit.model)] = fit
                break
    return selected
