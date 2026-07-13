# Cubic spline and SVI smile models

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares

from vol_platform.surface.weights import Weighting, build_weights


@dataclass(frozen=True, slots=True)
class SmileFit:
    model: str
    weighting: str
    expiration: date
    success: bool
    message: str
    x_min: float | None
    x_max: float | None
    parameters: dict[str, float]
    predictor: Callable[[np.ndarray], np.ndarray] | None

    def predict_total_variance(self, x: np.ndarray) -> np.ndarray:
        if not self.success or self.predictor is None:
            return np.full_like(np.asarray(x, dtype=float), np.nan, dtype=float)
        return np.asarray(self.predictor(np.asarray(x, dtype=float)), dtype=float)


def _prepared_arrays(
    frame: pl.DataFrame, weighting: Weighting | str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort("forward_moneyness")
    x_raw = np.asarray(ordered["forward_moneyness"], dtype=float)
    y_raw = np.asarray(ordered["total_variance"], dtype=float)
    w_raw = build_weights(ordered, weighting)

    x_values: list[float] = []
    y_values: list[float] = []
    weights: list[float] = []
    for value in np.unique(x_raw):
        mask = x_raw == value
        local_weights = w_raw[mask]
        x_values.append(float(value))
        y_values.append(float(np.average(y_raw[mask], weights=local_weights)))
        weights.append(float(local_weights.sum()))
    normalized = np.asarray(weights, dtype=float)
    normalized /= np.mean(normalized)
    return np.asarray(x_values), np.asarray(y_values), normalized


def fit_cubic_spline(
    frame: pl.DataFrame,
    weighting: Weighting | str,
    *,
    smoothing: float = 1e-7,
) -> SmileFit:
    """Fit a weighted cubic smoothing spline to total variance."""

    expiration = frame["expiration"][0]
    weighting = Weighting(weighting)
    x, y, weights = _prepared_arrays(frame, weighting)
    if x.size < 4:
        return SmileFit(
            "cubic_spline",
            str(weighting),
            expiration,
            False,
            "at least four distinct strikes are required",
            None,
            None,
            {},
            None,
        )
    try:
        spline = UnivariateSpline(x, y, w=np.sqrt(weights), k=3, s=smoothing * x.size)
        return SmileFit(
            "cubic_spline",
            str(weighting),
            expiration,
            True,
            "success",
            float(x.min()),
            float(x.max()),
            {"smoothing": smoothing},
            spline,
        )
    except (ValueError, RuntimeError) as exc:
        return SmileFit(
            "cubic_spline",
            str(weighting),
            expiration,
            False,
            str(exc),
            None,
            None,
            {},
            None,
        )


def svi_total_variance(x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    a, b, rho, m, sigma = parameters
    centered = x - m
    return a + b * (rho * centered + np.sqrt(centered**2 + sigma**2))


def fit_svi(frame: pl.DataFrame, weighting: Weighting | str) -> SmileFit:
    """Fit raw SVI total variance with bounded least squares."""

    expiration = frame["expiration"][0]
    weighting = Weighting(weighting)
    x, y, weights = _prepared_arrays(frame, weighting)
    if x.size < 5:
        return SmileFit(
            "svi",
            str(weighting),
            expiration,
            False,
            "at least five distinct strikes are required",
            None,
            None,
            {},
            None,
        )

    span = max(float(x.max() - x.min()), 0.05)
    initial = np.array(
        [max(float(y.min()) * 0.8, 1e-6), max(float(np.ptp(y)) / span, 1e-3), 0.0, 0.0, 0.10]
    )
    lower = np.array([1e-10, 1e-8, -0.999, float(x.min()) - span, 1e-4])
    upper = np.array([max(float(y.max()) * 5.0, 1.0), 10.0, 0.999, float(x.max()) + span, 2.0])

    def residuals(parameters: np.ndarray) -> np.ndarray:
        return np.sqrt(weights) * (svi_total_variance(x, parameters) - y)

    try:
        result = least_squares(
            residuals,
            initial,
            bounds=(lower, upper),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=10_000,
        )
        parameters = result.x
        success = bool(result.success and np.all(svi_total_variance(x, parameters) > 0.0))
        names = ("a", "b", "rho", "m", "sigma")
        return SmileFit(
            "svi",
            str(weighting),
            expiration,
            success,
            result.message,
            float(x.min()),
            float(x.max()),
            {name: float(value) for name, value in zip(names, parameters, strict=True)},
            lambda values: svi_total_variance(values, parameters),
        )
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return SmileFit(
            "svi",
            str(weighting),
            expiration,
            False,
            str(exc),
            None,
            None,
            {},
            None,
        )


def fit_all_smiles(
    iv_data: pl.DataFrame,
    *,
    smoothing: float = 1e-7,
) -> list[SmileFit]:
    fits: list[SmileFit] = []
    eligible = iv_data.filter(pl.col("fit_eligible"))
    for frame in eligible.partition_by("expiration", maintain_order=True):
        for weighting in Weighting:
            fits.append(fit_cubic_spline(frame, weighting, smoothing=smoothing))
            fits.append(fit_svi(frame, weighting))
    return fits
