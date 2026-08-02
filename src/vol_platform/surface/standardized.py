# Standardized delta-point interpolation and daily surface features

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from vol_platform.pricing.greeks import black76_greeks
from vol_platform.surface.models import SmileFit

POINT_NAMES = ("10d_put", "25d_put", "atm", "25d_call", "10d_call")

POINT_SCHEMA = {
    "quote_date": pl.Date,
    "symbol": pl.String,
    "expiration": pl.Date,
    "time_to_expiry": pl.Float64,
    "point": pl.String,
    "option_type": pl.String,
    "target_delta": pl.Float64,
    "actual_delta": pl.Float64,
    "delta_error": pl.Float64,
    "delta_convention": pl.String,
    "strike": pl.Float64,
    "forward_moneyness": pl.Float64,
    "implied_volatility": pl.Float64,
    "total_variance": pl.Float64,
    "model": pl.String,
    "weighting": pl.String,
    "status": pl.String,
}


def _best_fit(
    expiration: object,
    details: pl.DataFrame,
    fits: list[SmileFit],
) -> tuple[SmileFit | None, dict[str, Any] | None]:
    candidates = details.filter(
        (pl.col("expiration") == expiration)
        & pl.col("fit_success")
        & pl.col("rmse").is_not_null()
    ).sort(["rmse", "model", "weighting"])
    if candidates.is_empty():
        return None, None
    detail = candidates.row(0, named=True)
    fit = next(
        (
            candidate
            for candidate in fits
            if candidate.expiration == expiration
            and candidate.model == detail["model"]
            and candidate.weighting == detail["weighting"]
            and candidate.success
        ),
        None,
    )
    return fit, detail


def _point_row(
    frame: pl.DataFrame,
    fit: SmileFit,
    *,
    name: str,
    option_type: str,
    target_delta: float | None,
    actual_delta: float | None,
    x: float,
    status: str,
) -> dict[str, Any]:
    time_to_expiry = float(frame["time_to_expiry"].median())
    forward = float(frame["forward"].median())
    variance = float(fit.predict_total_variance(np.asarray([x]))[0])
    implied_volatility = (
        float(np.sqrt(variance / time_to_expiry))
        if variance > 0.0 and time_to_expiry > 0.0
        else None
    )
    return {
        "quote_date": frame["quote_date"][0],
        "symbol": str(frame["underlying_symbol"][0]),
        "expiration": frame["expiration"][0],
        "time_to_expiry": time_to_expiry,
        "point": name,
        "option_type": option_type,
        "target_delta": target_delta,
        "actual_delta": actual_delta,
        "delta_error": (
            actual_delta - target_delta
            if actual_delta is not None and target_delta is not None
            else None
        ),
        "delta_convention": "black76_discounted_forward",
        "strike": forward * float(np.exp(x)) if implied_volatility is not None else None,
        "forward_moneyness": x,
        "implied_volatility": implied_volatility,
        "total_variance": variance if implied_volatility is not None else None,
        "model": fit.model,
        "weighting": fit.weighting,
        "status": status if implied_volatility is not None else "invalid_prediction",
    }


def interpolate_standardized_delta_points(
    iv_data: pl.DataFrame,
    details: pl.DataFrame,
    fits: list[SmileFit],
    *,
    extrapolation_limit: float = 0.15,
) -> pl.DataFrame:
    # Interpolate 10P, 25P, ATM, 25C, and 10C points from the best valid fit

    rows: list[dict[str, Any]] = []
    for frame in iv_data.partition_by(
        ["quote_date", "underlying_symbol", "expiration"], maintain_order=True
    ):
        expiration = frame["expiration"][0]
        fit, _ = _best_fit(expiration, details, fits)
        if fit is None or fit.x_min is None or fit.x_max is None:
            continue
        time_to_expiry = float(frame["time_to_expiry"].median())
        forward = float(frame["forward"].median())
        rate = float(frame["interpolated_rate"].median())
        lower = float(fit.x_min) - extrapolation_limit
        upper = float(fit.x_max) + extrapolation_limit
        grid = np.linspace(lower, upper, 4001)
        variance = fit.predict_total_variance(grid)
        valid = np.isfinite(variance) & (variance > 0.0)
        volatility = np.full_like(variance, np.nan)
        volatility[valid] = np.sqrt(variance[valid] / time_to_expiry)
        strikes = forward * np.exp(grid)

        rows.append(
            _point_row(
                frame,
                fit,
                name="atm",
                option_type="none",
                target_delta=None,
                actual_delta=None,
                x=0.0,
                status=(
                    "interpolated"
                    if float(fit.x_min) <= 0.0 <= float(fit.x_max)
                    else "controlled_extrapolation"
                ),
            )
        )

        for name, option_type, target, side in (
            ("10d_put", "put", 0.10, "left"),
            ("25d_put", "put", 0.25, "left"),
            ("25d_call", "call", 0.25, "right"),
            ("10d_call", "call", 0.10, "right"),
        ):
            side_mask = grid <= 0.0 if side == "left" else grid >= 0.0
            candidate_indices = np.flatnonzero(valid & side_mask)
            if candidate_indices.size == 0:
                continue
            deltas = np.asarray(
                [
                    abs(
                        black76_greeks(
                            forward,
                            float(strikes[index]),
                            time_to_expiry,
                            rate,
                            float(volatility[index]),
                            option_type,
                        ).delta
                    )
                    for index in candidate_indices
                ]
            )
            best_position = int(np.argmin(np.abs(deltas - target)))
            best_index = int(candidate_indices[best_position])
            x = float(grid[best_index])
            status = (
                "interpolated"
                if float(fit.x_min) <= x <= float(fit.x_max)
                else "controlled_extrapolation"
            )
            rows.append(
                _point_row(
                    frame,
                    fit,
                    name=name,
                    option_type=option_type,
                    target_delta=target,
                    actual_delta=float(deltas[best_position]),
                    x=x,
                    status=status,
                )
            )
    return (
        pl.DataFrame(rows, schema=POINT_SCHEMA, strict=False)
        if rows
        else pl.DataFrame(schema=POINT_SCHEMA)
    )


def _median_or_default(
    frame: pl.DataFrame,
    column: str,
    default: float | None = 0.0,
) -> float | None:
    if column not in frame.columns:
        return default
    values = frame[column].drop_nulls()
    return float(values.median()) if len(values) else default


def _count_true(frame: pl.DataFrame, column: str) -> int:
    return int(frame[column].sum()) if column in frame.columns else 0


def _sum_or_default(frame: pl.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].fill_null(0).sum())


def build_daily_volatility_features(
    points: pl.DataFrame,
    iv_data: pl.DataFrame,
    details: pl.DataFrame,
    diagnostics: pl.DataFrame,
) -> pl.DataFrame:
    # Build one quality-controlled feature row per symbol, date, and expiration

    rows: list[dict[str, Any]] = []
    for frame in iv_data.partition_by(
        ["quote_date", "underlying_symbol", "expiration"], maintain_order=True
    ):
        quote_date = frame["quote_date"][0]
        symbol = str(frame["underlying_symbol"][0])
        expiration = frame["expiration"][0]
        local_points = points.filter(
            (pl.col("quote_date") == quote_date)
            & (pl.col("symbol") == symbol)
            & (pl.col("expiration") == expiration)
        )
        point_map = {row["point"]: row for row in local_points.iter_rows(named=True)}
        values = {
            name: point_map.get(name, {}).get("implied_volatility") for name in POINT_NAMES
        }
        strikes = {name: point_map.get(name, {}).get("strike") for name in POINT_NAMES}
        statuses = {name: point_map.get(name, {}).get("status") for name in POINT_NAMES}

        best_details = details.filter(
            (pl.col("expiration") == expiration)
            & pl.col("fit_success")
            & pl.col("rmse").is_not_null()
        ).sort("rmse")
        best = best_details.row(0, named=True) if not best_details.is_empty() else {}
        local_diagnostics = diagnostics.filter(
            (pl.col("quote_date") == quote_date)
            & (pl.col("symbol") == symbol)
            & (pl.col("expiration") == expiration)
        )
        violations = local_diagnostics.filter(pl.col("is_violation"))
        unresolved_material = violations.filter(
            (pl.col("severity") == "material") & ~pl.col("resolved")
        )

        atm = values["atm"]
        put_25 = values["25d_put"]
        call_25 = values["25d_call"]
        put_10 = values["10d_put"]
        call_10 = values["10d_call"]
        complete = all(value is not None for value in values.values())
        row = {
            "quote_date": quote_date,
            "quote_timestamp": frame["quote_timestamp"][0],
            "symbol": symbol,
            "expiration": expiration,
            "time_to_expiry": float(frame["time_to_expiry"].median()),
            "forward": float(frame["forward"].median()),
            "dividend_present_value": _median_or_default(
                frame, "dividend_present_value"
            ),
            "dividend_yield_estimate": _median_or_default(
                frame, "dividend_yield_estimate"
            ),
            "early_exercise_risk_count": _count_true(frame, "early_exercise_risk"),
            "total_option_volume": _sum_or_default(frame, "volume"),
            "total_open_interest": _sum_or_default(frame, "open_interest"),
            "iv_10d_put": put_10,
            "iv_25d_put": put_25,
            "atm_implied_volatility": atm,
            "iv_25d_call": call_25,
            "iv_10d_call": call_10,
            "strike_10d_put": strikes["10d_put"],
            "strike_25d_put": strikes["25d_put"],
            "strike_atm": strikes["atm"],
            "strike_25d_call": strikes["25d_call"],
            "strike_10d_call": strikes["10d_call"],
            "downside_skew_25": put_25 - atm if put_25 is not None and atm is not None else None,
            "risk_reversal_25": (
                call_25 - put_25
                if call_25 is not None and put_25 is not None
                else None
            ),
            "butterfly_25": (
                0.5 * (put_25 + call_25) - atm
                if put_25 is not None and call_25 is not None and atm is not None
                else None
            ),
            "wing_curvature_10_25": (
                0.5 * ((put_10 - put_25) + (call_10 - call_25))
                if all(value is not None for value in (put_10, put_25, call_10, call_25))
                else None
            ),
            "iv_bid_ask_width": _median_or_default(
                frame.with_columns(
                    (
                        pl.col("ask_implied_volatility")
                        - pl.col("bid_implied_volatility")
                    ).alias("_iv_width")
                ),
                "_iv_width",
                default=None,
            ),
            "surface_residual_rmse": best.get("rmse"),
            "surface_maximum_residual": best.get("maximum_residual"),
            "surface_model": best.get("model"),
            "surface_weighting": best.get("weighting"),
            "arbitrage_violation_count": violations.height,
            "material_arbitrage_violation_count": unresolved_material.height,
            "resolved_arbitrage_violation_count": violations.filter(
                pl.col("resolved")
            ).height,
            "midpoint_violation_count": violations.filter(pl.col("source") == "midpoint").height,
            "executable_violation_count": violations.filter(
                pl.col("source") == "executable_bid_ask"
            ).height,
            "fitted_surface_violation_count": violations.filter(
                pl.col("source") == "fitted_surface"
            ).height,
            "standardized_points_complete": complete,
            "used_controlled_extrapolation": any(
                status == "controlled_extrapolation" for status in statuses.values()
            ),
            "chain_valid": bool(complete and unresolved_material.is_empty()),
        }
        rows.append(row)

    features = pl.DataFrame(rows) if rows else pl.DataFrame()
    if features.is_empty():
        return features

    # Cross-expiration term slopes use only the current date's available surface.
    output: list[dict[str, Any]] = []
    for frame in features.partition_by(["quote_date", "symbol"], maintain_order=True):
        ordered = frame.sort("time_to_expiry")
        previous: dict[str, Any] | None = None
        for row in ordered.iter_rows(named=True):
            if previous is None:
                row["atm_term_structure_slope"] = None
                row["skew_term_structure_slope"] = None
            else:
                delta_time = float(row["time_to_expiry"] - previous["time_to_expiry"])
                row["atm_term_structure_slope"] = (
                    (row["atm_implied_volatility"] - previous["atm_implied_volatility"])
                    / delta_time
                    if delta_time > 0.0
                    and row["atm_implied_volatility"] is not None
                    and previous["atm_implied_volatility"] is not None
                    else None
                )
                row["skew_term_structure_slope"] = (
                    (row["downside_skew_25"] - previous["downside_skew_25"]) / delta_time
                    if delta_time > 0.0
                    and row["downside_skew_25"] is not None
                    and previous["downside_skew_25"] is not None
                    else None
                )
            output.append(row)
            previous = row
    return pl.DataFrame(output).sort(["quote_date", "symbol", "expiration"])
