# Static and calendar no-arbitrage diagnostics

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from vol_platform.pricing.black76 import price as black76_price
from vol_platform.surface.evaluation import best_fits_by_expiration_and_model
from vol_platform.surface.models import SmileFit

DIAGNOSTIC_SCHEMA = {
    "quote_date": pl.Date,
    "symbol": pl.String,
    "expiration": pl.Date,
    "source": pl.String,
    "model": pl.String,
    "weighting": pl.String,
    "check": pl.String,
    "option_type": pl.String,
    "location": pl.Float64,
    "value": pl.Float64,
    "tolerance": pl.Float64,
    "is_violation": pl.Boolean,
    "severity": pl.String,
    "resolved": pl.Boolean,
    "message": pl.String,
}

ADJUSTMENT_SCHEMA = {
    "quote_date": pl.Date,
    "symbol": pl.String,
    "expiration": pl.Date,
    "model": pl.String,
    "weighting": pl.String,
    "action": pl.String,
    "check": pl.String,
    "before_value": pl.Float64,
    "after_value": pl.Float64,
    "reason": pl.String,
}


def _diagnostic(
    frame: pl.DataFrame,
    *,
    source: str,
    check: str,
    option_type: str,
    model: str | None = None,
    weighting: str | None = None,
    location: float | None,
    value: float,
    tolerance: float,
    message: str,
) -> dict[str, Any]:
    violation = bool(value < -tolerance)
    return {
        "quote_date": frame["quote_date"][0],
        "symbol": str(frame["underlying_symbol"][0]),
        "expiration": frame["expiration"][0],
        "source": source,
        "model": model,
        "weighting": weighting,
        "check": check,
        "option_type": option_type,
        "location": location,
        "value": value,
        "tolerance": tolerance,
        "is_violation": violation,
        "severity": (
            "material"
            if violation and value < -10.0 * tolerance
            else "warning" if violation else "none"
        ),
        "resolved": False,
        "message": message,
    }


def _market_static_diagnostics(
    iv_data: pl.DataFrame,
    price_tolerance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in iv_data.partition_by(
        ["quote_date", "underlying_symbol", "expiration", "option_type"],
        maintain_order=True,
    ):
        ordered = frame.sort("strike")
        strikes = np.asarray(ordered["strike"], dtype=float)
        mids = np.asarray(ordered["mid"], dtype=float)
        bids = np.asarray(ordered["bid"], dtype=float)
        asks = np.asarray(ordered["ask"], dtype=float)
        option_type = str(ordered["option_type"][0])
        direction = -1.0 if option_type == "call" else 1.0

        for index in range(strikes.size - 1):
            midpoint_margin = direction * (mids[index + 1] - mids[index])
            executable_margin = (
                asks[index] - bids[index + 1]
                if option_type == "call"
                else asks[index + 1] - bids[index]
            )
            rows.append(
                _diagnostic(
                    ordered,
                    source="midpoint",
                    check="strike_monotonicity",
                    option_type=option_type,
                    location=float(0.5 * (strikes[index] + strikes[index + 1])),
                    value=float(midpoint_margin),
                    tolerance=price_tolerance,
                    message="Adjacent midpoint prices must move in the correct strike direction.",
                )
            )
            rows.append(
                _diagnostic(
                    ordered,
                    source="executable_bid_ask",
                    check="strike_monotonicity",
                    option_type=option_type,
                    location=float(0.5 * (strikes[index] + strikes[index + 1])),
                    value=float(executable_margin),
                    tolerance=price_tolerance,
                    message=(
                        "The bid-ask quotes must not permit an executable "
                        "vertical-spread arbitrage."
                    ),
                )
            )

        for index in range(1, strikes.size - 1):
            left, center, right = strikes[index - 1 : index + 2]
            weight_left = (right - center) / (right - left)
            weight_right = 1.0 - weight_left
            midpoint_margin = (
                weight_left * mids[index - 1]
                + weight_right * mids[index + 1]
                - mids[index]
            )
            executable_margin = (
                weight_left * asks[index - 1] + weight_right * asks[index + 1] - bids[index]
            )
            rows.append(
                _diagnostic(
                    ordered,
                    source="midpoint",
                    check="butterfly_convexity",
                    option_type=option_type,
                    location=float(center),
                    value=float(midpoint_margin),
                    tolerance=price_tolerance,
                    message="The midpoint price curve must be convex in strike.",
                )
            )
            rows.append(
                _diagnostic(
                    ordered,
                    source="executable_bid_ask",
                    check="butterfly_convexity",
                    option_type=option_type,
                    location=float(center),
                    value=float(executable_margin),
                    tolerance=price_tolerance,
                    message="The quoted bid-ask butterfly must not have negative executable cost.",
                )
            )
    return rows


def _variance_diagnostics(iv_data: pl.DataFrame, variance_tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in iv_data.partition_by(
        ["quote_date", "underlying_symbol", "expiration"], maintain_order=True
    ):
        for source, column in (
            ("midpoint", "total_variance"),
            ("executable_bid_ask", "bid_total_variance"),
            ("executable_bid_ask", "ask_total_variance"),
        ):
            values = np.asarray(frame[column].fill_null(float("nan")), dtype=float)
            finite = values[np.isfinite(values)]
            minimum = float(finite.min()) if finite.size else float("nan")
            margin = minimum if np.isfinite(minimum) else -float("inf")
            rows.append(
                _diagnostic(
                    frame,
                    source=source,
                    check="negative_total_variance",
                    option_type="all",
                    location=None,
                    value=margin,
                    tolerance=variance_tolerance,
                    message=f"{column} must remain non-negative.",
                )
            )
    return rows


def _market_calendar_diagnostics(iv_data: pl.DataFrame, tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eligible = iv_data.filter(pl.col("fit_eligible")).sort("time_to_expiry")
    for symbol_frame in eligible.partition_by(
        ["quote_date", "underlying_symbol"], maintain_order=True
    ):
        expirations = symbol_frame["expiration"].unique(maintain_order=True).to_list()
        for short_expiration, long_expiration in pairwise(expirations):
            short = symbol_frame.filter(pl.col("expiration") == short_expiration).sort(
                "forward_moneyness"
            )
            long = symbol_frame.filter(pl.col("expiration") == long_expiration).sort(
                "forward_moneyness"
            )
            short_x = np.asarray(short["forward_moneyness"], dtype=float)
            long_x = np.asarray(long["forward_moneyness"], dtype=float)
            lower = max(float(short_x.min()), float(long_x.min()))
            upper = min(float(short_x.max()), float(long_x.max()))
            if upper <= lower:
                continue
            grid = np.linspace(lower, upper, 21)
            short_mid = np.interp(grid, short_x, np.asarray(short["total_variance"], dtype=float))
            long_mid = np.interp(grid, long_x, np.asarray(long["total_variance"], dtype=float))
            short_bid = np.interp(
                grid, short_x, np.asarray(short["bid_total_variance"], dtype=float)
            )
            long_ask = np.interp(
                grid, long_x, np.asarray(long["ask_total_variance"], dtype=float)
            )
            midpoint_margin = float(np.min(long_mid - short_mid))
            executable_margin = float(np.min(long_ask - short_bid))
            location = float(grid[int(np.argmin(long_mid - short_mid))])
            long_frame = long.with_columns(pl.lit(short["quote_date"][0]).alias("quote_date"))
            rows.append(
                _diagnostic(
                    long_frame,
                    source="midpoint",
                    check="calendar_consistency",
                    option_type="otm",
                    location=location,
                    value=midpoint_margin,
                    tolerance=tolerance,
                    message=(
                        f"Total variance at {long_expiration} must not fall below "
                        f"{short_expiration}."
                    ),
                )
            )
            rows.append(
                _diagnostic(
                    long_frame,
                    source="executable_bid_ask",
                    check="calendar_consistency",
                    option_type="otm",
                    location=location,
                    value=executable_margin,
                    tolerance=tolerance,
                    message=(
                        "The longer ask-variance curve must not lie below the "
                        "shorter bid-variance curve."
                    ),
                )
            )
    return rows


def apply_surface_controls(
    iv_data: pl.DataFrame,
    fits: list[SmileFit],
    *,
    variance_tolerance: float = 1e-10,
    price_tolerance: float = 1e-6,
    calendar_tolerance: float = 1e-6,
    extrapolation_padding: float = 0.15,
    maximum_extrapolated_iv: float = 3.0,
    maximum_variance_multiple: float = 8.0,
) -> tuple[list[SmileFit], pl.DataFrame]:
    # Floor tiny negative variance and reject materially invalid fitted surfaces

    controlled: list[SmileFit] = []
    adjustments: list[dict[str, Any]] = []
    by_expiration = {
        frame["expiration"][0]: frame
        for frame in iv_data.filter(pl.col("fit_eligible")).partition_by("expiration")
    }

    for fit in fits:
        frame = by_expiration.get(fit.expiration)
        if not fit.success or frame is None or fit.x_min is None or fit.x_max is None:
            controlled.append(fit)
            continue
        time_to_expiry = float(frame["time_to_expiry"].median())
        forward = float(frame["forward"].median())
        rate = float(frame["interpolated_rate"].median())
        observed_grid = np.linspace(fit.x_min, fit.x_max, 151)
        observed_variance = fit.predict_total_variance(observed_grid)
        minimum_variance = float(np.nanmin(observed_variance))
        candidate = fit

        if minimum_variance < 0.0 and minimum_variance >= -10.0 * variance_tolerance:
            original_predictor = fit.predictor
            candidate = replace(
                fit,
                message=f"{fit.message}; variance floor applied",
                predictor=lambda values, predictor=original_predictor: np.maximum(
                    predictor(np.asarray(values, dtype=float)), variance_tolerance
                ),
            )
            adjustments.append(
                {
                    "quote_date": frame["quote_date"][0],
                    "symbol": str(frame["underlying_symbol"][0]),
                    "expiration": fit.expiration,
                    "model": fit.model,
                    "weighting": fit.weighting,
                    "action": "variance_floor",
                    "check": "negative_total_variance",
                    "before_value": minimum_variance,
                    "after_value": variance_tolerance,
                    "reason": "A small numerical negative was clipped to the configured floor.",
                }
            )
        elif (
            not np.all(np.isfinite(observed_variance))
            or minimum_variance < -10.0 * variance_tolerance
        ):
            candidate = replace(
                fit,
                success=False,
                message=f"{fit.message}; rejected for negative or non-finite total variance",
                predictor=None,
            )
            adjustments.append(
                {
                    "quote_date": frame["quote_date"][0],
                    "symbol": str(frame["underlying_symbol"][0]),
                    "expiration": fit.expiration,
                    "model": fit.model,
                    "weighting": fit.weighting,
                    "action": "surface_rejected",
                    "check": "negative_total_variance",
                    "before_value": minimum_variance,
                    "after_value": None,
                    "reason": "The fitted surface had material negative or non-finite variance.",
                }
            )
            controlled.append(candidate)
            continue

        observed_variance = candidate.predict_total_variance(observed_grid)
        strikes = forward * np.exp(observed_grid)
        volatilities = np.sqrt(np.maximum(observed_variance, 0.0) / time_to_expiry)
        call_prices = np.array(
            [
                black76_price(forward, strike, time_to_expiry, rate, volatility, "call")
                for strike, volatility in zip(strikes, volatilities, strict=True)
            ]
        )
        monotonic_margin = float(np.min(-np.diff(call_prices)))
        slopes = np.diff(call_prices) / np.diff(strikes)
        convexity_margin = float(np.min(np.diff(slopes)))

        extended_grid = np.linspace(
            fit.x_min - extrapolation_padding,
            fit.x_max + extrapolation_padding,
            181,
        )
        extended_variance = candidate.predict_total_variance(extended_grid)
        maximum_observed = max(float(np.nanmax(observed_variance)), variance_tolerance)
        extended_iv = np.sqrt(np.maximum(extended_variance, 0.0) / time_to_expiry)
        unreasonable = (
            not np.all(np.isfinite(extended_variance))
            or float(np.nanmin(extended_variance)) < -10.0 * variance_tolerance
            or float(np.nanmax(extended_iv)) > maximum_extrapolated_iv
            or float(np.nanmax(extended_variance)) > maximum_variance_multiple * maximum_observed
        )
        failed_check = None
        failed_value = None
        if monotonic_margin < -10.0 * price_tolerance:
            failed_check, failed_value = "strike_monotonicity", monotonic_margin
        elif convexity_margin < -10.0 * price_tolerance:
            failed_check, failed_value = "butterfly_convexity", convexity_margin
        elif unreasonable:
            failed_check, failed_value = "unreasonable_extrapolation", float(
                np.nanmax(extended_iv)
            )

        if failed_check is not None:
            candidate = replace(
                fit,
                success=False,
                message=f"{fit.message}; rejected for {failed_check}",
                predictor=None,
            )
            adjustments.append(
                {
                    "quote_date": frame["quote_date"][0],
                    "symbol": str(frame["underlying_symbol"][0]),
                    "expiration": fit.expiration,
                    "model": fit.model,
                    "weighting": fit.weighting,
                    "action": "surface_rejected",
                    "check": failed_check,
                    "before_value": failed_value,
                    "after_value": None,
                    "reason": "The fitted surface failed a configured no-arbitrage control.",
                }
            )
        controlled.append(candidate)

    # Reject a longer fit when a like-for-like model and weighting has material calendar crossing.
    updated = list(controlled)
    for model in {fit.model for fit in updated}:
        for weighting in {fit.weighting for fit in updated if fit.model == model}:
            indices = [
                index
                for index, fit in enumerate(updated)
                if fit.model == model and fit.weighting == weighting and fit.success
            ]
            indices.sort(
                key=lambda index: float(
                    by_expiration[updated[index].expiration]["time_to_expiry"].median()
                )
            )
            for short_index, long_index in pairwise(indices):
                short_fit, long_fit = updated[short_index], updated[long_index]
                lower = max(float(short_fit.x_min), float(long_fit.x_min))
                upper = min(float(short_fit.x_max), float(long_fit.x_max))
                if upper <= lower:
                    continue
                grid = np.linspace(lower, upper, 101)
                margin = float(
                    np.min(
                        long_fit.predict_total_variance(grid)
                        - short_fit.predict_total_variance(grid)
                    )
                )
                if margin < -calendar_tolerance:
                    updated[long_index] = replace(
                        long_fit,
                        success=False,
                        message=f"{long_fit.message}; rejected for calendar inconsistency",
                        predictor=None,
                    )
                    adjustments.append(
                        {
                            "quote_date": by_expiration[long_fit.expiration]["quote_date"][0],
                            "symbol": str(
                                by_expiration[long_fit.expiration]["underlying_symbol"][0]
                            ),
                            "expiration": long_fit.expiration,
                            "model": long_fit.model,
                            "weighting": long_fit.weighting,
                            "action": "surface_rejected",
                            "check": "calendar_consistency",
                            "before_value": margin,
                            "after_value": None,
                            "reason": "Total variance crossed below the preceding expiration.",
                        }
                    )

    frame = (
        pl.DataFrame(adjustments, schema=ADJUSTMENT_SCHEMA, strict=False)
        if adjustments
        else pl.DataFrame(schema=ADJUSTMENT_SCHEMA)
    )
    return updated, frame


def _fitted_diagnostics(
    iv_data: pl.DataFrame,
    details: pl.DataFrame,
    fits: list[SmileFit],
    *,
    price_tolerance: float,
    variance_tolerance: float,
    calendar_tolerance: float,
    extrapolation_padding: float,
    maximum_extrapolated_iv: float,
    maximum_variance_multiple: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_by_model = best_fits_by_expiration_and_model(details, fits)
    selected: dict[object, SmileFit] = {}
    for expiration in iv_data["expiration"].unique().to_list():
        candidates = [
            fit
            for (candidate_expiration, _), fit in selected_by_model.items()
            if candidate_expiration == expiration and fit.success
        ]
        if not candidates:
            continue
        detail_rows = details.filter(
            (pl.col("expiration") == expiration) & pl.col("fit_success")
        ).sort("rmse")
        best = detail_rows.row(0, named=True)
        selected[expiration] = next(
            fit
            for fit in candidates
            if fit.model == best["model"] and fit.weighting == best["weighting"]
        )

    ordered_expirations = sorted(
        selected,
        key=lambda expiration: float(
            iv_data.filter(pl.col("expiration") == expiration)["time_to_expiry"].median()
        ),
    )
    for expiration in ordered_expirations:
        fit = selected[expiration]
        frame = iv_data.filter(pl.col("expiration") == expiration)
        time_to_expiry = float(frame["time_to_expiry"].median())
        forward = float(frame["forward"].median())
        rate = float(frame["interpolated_rate"].median())
        grid = np.linspace(float(fit.x_min), float(fit.x_max), 151)
        variance = fit.predict_total_variance(grid)
        strikes = forward * np.exp(grid)
        volatility = np.sqrt(np.maximum(variance, 0.0) / time_to_expiry)
        prices = np.array(
            [
                black76_price(forward, strike, time_to_expiry, rate, vol, "call")
                for strike, vol in zip(strikes, volatility, strict=True)
            ]
        )
        monotonic_margin = float(np.min(-np.diff(prices)))
        convexity_margin = float(np.min(np.diff(np.diff(prices) / np.diff(strikes))))
        minimum_variance = float(np.nanmin(variance))
        rows.extend(
            [
                _diagnostic(
                    frame,
                    source="fitted_surface",
                    model=fit.model,
                    weighting=fit.weighting,
                    check="strike_monotonicity",
                    option_type="call",
                    location=None,
                    value=monotonic_margin,
                    tolerance=price_tolerance,
                    message="Call prices implied by the selected fit must decrease with strike.",
                ),
                _diagnostic(
                    frame,
                    source="fitted_surface",
                    model=fit.model,
                    weighting=fit.weighting,
                    check="butterfly_convexity",
                    option_type="call",
                    location=None,
                    value=convexity_margin,
                    tolerance=price_tolerance,
                    message="Call prices implied by the selected fit must be convex in strike.",
                ),
                _diagnostic(
                    frame,
                    source="fitted_surface",
                    model=fit.model,
                    weighting=fit.weighting,
                    check="negative_total_variance",
                    option_type="all",
                    location=None,
                    value=minimum_variance,
                    tolerance=variance_tolerance,
                    message="The selected fitted total-variance curve must be non-negative.",
                ),
            ]
        )
        extended = np.linspace(
            float(fit.x_min) - extrapolation_padding,
            float(fit.x_max) + extrapolation_padding,
            181,
        )
        extended_variance = fit.predict_total_variance(extended)
        maximum_observed = max(float(np.nanmax(variance)), variance_tolerance)
        extended_iv = np.sqrt(np.maximum(extended_variance, 0.0) / time_to_expiry)
        extrapolation_margin = (
            min(
                maximum_extrapolated_iv - float(np.nanmax(extended_iv)),
                maximum_variance_multiple * maximum_observed
                - float(np.nanmax(extended_variance)),
                float(np.nanmin(extended_variance)),
            )
            if np.all(np.isfinite(extended_variance))
            else -float("inf")
        )
        rows.append(
            _diagnostic(
                frame,
                source="fitted_surface",
                model=fit.model,
                weighting=fit.weighting,
                check="unreasonable_extrapolation",
                option_type="all",
                location=None,
                value=extrapolation_margin,
                tolerance=variance_tolerance,
                message=(
                    "The fitted wings must remain finite, positive, and within "
                    "configured limits."
                ),
            )
        )

    for short_expiration, long_expiration in pairwise(ordered_expirations):
        short_fit, long_fit = selected[short_expiration], selected[long_expiration]
        lower = max(float(short_fit.x_min), float(long_fit.x_min))
        upper = min(float(short_fit.x_max), float(long_fit.x_max))
        if upper <= lower:
            continue
        grid = np.linspace(lower, upper, 101)
        margin_values = (
            long_fit.predict_total_variance(grid)
            - short_fit.predict_total_variance(grid)
        )
        frame = iv_data.filter(pl.col("expiration") == long_expiration)
        rows.append(
            _diagnostic(
                frame,
                source="fitted_surface",
                model=long_fit.model,
                weighting=long_fit.weighting,
                check="calendar_consistency",
                option_type="all",
                location=float(grid[int(np.argmin(margin_values))]),
                value=float(np.min(margin_values)),
                tolerance=calendar_tolerance,
                message=(
                    f"Selected fitted variance at {long_expiration} must exceed "
                    f"{short_expiration}."
                ),
            )
        )
    return rows


def build_arbitrage_diagnostics(
    iv_data: pl.DataFrame,
    details: pl.DataFrame,
    fits: list[SmileFit],
    *,
    price_tolerance: float = 1e-6,
    variance_tolerance: float = 1e-10,
    calendar_tolerance: float = 1e-6,
    extrapolation_padding: float = 0.15,
    maximum_extrapolated_iv: float = 3.0,
    maximum_variance_multiple: float = 8.0,
) -> pl.DataFrame:
    """Return midpoint, executable bid-ask, and fitted-surface diagnostics."""

    rows = _market_static_diagnostics(iv_data, price_tolerance)
    rows.extend(_variance_diagnostics(iv_data, variance_tolerance))
    rows.extend(_market_calendar_diagnostics(iv_data, calendar_tolerance))
    rows.extend(
        _fitted_diagnostics(
            iv_data,
            details,
            fits,
            price_tolerance=price_tolerance,
            variance_tolerance=variance_tolerance,
            calendar_tolerance=calendar_tolerance,
            extrapolation_padding=extrapolation_padding,
            maximum_extrapolated_iv=maximum_extrapolated_iv,
            maximum_variance_multiple=maximum_variance_multiple,
        )
    )
    return (
        pl.DataFrame(rows, schema=DIAGNOSTIC_SCHEMA, strict=False)
        if rows
        else pl.DataFrame(schema=DIAGNOSTIC_SCHEMA)
    )


def mark_resolved_diagnostics(
    diagnostics: pl.DataFrame,
    adjustments: pl.DataFrame,
) -> pl.DataFrame:
    # Mark fitted violations resolved by an exact model control

    if diagnostics.is_empty() or adjustments.is_empty():
        return diagnostics
    resolved_keys = {
        (
            row["quote_date"],
            row["symbol"],
            row["expiration"],
            row["model"],
            row["weighting"],
            row["check"],
        )
        for row in adjustments.iter_rows(named=True)
    }
    rows = []
    for row in diagnostics.iter_rows(named=True):
        key = (
            row["quote_date"],
            row["symbol"],
            row["expiration"],
            row.get("model"),
            row.get("weighting"),
            row["check"],
        )
        row["resolved"] = bool(
            row["source"] == "fitted_surface"
            and row["is_violation"]
            and key in resolved_keys
        )
        rows.append(row)
    return pl.DataFrame(rows, schema=DIAGNOSTIC_SCHEMA, strict=False)


def build_arbitrage_report(
    diagnostics: pl.DataFrame,
    adjustments: pl.DataFrame,
    path: str | Path,
) -> None:
    # Write a compact Markdown arbitrage report

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    violations = diagnostics.filter(pl.col("is_violation"))
    unresolved = violations.filter(~pl.col("resolved"))
    lines = [
        "# Arbitrage diagnostic report",
        "",
        f"- Checks evaluated: {diagnostics.height}",
        f"- Violations found: {violations.height}",
        f"- Unresolved violations: {unresolved.height}",
        f"- Surface adjustments or rejections: {adjustments.height}",
        "",
        "## Violations by source and test",
        "",
    ]
    if violations.is_empty():
        lines.append("No configured violations were found.")
    else:
        summary = (
            violations.group_by(["source", "check"])
            .agg(
                pl.len().alias("count"),
                pl.col("resolved").sum().alias("resolved_count"),
                pl.col("value").min().alias("worst_margin"),
            )
            .sort(["source", "check"])
        )
        lines.extend(
            [
                "| Source | Test | Count | Resolved | Worst margin |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in summary.iter_rows(named=True):
            lines.append(
                f"| {row['source']} | {row['check']} | {row['count']} | "
                f"{row['resolved_count']} | {row['worst_margin']:.8f} |"
            )
    lines.extend(["", "## Recorded controls", ""])
    if adjustments.is_empty():
        lines.append("No fitted-surface control was required.")
    else:
        lines.extend(
            [
                "| Expiration | Model | Weighting | Action | Test |",
                "|---|---|---|---|---|",
            ]
        )
        for row in adjustments.iter_rows(named=True):
            lines.append(
                f"| {row['expiration']} | {row['model']} | {row['weighting']} | "
                f"{row['action']} | {row['check']} |"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
