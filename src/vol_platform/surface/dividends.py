# Discrete-dividend and early-exercise adjustments

from __future__ import annotations

import math
from datetime import date
from typing import Any

import polars as pl


def _year_fraction(start: date, end: date, basis: float) -> float:
    return max((end - start).days / basis, 0.0)


def add_dividend_and_exercise_features(
    quotes: pl.DataFrame,
    dividends: pl.DataFrame | None,
    *,
    day_count_basis: float = 365.0,
    exercise_style: str = "american",
) -> pl.DataFrame:
    # Attach point-in-time dividend values and simple early-exercise risk flags

    dividend_rows = dividends.iter_rows(named=True) if dividends is not None else []
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in dividend_rows:
        if row.get("symbol") is None or row.get("ex_date") is None:
            continue
        if row.get("amount") is None or float(row["amount"]) < 0.0:
            continue
        by_symbol.setdefault(str(row["symbol"]).upper(), []).append(row)

    output: list[dict[str, Any]] = []
    for row in quotes.iter_rows(named=True):
        quote_timestamp = row["quote_timestamp"]
        quote_date = quote_timestamp.date()
        expiration = row["expiration"]
        symbol = str(row.get("underlying_symbol") or row.get("symbol")).upper()
        rate = float(row["interpolated_rate"])
        spot = float(row["underlying_price"])
        time_to_expiry = float(row["time_to_expiry"])

        eligible_dividends = [
            dividend
            for dividend in by_symbol.get(symbol, [])
            if quote_date < dividend["ex_date"] <= expiration
            and (
                dividend.get("known_timestamp") is None
                or dividend["known_timestamp"] <= quote_timestamp
            )
        ]
        present_value = sum(
            float(dividend["amount"])
            * math.exp(-rate * _year_fraction(quote_date, dividend["ex_date"], day_count_basis))
            for dividend in eligible_dividends
        )
        adjusted_spot = max(spot - present_value, 1e-12)
        yield_estimate = (
            -math.log(adjusted_spot / spot) / time_to_expiry
            if present_value > 0.0 and time_to_expiry > 0.0
            else 0.0
        )
        theoretical_forward = adjusted_spot * math.exp(rate * time_to_expiry)

        strike = float(row["strike"])
        mid = float(row["mid"])
        spread = float(row.get("spread") or 0.0)
        option_type = str(row["option_type"])
        intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        time_value = max(mid - intrinsic, 0.0)
        interest_benefit = strike * max(1.0 - math.exp(-rate * time_to_expiry), 0.0)

        american = exercise_style.lower() == "american"
        call_risk = (
            american
            and option_type == "call"
            and intrinsic > 0.0
            and present_value > interest_benefit + time_value + max(spread, 1e-8)
        )
        put_risk = (
            american
            and option_type == "put"
            and intrinsic > 0.0
            and rate > 0.0
            and time_value <= max(spread, 0.0025 * spot)
        )

        enriched = dict(row)
        enriched.update(
            {
                "exercise_style": exercise_style.lower(),
                "dividend_count_to_expiry": len(eligible_dividends),
                "dividend_present_value": present_value,
                "dividend_adjusted_spot": adjusted_spot,
                "dividend_yield_estimate": yield_estimate,
                "dividend_adjusted_forward": theoretical_forward,
                "option_intrinsic_value": intrinsic,
                "option_time_value": time_value,
                "early_exercise_risk": bool(call_risk or put_risk),
                "early_exercise_reason": (
                    "dividend_call" if call_risk else "deep_itm_put" if put_risk else None
                ),
            }
        )
        output.append(enriched)
    return pl.DataFrame(output) if output else quotes


def apply_dividend_forward_adjustments(
    forwards: pl.DataFrame,
    enriched_quotes: pl.DataFrame,
    *,
    use_fallback: bool = True,
) -> pl.DataFrame:
    # Add dividend-aware forward comparisons and fill missing parity estimates

    if forwards.is_empty():
        return forwards

    quote_summary = (
        enriched_quotes.group_by("expiration")
        .agg(
            pl.col("dividend_present_value").median(),
            pl.col("dividend_yield_estimate").median(),
            pl.col("dividend_adjusted_forward").median().alias("theoretical_forward"),
        )
        .sort("expiration")
    )
    rows: list[dict[str, Any]] = []
    for row in forwards.join(quote_summary, on="expiration", how="left").iter_rows(named=True):
        parity_forward = row.get("forward")
        theoretical = row.get("theoretical_forward")
        fallback = bool(use_fallback and parity_forward is None and theoretical is not None)
        if fallback:
            row["forward"] = float(theoretical)
            row["reliability"] = "low"
        row["forward_source"] = "dividend_adjusted_spot_fallback" if fallback else "put_call_parity"
        row["forward_dividend_difference"] = (
            float(row["forward"]) - float(theoretical)
            if row.get("forward") is not None and theoretical is not None
            else None
        )
        rows.append(row)
    return pl.DataFrame(rows)
