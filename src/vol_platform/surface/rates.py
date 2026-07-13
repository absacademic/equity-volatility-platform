# Zero-rate interpolation and discount-factor helper functions

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class InterpolatedRate:
    rate: float
    discount_factor: float
    source_as_of_date: date | None
    used_default: bool


def discount_factor(rate: float, time_to_expiry: float) -> float:
    # Returns continuous-compounding discount factor exp(-rT)

    if not math.isfinite(rate):
        raise ValueError("rate must be finite")
    if not math.isfinite(time_to_expiry) or time_to_expiry < 0.0:
        raise ValueError("time_to_expiry must be finite and nonnegative")
    return math.exp(-rate * time_to_expiry)


def interpolate_rate(
    curve: pl.DataFrame | None,
    *,
    quote_date: date,
    expiration: date,
    default_rate: float,
    day_count_basis: float = 365.0,
    currency: str = "USD",
) -> InterpolatedRate:
    # Linearly interpolate continuously compounded zero rates by maturity

    if day_count_basis <= 0.0:
        raise ValueError("day_count_basis must be positive")
    if curve is None or curve.is_empty():
        return InterpolatedRate(
            rate=default_rate,
            discount_factor=discount_factor(
                default_rate, max((expiration - quote_date).days / day_count_basis, 0.0)
            ),
            source_as_of_date=None,
            used_default=True,
        )

    candidates = curve.filter(
        (pl.col("as_of_date") <= quote_date) & (pl.col("currency") == currency)
    )
    if candidates.is_empty():
        return InterpolatedRate(
            rate=default_rate,
            discount_factor=discount_factor(
                default_rate, max((expiration - quote_date).days / day_count_basis, 0.0)
            ),
            source_as_of_date=None,
            used_default=True,
        )
    latest = candidates["as_of_date"].max()
    points = candidates.filter(pl.col("as_of_date") == latest).sort("maturity_date")
    times = np.array(
        [
            max((maturity - latest).days / day_count_basis, 0.0)
            for maturity in points["maturity_date"]
        ],
        dtype=float,
    )
    rates = np.array(points["rate"], dtype=float)
    target = max((expiration - latest).days / day_count_basis, 0.0)
    rate = float(np.interp(target, times, rates, left=rates[0], right=rates[-1]))
    quote_time = max((expiration - quote_date).days / day_count_basis, 0.0)
    return InterpolatedRate(
        rate=rate,
        discount_factor=discount_factor(rate, quote_time),
        source_as_of_date=latest,
        used_default=False,
    )
