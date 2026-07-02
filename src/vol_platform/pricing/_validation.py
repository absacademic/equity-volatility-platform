from __future__ import annotations

import math

from vol_platform.types import OptionType


def coerce_option_type(option_type: OptionType | str) -> OptionType:
    # Convert a string to :class:`OptionType` and raise an error if invalid
    try:
        return OptionType(option_type)
    except ValueError as exc:
        raise ValueError("option_type must be 'call' or 'put'") from exc


def require_finite(name: str, value: float) -> float:
    # Validate that a numeric input is finite
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_common_inputs(
    underlying: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
) -> tuple[float, float, float, float, float]:
    # Validate shared Black-Scholes/Black-76 scalar inputs
    underlying = require_finite("underlying", underlying)
    strike = require_finite("strike", strike)
    time_to_expiry = require_finite("time_to_expiry", time_to_expiry)
    rate = require_finite("rate", rate)
    volatility = require_finite("volatility", volatility)

    if underlying <= 0.0:
        raise ValueError("underlying must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if time_to_expiry < 0.0:
        raise ValueError("time_to_expiry cannot be negative")
    if volatility < 0.0:
        raise ValueError("volatility cannot be negative")

    return underlying, strike, time_to_expiry, rate, volatility
