from __future__ import annotations

import math

from vol_platform.pricing._validation import coerce_option_type, require_finite
from vol_platform.types import OptionType


def black_scholes_bounds(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: OptionType | str,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    # Return lower and upper European price bounds under spot conventions
    spot = require_finite("spot", spot)
    strike = require_finite("strike", strike)
    time_to_expiry = require_finite("time_to_expiry", time_to_expiry)
    rate = require_finite("rate", rate)
    dividend_yield = require_finite("dividend_yield", dividend_yield)
    option_type = coerce_option_type(option_type)
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if time_to_expiry < 0.0:
        raise ValueError("time_to_expiry cannot be negative")

    discounted_spot = spot * math.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * math.exp(-rate * time_to_expiry)
    if option_type is OptionType.CALL:
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot
    return max(discounted_strike - discounted_spot, 0.0), discounted_strike


def black76_bounds(
    forward: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: OptionType | str,
) -> tuple[float, float]:
    # Return lower and upper European price bounds under Black-76
    forward = require_finite("forward", forward)
    strike = require_finite("strike", strike)
    time_to_expiry = require_finite("time_to_expiry", time_to_expiry)
    rate = require_finite("rate", rate)
    option_type = coerce_option_type(option_type)
    if forward <= 0.0 or strike <= 0.0:
        raise ValueError("forward and strike must be positive")
    if time_to_expiry < 0.0:
        raise ValueError("time_to_expiry cannot be negative")

    discount = math.exp(-rate * time_to_expiry)
    if option_type is OptionType.CALL:
        return discount * max(forward - strike, 0.0), discount * forward
    return discount * max(strike - forward, 0.0), discount * strike
