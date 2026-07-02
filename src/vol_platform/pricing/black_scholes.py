from __future__ import annotations

import math

from scipy.stats import norm

from vol_platform.pricing._validation import (
    coerce_option_type,
    require_finite,
    validate_common_inputs,
)
from vol_platform.types import OptionType


def d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return the Black-Scholes ``d1`` and ``d2`` terms.

    ``time_to_expiry`` and ``volatility`` must both be strictly positive.
    """
    spot, strike, time_to_expiry, rate, volatility = validate_common_inputs(
        spot, strike, time_to_expiry, rate, volatility
    )
    dividend_yield = require_finite("dividend_yield", dividend_yield)
    if time_to_expiry == 0.0:
        raise ValueError("d1 and d2 are undefined at expiry")
    if volatility == 0.0:
        raise ValueError("d1 and d2 are undefined at zero volatility")

    root_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * root_t)
    return d1, d1 - volatility * root_t


def price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European option under Black-Scholes."""
    spot, strike, time_to_expiry, rate, volatility = validate_common_inputs(
        spot, strike, time_to_expiry, rate, volatility
    )
    dividend_yield = require_finite("dividend_yield", dividend_yield)
    option_type = coerce_option_type(option_type)

    if time_to_expiry == 0.0:
        payoff = spot - strike if option_type is OptionType.CALL else strike - spot
        return max(payoff, 0.0)

    discounted_spot = spot * math.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * math.exp(-rate * time_to_expiry)

    if volatility == 0.0:
        deterministic_value = discounted_spot - discounted_strike
        if option_type is OptionType.CALL:
            return max(deterministic_value, 0.0)
        return max(-deterministic_value, 0.0)

    d1, d2 = d1_d2(
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        dividend_yield,
    )
    if option_type is OptionType.CALL:
        return discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    return discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
