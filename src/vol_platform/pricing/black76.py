from __future__ import annotations

import math

from scipy.stats import norm

from vol_platform.pricing._validation import coerce_option_type, validate_common_inputs
from vol_platform.types import OptionType


def d1_d2(
    forward: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
) -> tuple[float, float]:
    # Return the Black-76 ``d1`` and ``d2`` terms
    forward, strike, time_to_expiry, _, volatility = validate_common_inputs(
        forward, strike, time_to_expiry, rate, volatility
    )
    if time_to_expiry == 0.0:
        raise ValueError("d1 and d2 are undefined at expiry")
    if volatility == 0.0:
        raise ValueError("d1 and d2 are undefined at zero volatility")

    root_t = math.sqrt(time_to_expiry)
    d1 = (math.log(forward / strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * root_t)
    return d1, d1 - volatility * root_t


def price(
    forward: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType | str = OptionType.CALL,
) -> float:
    # Price a European option under Black-76
    forward, strike, time_to_expiry, rate, volatility = validate_common_inputs(
        forward, strike, time_to_expiry, rate, volatility
    )
    option_type = coerce_option_type(option_type)

    payoff = forward - strike if option_type is OptionType.CALL else strike - forward
    if time_to_expiry == 0.0:
        return max(payoff, 0.0)

    discount = math.exp(-rate * time_to_expiry)
    if volatility == 0.0:
        return discount * max(payoff, 0.0)

    d1, d2 = d1_d2(forward, strike, time_to_expiry, rate, volatility)
    if option_type is OptionType.CALL:
        return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
