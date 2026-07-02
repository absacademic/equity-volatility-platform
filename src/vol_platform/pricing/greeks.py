from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from vol_platform.pricing import black76, black_scholes
from vol_platform.pricing._validation import (
    coerce_option_type,
    require_finite,
    validate_common_inputs,
)
from vol_platform.types import OptionType, PricingModel


@dataclass(frozen=True, slots=True)
class Greeks:
    """Core option sensitivities.

    Vega is per 1.00 absolute volatility change, not per volatility point.
    Theta is calendar-time decay per year. Black-76 theta holds the forward fixed.
    """

    delta: float
    gamma: float
    vega: float
    theta: float


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
) -> Greeks:
    # Return analytic Black-Scholes delta, gamma, vega, and theta
    spot, strike, time_to_expiry, rate, volatility = validate_common_inputs(
        spot, strike, time_to_expiry, rate, volatility
    )
    dividend_yield = require_finite("dividend_yield", dividend_yield)
    option_type = coerce_option_type(option_type)
    if time_to_expiry == 0.0 or volatility == 0.0:
        raise ValueError("Greeks require positive time_to_expiry and volatility")

    d1, d2 = black_scholes.d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    root_t = math.sqrt(time_to_expiry)
    discount_r = math.exp(-rate * time_to_expiry)
    discount_q = math.exp(-dividend_yield * time_to_expiry)
    density = norm.pdf(d1)

    if option_type is OptionType.CALL:
        delta = discount_q * norm.cdf(d1)
        theta = (
            -(spot * discount_q * density * volatility) / (2.0 * root_t)
            - rate * strike * discount_r * norm.cdf(d2)
            + dividend_yield * spot * discount_q * norm.cdf(d1)
        )
    else:
        delta = discount_q * (norm.cdf(d1) - 1.0)
        theta = (
            -(spot * discount_q * density * volatility) / (2.0 * root_t)
            + rate * strike * discount_r * norm.cdf(-d2)
            - dividend_yield * spot * discount_q * norm.cdf(-d1)
        )

    gamma = discount_q * density / (spot * volatility * root_t)
    vega = spot * discount_q * density * root_t
    return Greeks(delta=float(delta), gamma=float(gamma), vega=float(vega), theta=float(theta))


def black76_greeks(
    forward: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType | str = OptionType.CALL,
) -> Greeks:
    forward, strike, time_to_expiry, rate, volatility = validate_common_inputs(
        forward, strike, time_to_expiry, rate, volatility
    )
    option_type = coerce_option_type(option_type)
    if time_to_expiry == 0.0 or volatility == 0.0:
        raise ValueError("Greeks require positive time_to_expiry and volatility")

    d1, _ = black76.d1_d2(forward, strike, time_to_expiry, rate, volatility)
    root_t = math.sqrt(time_to_expiry)
    discount = math.exp(-rate * time_to_expiry)
    density = norm.pdf(d1)

    if option_type is OptionType.CALL:
        delta = discount * norm.cdf(d1)
    else:
        delta = discount * (norm.cdf(d1) - 1.0)

    gamma = discount * density / (forward * volatility * root_t)
    vega = discount * forward * density * root_t
    option_price = black76.price(forward, strike, time_to_expiry, rate, volatility, option_type)
    theta = rate * option_price - discount * forward * density * volatility / (2.0 * root_t)
    return Greeks(delta=float(delta), gamma=float(gamma), vega=float(vega), theta=float(theta))


def calculate_greeks(
    model: PricingModel | str,
    underlying: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType | str,
    dividend_yield: float = 0.0,
) -> Greeks:
    # Dispatch Greek calculation to the requested model
    model = PricingModel(model)
    if model is PricingModel.BLACK_SCHOLES:
        return black_scholes_greeks(
            underlying,
            strike,
            time_to_expiry,
            rate,
            volatility,
            option_type,
            dividend_yield,
        )
    return black76_greeks(
        underlying,
        strike,
        time_to_expiry,
        rate,
        volatility,
        option_type,
    )
