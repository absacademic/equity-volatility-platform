# Some intrinsic-value and time-value utilities

from __future__ import annotations

from vol_platform.pricing._validation import coerce_option_type, require_finite
from vol_platform.types import OptionType


def intrinsic_value(
    underlying: float,
    strike: float,
    option_type: OptionType | str,
) -> float:
    # Return the immediate exercise payoff, w/o discounting
    underlying = require_finite("underlying", underlying)
    strike = require_finite("strike", strike)
    option_type = coerce_option_type(option_type)

    if underlying < 0.0 or strike < 0.0:
        raise ValueError("underlying and strike cannot be negative")
    difference = underlying - strike
    return max(difference if option_type is OptionType.CALL else -difference, 0.0)


def time_value(option_price: float, intrinsic: float) -> float:
    """Return the portion of an option price exceeding intrinsic value."""

    option_price = require_finite("option_price", option_price)
    intrinsic = require_finite("intrinsic", intrinsic)
    if option_price < 0.0 or intrinsic < 0.0:
        raise ValueError("option_price and instrinsic cannot be negative")
    return option_price - intrinsic
