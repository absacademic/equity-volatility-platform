# Put-call parity identities/diagnostics

from __future__ import annotations

import math

from vol_platform.pricing._validation import require_finite


def black_scholes_parity_rhs(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float = 0.0,
) -> float:
    # Returns the theoretical value of call-put under Black-Scholes
    values = {
        "spot": spot,
        "strike": strike,
        "time_to_expiry": time_to_expiry,
        "rate": rate,
        "dividend_yield": dividend_yield,
    }
    values = {name: require_finite(name, value) for name, value in values.items()}
    if values["spot"] <= 0.0 or values["strike"] <= 0.0:
        raise ValueError("spot and strike must be positive")
    if values["time_to_expiry"] < 0.0:
        raise ValueError("time_to_expiry cannot be negative")
    return values["spot"] * math.exp(-values["dividend_yield"] * values["time_to_expiry"]) - values[
        "strike"
    ] * math.exp(-values["rate"] * values["time_to_expiry"])


def black76_parity_rhs(
    forward: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
) -> float:
    # Return the theoretical value of call-put under Black-76
    forward = require_finite("forward", forward)
    strike = require_finite("strike", strike)
    time_to_expiry = require_finite("time_to_expiry", time_to_expiry)
    rate = require_finite("rate", rate)
    if forward <= 0.0 or strike <= 0.0:
        raise ValueError("forward and strike must be positive")
    if time_to_expiry < 0.0:
        raise ValueError("time_to_expiry cannot be negative")
    return math.exp(-rate * time_to_expiry) * (forward - strike)


def parity_residual(call_price: float, put_price: float, parity_rhs: float) -> float:
    # Return call-put parity_rhs; zero indicates exact parity
    call_price = require_finite("call_price", call_price)
    put_price = require_finite("put_price", put_price)
    parity_rhs = require_finite("parity_rhs", parity_rhs)
    return call_price - put_price - parity_rhs


def forward_from_spot(
    spot: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float = 0.0,
) -> float:
    # Converts spot to its continuously compounded theoretical forward
    spot = require_finite("spot", spot)
    time_to_expiry = require_finite("time_to_expiry", time_to_expiry)
    rate = require_finite("rate", rate)
    dividend_yield = require_finite("dividend_yield", dividend_yield)
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if time_to_expiry < 0.0:
        raise ValueError("time_to_expiry cannot be negative")
    return spot * math.exp((rate - dividend_yield) * time_to_expiry)


def implied_forward_from_parity(
    call_price: float,
    put_price: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
) -> float:
    # Infers a forward from a same-strike call-put pair
    call_price = require_finite("call_price", call_price)
    put_price = require_finite("put_price", put_price)
    strike = require_finite("strike", strike)
    time_to_expiry = require_finite("time_to_expiry", time_to_expiry)
    rate = require_finite("rate", rate)
    if call_price < 0.0 or put_price < 0.0:
        raise ValueError("option prices cannot be negative")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if time_to_expiry < 0.0:
        raise ValueError("time_to_expiry cannot be negative")
    return strike + math.exp(rate * time_to_expiry) * (call_price - put_price)
