import math

import pytest

from vol_platform.pricing import black_scholes
from vol_platform.pricing.bounds import black_scholes_bounds
from vol_platform.pricing.parity import (
    black_scholes_parity_rhs,
    forward_from_spot,
    implied_forward_from_parity,
    parity_residual,
)
from vol_platform.pricing.values import intrinsic_value, time_value


def test_black_scholes_known_call_value() -> None:
    result = black_scholes.price(100.0, 100.0, 1.0, 0.05, 0.20, "call")
    assert result == pytest.approx(10.4505835722, rel=1e-10)


def test_black_scholes_known_put_value() -> None:
    result = black_scholes.price(100.0, 100.0, 1.0, 0.05, 0.20, "put")
    assert result == pytest.approx(5.5735260223, rel=1e-10)


def test_black_scholes_at_expiry_returns_payoff() -> None:
    assert black_scholes.price(110, 100, 0, 0.05, 0.2, "call") == 10.0
    assert black_scholes.price(90, 100, 0, 0.05, 0.2, "put") == 10.0


def test_black_scholes_zero_volatility_returns_discounted_deterministic_value() -> None:
    result = black_scholes.price(100, 95, 0.5, 0.04, 0.0, "call", 0.01)
    expected = max(100 * math.exp(-0.01 * 0.5) - 95 * math.exp(-0.04 * 0.5), 0.0)
    assert result == pytest.approx(expected)


def test_black_scholes_put_call_parity() -> None:
    args = (102.0, 100.0, 0.75, 0.035, 0.24)
    call = black_scholes.price(*args, "call", 0.012)
    put = black_scholes.price(*args, "put", 0.012)
    rhs = black_scholes_parity_rhs(102.0, 100.0, 0.75, 0.035, 0.012)
    assert parity_residual(call, put, rhs) == pytest.approx(0.0, abs=1e-12)


def test_forward_recovered_from_put_call_parity() -> None:
    spot, strike, maturity, rate, dividend, sigma = 100.0, 103.0, 0.6, 0.04, 0.01, 0.27
    call = black_scholes.price(spot, strike, maturity, rate, sigma, "call", dividend)
    put = black_scholes.price(spot, strike, maturity, rate, sigma, "put", dividend)
    recovered = implied_forward_from_parity(call, put, strike, maturity, rate)
    expected = forward_from_spot(spot, maturity, rate, dividend)
    assert recovered == pytest.approx(expected, rel=1e-12)


def test_black_scholes_bounds_contain_price() -> None:
    price = black_scholes.price(100, 120, 1.2, 0.03, 0.5, "put", 0.01)
    lower, upper = black_scholes_bounds(100, 120, 1.2, 0.03, "put", 0.01)
    assert lower <= price <= upper


def test_intrinsic_and_time_value() -> None:
    intrinsic = intrinsic_value(110, 100, "call")
    assert intrinsic == 10.0
    assert time_value(13.5, intrinsic) == 3.5


def test_european_time_value_can_be_negative() -> None:
    assert time_value(9.0, 10.0) == -1.0


def test_black_scholes_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match=r"spot|underlying"):
        black_scholes.price(-1, 100, 1, 0.03, 0.2)
    with pytest.raises(ValueError, match="option_type"):
        black_scholes.price(100, 100, 1, 0.03, 0.2, "straddle")
