import math

import pytest

from vol_platform.pricing import black76, black_scholes
from vol_platform.pricing.bounds import black76_bounds
from vol_platform.pricing.parity import black76_parity_rhs, parity_residual


def test_black76_matches_black_scholes_via_forward_transformation() -> None:
    spot, strike, maturity, rate, dividend, sigma = 100.0, 105.0, 0.8, 0.04, 0.015, 0.3
    forward = spot * math.exp((rate - dividend) * maturity)
    bs_call = black_scholes.price(spot, strike, maturity, rate, sigma, "call", dividend)
    b76_call = black76.price(forward, strike, maturity, rate, sigma, "call")
    assert b76_call == pytest.approx(bs_call, rel=1e-12)


def test_black76_put_call_parity() -> None:
    forward, strike, maturity, rate, sigma = 103.0, 100.0, 0.5, 0.045, 0.22
    call = black76.price(forward, strike, maturity, rate, sigma, "call")
    put = black76.price(forward, strike, maturity, rate, sigma, "put")
    rhs = black76_parity_rhs(forward, strike, maturity, rate)
    assert parity_residual(call, put, rhs) == pytest.approx(0.0, abs=1e-12)


def test_black76_expiry_and_zero_volatility() -> None:
    assert black76.price(110, 100, 0, 0.03, 0.2, "call") == 10.0
    expected = math.exp(-0.03) * 10.0
    assert black76.price(110, 100, 1, 0.03, 0.0, "call") == pytest.approx(expected)


def test_black76_bounds_contain_price() -> None:
    price = black76.price(98, 100, 0.4, 0.05, 0.35, "put")
    lower, upper = black76_bounds(98, 100, 0.4, 0.05, "put")
    assert lower <= price <= upper


def test_black76_rejects_negative_time() -> None:
    with pytest.raises(ValueError, match="negative"):
        black76.price(100, 100, -0.1, 0.03, 0.2)
