import pytest

from vol_platform.pricing import black76, black_scholes
from vol_platform.pricing.greeks import black76_greeks, black_scholes_greeks


def central_difference(function, x: float, step: float) -> float:
    return (function(x + step) - function(x - step)) / (2.0 * step)


def second_difference(function, x: float, step: float) -> float:
    return (function(x + step) - 2.0 * function(x) + function(x - step)) / step**2


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_black_scholes_delta_matches_finite_difference(option_type: str) -> None:
    args = dict(strike=105.0, time_to_expiry=0.7, rate=0.03, volatility=0.26)
    dividend = 0.012
    greek = black_scholes_greeks(100.0, option_type=option_type, dividend_yield=dividend, **args)
    numeric = central_difference(
        lambda spot: black_scholes.price(
            spot, option_type=option_type, dividend_yield=dividend, **args
        ),
        100.0,
        1e-3,
    )
    assert greek.delta == pytest.approx(numeric, rel=2e-7, abs=2e-7)


def test_black_scholes_gamma_matches_finite_difference() -> None:
    args = dict(strike=100.0, time_to_expiry=0.8, rate=0.04, volatility=0.2)
    greek = black_scholes_greeks(102.0, option_type="call", dividend_yield=0.01, **args)
    numeric = second_difference(
        lambda spot: black_scholes.price(spot, option_type="call", dividend_yield=0.01, **args),
        102.0,
        0.02,
    )
    assert greek.gamma == pytest.approx(numeric, rel=2e-5, abs=2e-7)


def test_black_scholes_vega_matches_finite_difference() -> None:
    greek = black_scholes_greeks(100, 98, 0.9, 0.025, 0.31, "put", 0.005)
    numeric = central_difference(
        lambda vol: black_scholes.price(100, 98, 0.9, 0.025, vol, "put", 0.005),
        0.31,
        1e-5,
    )
    assert greek.vega == pytest.approx(numeric, rel=2e-8)


def test_black_scholes_theta_matches_finite_difference() -> None:
    greek = black_scholes_greeks(100, 103, 0.6, 0.04, 0.22, "call", 0.01)
    d_price_d_maturity = central_difference(
        lambda maturity: black_scholes.price(100, 103, maturity, 0.04, 0.22, "call", 0.01),
        0.6,
        1e-5,
    )
    assert greek.theta == pytest.approx(-d_price_d_maturity, rel=2e-8)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_black76_delta_matches_finite_difference(option_type: str) -> None:
    greek = black76_greeks(101, 100, 0.75, 0.03, 0.25, option_type)
    numeric = central_difference(
        lambda forward: black76.price(forward, 100, 0.75, 0.03, 0.25, option_type),
        101.0,
        1e-3,
    )
    assert greek.delta == pytest.approx(numeric, rel=2e-7)


def test_black76_gamma_and_vega_match_finite_differences() -> None:
    greek = black76_greeks(101, 100, 0.75, 0.03, 0.25, "call")
    gamma_numeric = second_difference(
        lambda forward: black76.price(forward, 100, 0.75, 0.03, 0.25, "call"),
        101.0,
        0.02,
    )
    vega_numeric = central_difference(
        lambda vol: black76.price(101, 100, 0.75, 0.03, vol, "call"),
        0.25,
        1e-5,
    )
    assert greek.gamma == pytest.approx(gamma_numeric, rel=2e-5)
    assert greek.vega == pytest.approx(vega_numeric, rel=2e-8)


def test_greeks_reject_expiry() -> None:
    with pytest.raises(ValueError, match="positive"):
        black_scholes_greeks(100, 100, 0, 0.03, 0.2)
