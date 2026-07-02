import pytest

from vol_platform.pricing import black_scholes
from vol_platform.pricing.greeks import black_scholes_greeks
from vol_platform.pricing.implied_vol import solve_implied_volatility


def test_week_one_completion_criterion() -> None:
    # Price -> recover volatility -> verify delta and vega by finite differences
    spot, strike, maturity, rate, dividend, sigma = 100.0, 105.0, 0.65, 0.035, 0.01, 0.28
    option_price = black_scholes.price(spot, strike, maturity, rate, sigma, "call", dividend)
    recovered = solve_implied_volatility(
        option_price,
        spot,
        strike,
        maturity,
        rate,
        "call",
        dividend_yield=dividend,
    )
    assert recovered.converged
    assert recovered.volatility == pytest.approx(sigma, abs=1e-9)

    greeks = black_scholes_greeks(spot, strike, maturity, rate, sigma, "call", dividend)
    spot_step = 1e-3
    vol_step = 1e-5
    delta_fd = (
        black_scholes.price(spot + spot_step, strike, maturity, rate, sigma, "call", dividend)
        - black_scholes.price(spot - spot_step, strike, maturity, rate, sigma, "call", dividend)
    ) / (2 * spot_step)
    vega_fd = (
        black_scholes.price(spot, strike, maturity, rate, sigma + vol_step, "call", dividend)
        - black_scholes.price(spot, strike, maturity, rate, sigma - vol_step, "call", dividend)
    ) / (2 * vol_step)
    assert greeks.delta == pytest.approx(delta_fd, rel=2e-7)
    assert greeks.vega == pytest.approx(vega_fd, rel=2e-8)
