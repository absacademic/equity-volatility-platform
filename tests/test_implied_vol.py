import pytest

from vol_platform.pricing import black76, black_scholes
from vol_platform.pricing.bounds import black76_bounds, black_scholes_bounds
from vol_platform.pricing.implied_vol import IVMethod, IVStatus, solve_implied_volatility


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_recovers_black_scholes_volatility(option_type: str, strike: float) -> None:
    target_vol = 0.37
    option_price = black_scholes.price(100, strike, 0.85, 0.04, target_vol, option_type, 0.01)
    result = solve_implied_volatility(
        option_price, 100, strike, 0.85, 0.04, option_type, dividend_yield=0.01
    )
    assert result.status is IVStatus.SUCCESS
    assert result.volatility == pytest.approx(target_vol, abs=2e-9)
    assert abs(result.residual or 0.0) < 1e-8


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_recovers_black76_volatility(option_type: str) -> None:
    option_price = black76.price(103, 100, 0.45, 0.05, 0.29, option_type)
    result = solve_implied_volatility(
        option_price,
        103,
        100,
        0.45,
        0.05,
        option_type,
        model="black_76",
        initial_volatility=2.0,
    )
    assert result.converged
    assert result.volatility == pytest.approx(0.29, abs=2e-9)


def test_solver_reports_price_below_lower_bound() -> None:
    lower, _ = black_scholes_bounds(100, 80, 1, 0.05, "call", 0.0)
    result = solve_implied_volatility(lower - 0.01, 100, 80, 1, 0.05, "call")
    assert result.status is IVStatus.PRICE_BELOW_LOWER_BOUND
    assert not result.converged


def test_solver_reports_price_above_upper_bound() -> None:
    _, upper = black_scholes_bounds(100, 100, 1, 0.05, "call", 0.0)
    result = solve_implied_volatility(upper + 0.01, 100, 100, 1, 0.05, "call")
    assert result.status is IVStatus.PRICE_ABOVE_UPPER_BOUND


def test_solver_handles_lower_bound_as_zero_volatility() -> None:
    lower, _ = black76_bounds(110, 100, 1, 0.03, "call")
    result = solve_implied_volatility(lower, 110, 100, 1, 0.03, "call", model="black_76")
    assert result.status is IVStatus.AT_LOWER_BOUND
    assert result.volatility == 0.0
    assert result.converged


def test_solver_reports_upper_bound() -> None:
    _, upper = black76_bounds(100, 100, 1, 0.03, "call")
    result = solve_implied_volatility(upper, 100, 100, 1, 0.03, "call", model="black_76")
    assert result.status is IVStatus.AT_UPPER_BOUND
    assert result.volatility is None


def test_solver_reports_expired_contract() -> None:
    result = solve_implied_volatility(10, 110, 100, 0, 0.03, "call")
    assert result.status is IVStatus.EXPIRED


def test_solver_returns_invalid_input_status() -> None:
    result = solve_implied_volatility(5, -100, 100, 1, 0.03, "call")
    assert result.status is IVStatus.INVALID_INPUT


def test_solver_reports_bracket_nonconvergence() -> None:
    option_price = black_scholes.price(100, 100, 1, 0.02, 0.8, "call")
    result = solve_implied_volatility(
        option_price, 100, 100, 1, 0.02, "call", maximum_volatility=0.3
    )
    assert result.status is IVStatus.NON_CONVERGENCE


def test_solver_uses_brent_fallback_when_newton_step_is_unsafe() -> None:
    option_price = black_scholes.price(100, 140, 0.2, 0.01, 0.45, "call")
    result = solve_implied_volatility(
        option_price,
        100,
        140,
        0.2,
        0.01,
        "call",
        initial_volatility=9.0,
    )
    assert result.status is IVStatus.SUCCESS
    assert result.method in {IVMethod.BRENTQ, IVMethod.NEWTON}
    assert result.volatility == pytest.approx(0.45, abs=2e-8)


def test_solver_bisection_last_resort(monkeypatch: pytest.MonkeyPatch) -> None:
    import vol_platform.pricing.implied_vol as iv_module

    def fail_brentq(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(iv_module, "brentq", fail_brentq)
    option_price = black_scholes.price(100, 115, 0.5, 0.02, 0.33, "put")
    result = solve_implied_volatility(
        option_price,
        100,
        115,
        0.5,
        0.02,
        "put",
        initial_volatility=9.0,
    )
    assert result.status is IVStatus.SUCCESS
    assert result.method is IVMethod.BISECTION
    assert result.volatility == pytest.approx(0.33, abs=1e-7)
