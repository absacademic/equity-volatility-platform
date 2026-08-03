import math

import pytest

from vol_platform.pricing import black_scholes
from vol_platform.pricing.monte_carlo import BarrierType, price_barrier_option


def test_barrier_in_out_parity_with_shared_seed() -> None:
    common = dict(
        spot=100.0,
        strike=100.0,
        barrier=125.0,
        time_to_expiry=1.0,
        rate=0.04,
        volatility=0.25,
        option_type="call",
        paths=40_000,
        steps=64,
        seed=11,
    )
    knock_out = price_barrier_option(**common, barrier_type=BarrierType.UP_AND_OUT)
    knock_in = price_barrier_option(**common, barrier_type=BarrierType.UP_AND_IN)
    vanilla = black_scholes.price(100.0, 100.0, 1.0, 0.04, 0.25, "call")

    assert knock_out.price + knock_in.price == pytest.approx(vanilla, abs=0.18)
    assert 0.0 < knock_out.knock_probability < 1.0
    assert knock_out.confidence_lower_95 <= knock_out.price <= knock_out.confidence_upper_95


def test_immediate_knock_out_returns_discounted_rebate() -> None:
    result = price_barrier_option(
        100.0,
        100.0,
        90.0,
        0.5,
        0.03,
        0.20,
        barrier_type="up_and_out",
        rebate=2.0,
        paths=2_000,
        steps=8,
        seed=3,
    )
    assert result.price == pytest.approx(2.0 * math.exp(-0.03 * 0.5))
    assert result.knock_probability == 1.0


def test_barrier_input_validation() -> None:
    with pytest.raises(ValueError, match="paths"):
        price_barrier_option(100, 100, 120, 1, 0.04, 0.2, paths=1)
    with pytest.raises(ValueError, match="positive"):
        price_barrier_option(0, 100, 120, 1, 0.04, 0.2)
