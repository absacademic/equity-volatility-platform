from datetime import UTC, date, datetime

import pytest

from vol_platform.surface.expiration import exact_time_to_expiry
from vol_platform.surface.rates import discount_factor, interpolate_rate
from vol_platform.surface.synthetic import synthetic_rate_curve


def test_exact_time_to_expiry_uses_clock_time() -> None:
    quote = datetime(2026, 7, 1, 20, 0, tzinfo=UTC)
    maturity = exact_time_to_expiry(quote, date(2026, 7, 2))
    assert maturity == pytest.approx(24.0 / (24.0 * 365.0))


def test_rate_interpolation_and_discount_factor() -> None:
    curve = synthetic_rate_curve()
    result = interpolate_rate(
        curve,
        quote_date=date(2026, 7, 1),
        expiration=date(2026, 8, 15),
        default_rate=0.03,
    )
    assert result.rate == pytest.approx(0.041)
    assert not result.used_default
    assert discount_factor(0.041, 0.5) == pytest.approx(0.9797086965)
