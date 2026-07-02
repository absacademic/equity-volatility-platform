from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from vol_platform.schemas import (
    DividendRecord,
    ImpliedVolatilityRecord,
    OptionQuote,
    RateCurvePoint,
    UnderlyingPrice,
)

NOW = datetime(2026, 7, 1, 14, 30, tzinfo=UTC)


def test_option_quote_midpoint() -> None:
    quote = OptionQuote(
        timestamp=NOW,
        symbol="SPY260717C00600000",
        underlying_symbol="SPY",
        expiration=date(2026, 7, 17),
        strike=600,
        option_type="call",
        bid=4.0,
        ask=4.2,
    )
    assert quote.mid == pytest.approx(4.1)


def test_option_quote_rejects_crossed_market() -> None:
    with pytest.raises(ValidationError, match="ask"):
        OptionQuote(
            timestamp=NOW,
            symbol="SPY_OPT",
            underlying_symbol="SPY",
            expiration=date(2026, 7, 17),
            strike=600,
            option_type="call",
            bid=4.2,
            ask=4.0,
        )


def test_option_quote_rejects_expired_quote() -> None:
    with pytest.raises(ValidationError, match="expiration"):
        OptionQuote(
            timestamp=NOW,
            symbol="SPY_OPT",
            underlying_symbol="SPY",
            expiration=date(2026, 6, 30),
            strike=600,
            option_type="put",
            bid=1.0,
            ask=1.2,
        )


def test_underlying_reference_price_prefers_midpoint() -> None:
    price = UnderlyingPrice(timestamp=NOW, symbol="SPY", bid=600.0, ask=600.2, last=599.9)
    assert price.reference_price == pytest.approx(600.1)


def test_underlying_requires_at_least_one_price() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        UnderlyingPrice(timestamp=NOW, symbol="SPY")


def test_rate_curve_requires_future_maturity() -> None:
    with pytest.raises(ValidationError, match="maturity"):
        RateCurvePoint(
            as_of_date=date(2026, 7, 1),
            maturity_date=date(2026, 7, 1),
            rate=0.04,
            source="TEST",
        )


def test_dividend_rejects_payment_before_ex_date() -> None:
    with pytest.raises(ValidationError, match="payment_date"):
        DividendRecord(
            symbol="SPY",
            ex_date=date(2026, 9, 18),
            payment_date=date(2026, 9, 17),
            amount=1.75,
        )


def test_successful_implied_vol_record_requires_volatility() -> None:
    with pytest.raises(ValidationError, match="implied_volatility"):
        ImpliedVolatilityRecord(
            quote_timestamp=NOW,
            symbol="SPY_OPT",
            expiration=date(2026, 7, 17),
            strike=600,
            option_type="call",
            model="black_scholes",
            option_price=4.1,
            spot=600.1,
            rate=0.04,
            status="success",
        )
