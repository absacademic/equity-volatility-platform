from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from vol_platform.event_study.datasets import (
    build_event_windows,
    normalize_point_in_time_events,
    select_pre_event_surface_features,
)
from vol_platform.event_study.outcomes import calculate_event_outcomes


def test_point_in_time_event_windows_and_outcomes() -> None:
    event_timestamp = datetime(2026, 7, 14, 12, 30, tzinfo=UTC)
    events = pl.DataFrame(
        {
            "event_id": ["cpi-1"],
            "event_type": ["cpi"],
            "event_timestamp": [event_timestamp],
            "known_timestamp": [event_timestamp - timedelta(days=20)],
            "title": ["US CPI release"],
            "symbols": ["SPY"],
            "source": ["test"],
            "expected": [True],
        }
    )
    normalized = normalize_point_in_time_events(events)
    assert normalized["market_session"][0] == "pre_market"
    assert normalized["event_date"][0] == date(2026, 7, 14)
    assert normalized["point_in_time_valid"][0]

    trading_dates = [date(2026, 6, 1) + timedelta(days=index) for index in range(60)]
    trading_dates = [value for value in trading_dates if value.weekday() < 5]
    windows = build_event_windows(normalized, trading_dates, pre_days=20, post_days=5)
    assert windows.height == 26
    assert windows.filter(pl.col("trading_day_offset") == 0)["window_date"][0] == date(2026, 7, 14)

    features = pl.DataFrame(
        {
            "quote_date": [date(2026, 7, 13), date(2026, 7, 14)],
            "quote_timestamp": [
                datetime(2026, 7, 13, 20, 0, tzinfo=UTC),
                datetime(2026, 7, 14, 20, 0, tzinfo=UTC),
            ],
            "symbol": ["SPY", "SPY"],
            "expiration": [date(2026, 8, 14), date(2026, 8, 14)],
            "time_to_expiry": [32 / 365, 31 / 365],
            "atm_implied_volatility": [0.20, 0.17],
            "downside_skew_25": [0.03, 0.025],
            "atm_term_structure_slope": [-0.02, -0.01],
            "total_option_volume_change": [10_000.0, 5_000.0],
            "total_open_interest_change": [4_000.0, 2_000.0],
            "atm_implied_volatility_expanding_percentile": [0.70, 0.50],
            "atm_implied_volatility_rolling_zscore": [1.0, 0.0],
            "downside_skew_25_rolling_zscore": [0.5, 0.0],
            "surface_residual_rmse_rolling_zscore": [0.25, 0.0],
            "iv_bid_ask_width": [0.02, 0.02],
            "surface_residual_rmse": [0.001, 0.001],
            "realized_volatility_20d": [0.16, 0.16],
            "vrp_variance_20d": [0.0144, 0.0033],
            "chain_valid": [True, True],
        }
    )
    selected = select_pre_event_surface_features(normalized, features, symbol="SPY")
    assert selected.height == 1
    assert selected["pre_event_quote_date"][0] == date(2026, 7, 13)
    assert selected["expected_move"][0] == pytest.approx(
        0.20 * (2.0 / 3.141592653589793) ** 0.5 / 252**0.5
    )

    price_rows = []
    price = 600.0
    for value in trading_dates:
        if value == date(2026, 7, 14):
            price *= 1.006
        price_rows.append(
            {
                "timestamp": datetime.combine(value, datetime.min.time(), tzinfo=UTC).replace(
                    hour=20
                ),
                "symbol": "SPY",
                "underlying_price": price,
            }
        )
    outcomes = calculate_event_outcomes(
        selected,
        windows,
        pl.DataFrame(price_rows),
        features,
        fixed_cost_bps=8.0,
    )
    assert outcomes.height == 1
    assert outcomes["absolute_return"][0] == pytest.approx(0.006)
    assert outcomes["post_event_iv_collapse"][0] == pytest.approx(0.03)
    assert outcomes["estimated_transaction_cost"][0] > 0.0
