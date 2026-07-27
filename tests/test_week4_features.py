from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from vol_platform.surface.arbitrage import DIAGNOSTIC_SCHEMA
from vol_platform.surface.dividends import add_dividend_and_exercise_features
from vol_platform.surface.evaluation import evaluate_fits
from vol_platform.surface.features import build_implied_volatility_dataset
from vol_platform.surface.forward import estimate_forwards
from vol_platform.surface.history import (
    HISTORICAL_COLUMNS,
    add_event_linked_features,
    add_historical_comparisons,
    add_realized_volatility_features,
)
from vol_platform.surface.models import fit_all_smiles
from vol_platform.surface.pipeline import _add_rate_and_expiration_features
from vol_platform.surface.standardized import (
    POINT_NAMES,
    build_daily_volatility_features,
    interpolate_standardized_delta_points,
)
from vol_platform.surface.synthetic import (
    synthetic_clean_chain,
    synthetic_events,
    synthetic_rate_curve,
)


def _week4_surface_inputs() -> tuple[pl.DataFrame, pl.DataFrame, list]:
    enriched = _add_rate_and_expiration_features(
        synthetic_clean_chain(),
        synthetic_rate_curve(),
        timezone="America/New_York",
        day_count_basis=365.0,
        default_rate=0.04,
    )
    enriched = add_dividend_and_exercise_features(enriched, None)
    forwards, _ = estimate_forwards(enriched)
    iv_data = build_implied_volatility_dataset(enriched, forwards)
    fits = fit_all_smiles(iv_data)
    details, _ = evaluate_fits(iv_data, fits)
    return iv_data, details, fits


def test_standardized_points_and_daily_rows_are_complete() -> None:
    iv_data, details, fits = _week4_surface_inputs()
    points = interpolate_standardized_delta_points(iv_data, details, fits)
    diagnostics = pl.DataFrame(schema=DIAGNOSTIC_SCHEMA)
    features = build_daily_volatility_features(points, iv_data, details, diagnostics)

    assert points.height == 15
    assert set(points["point"]) == set(POINT_NAMES)
    assert set(points["delta_convention"]) == {"black76_discounted_forward"}
    assert features.height == 3
    assert features["standardized_points_complete"].all()
    assert features["chain_valid"].all()
    assert features["atm_implied_volatility"].null_count() == 0
    assert features["downside_skew_25"].null_count() == 0


def test_later_known_dividend_is_not_used() -> None:
    enriched = _add_rate_and_expiration_features(
        synthetic_clean_chain().head(1),
        synthetic_rate_curve(),
        timezone="America/New_York",
        day_count_basis=365.0,
        default_rate=0.04,
    )
    dividends = pl.DataFrame(
        {
            "symbol": ["SPY"],
            "ex_date": [date(2026, 7, 17)],
            "amount": [10.0],
            "known_timestamp": [datetime(2026, 7, 2, 12, 0, tzinfo=UTC)],
        }
    )
    adjusted = add_dividend_and_exercise_features(enriched, dividends)

    assert adjusted["dividend_count_to_expiry"][0] == 0
    assert adjusted["dividend_present_value"][0] == 0.0


def test_realized_volatility_uses_only_prior_prices() -> None:
    quote_date = date(2026, 7, 1)
    features = pl.DataFrame(
        {
            "quote_date": [quote_date],
            "symbol": ["SPY"],
            "atm_implied_volatility": [0.22],
        }
    )
    dates = [quote_date - timedelta(days=30 - index) for index in range(31)]
    base_prices = [500.0 + index for index in range(31)]
    history = pl.DataFrame(
        {"quote_date": dates, "symbol": ["SPY"] * 31, "underlying_price": base_prices}
    )
    changed_future = history.with_columns(
        pl.when(pl.col("quote_date") >= quote_date)
        .then(pl.lit(10_000.0))
        .otherwise(pl.col("underlying_price"))
        .alias("underlying_price")
    )

    first = add_realized_volatility_features(features, history, windows=(5, 20))
    second = add_realized_volatility_features(features, changed_future, windows=(5, 20))

    assert first["realized_volatility_20d"][0] == pytest.approx(
        second["realized_volatility_20d"][0]
    )
    assert first["vrp_variance_20d"][0] is not None


def test_event_features_use_known_scheduled_events() -> None:
    features = pl.DataFrame(
        {
            "quote_date": [date(2026, 7, 1)],
            "quote_timestamp": [datetime(2026, 7, 1, 14, 30, tzinfo=UTC)],
            "symbol": ["SPY"],
            "expiration": [date(2026, 8, 21)],
            "time_to_expiry": [0.14],
            "atm_implied_volatility": [0.22],
        }
    )
    late_known = pl.DataFrame(
        {
            "event_id": ["late-known"],
            "event_type": ["macro"],
            "event_timestamp": [datetime(2026, 7, 10, 12, 0, tzinfo=UTC)],
            "known_timestamp": [datetime(2026, 7, 2, 12, 0, tzinfo=UTC)],
            "title": ["Later-added event"],
            "symbols": ["SPY"],
            "source": ["test"],
            "expected": [True],
        }
    )
    events = pl.concat([synthetic_events(), late_known], how="diagonal_relaxed")
    enriched = add_event_linked_features(features, events)

    assert enriched["event_count_to_expiry"][0] == 2
    assert enriched["next_event_id"][0] == "fomc-2026-07"
    assert enriched["event_variance_density"][0] > 0.0


def test_historical_statistics_do_not_change_when_future_rows_are_added() -> None:
    rows = []
    for index in range(3):
        row = {
            "quote_date": date(2026, 7, 1) + timedelta(days=index),
            "symbol": "SPY",
            "expiration": date(2026, 9, 18),
        }
        for column_index, column in enumerate(HISTORICAL_COLUMNS):
            row[column] = 0.10 + 0.01 * index + 0.001 * column_index
        rows.append(row)

    prefix = add_historical_comparisons(pl.DataFrame(rows[:2]), rolling_window=2)
    full = add_historical_comparisons(pl.DataFrame(rows), rolling_window=2).head(2)

    columns = [
        "atm_implied_volatility_change",
        "atm_implied_volatility_rolling_zscore",
        "atm_implied_volatility_expanding_percentile",
    ]
    assert prefix.select(columns).to_dicts() == full.select(columns).to_dicts()
