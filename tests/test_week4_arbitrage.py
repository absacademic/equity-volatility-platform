from datetime import date

import numpy as np
import polars as pl

from vol_platform.surface.arbitrage import (
    DIAGNOSTIC_SCHEMA,
    apply_surface_controls,
    build_arbitrage_diagnostics,
)
from vol_platform.surface.evaluation import SMILE_FIT_DETAIL_SCHEMA
from vol_platform.surface.models import SmileFit


def _invalid_market_chain() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "quote_date": [date(2026, 7, 1)] * 3,
            "underlying_symbol": ["SPY"] * 3,
            "expiration": [date(2026, 8, 21)] * 3,
            "option_type": ["call"] * 3,
            "strike": [90.0, 100.0, 110.0],
            "bid": [11.9, 12.9, 4.9],
            "ask": [12.1, 13.1, 5.1],
            "mid": [12.0, 13.0, 5.0],
            "total_variance": [0.04, 0.04, 0.04],
            "bid_total_variance": [0.039, 0.039, 0.039],
            "ask_total_variance": [0.041, 0.041, 0.041],
            "fit_eligible": [True] * 3,
            "time_to_expiry": [0.25] * 3,
            "forward_moneyness": [-0.1, 0.0, 0.1],
        }
    )


def test_invalid_chain_separates_midpoint_and_executable_violations() -> None:
    diagnostics = build_arbitrage_diagnostics(
        _invalid_market_chain(),
        pl.DataFrame(schema=SMILE_FIT_DETAIL_SCHEMA),
        [],
    )
    violations = diagnostics.filter(pl.col("is_violation"))

    assert set(violations["source"]) == {"midpoint", "executable_bid_ask"}
    assert set(violations["check"]) == {
        "strike_monotonicity",
        "butterfly_convexity",
    }
    assert violations.filter(pl.col("severity") == "material").height == 4


def test_material_negative_fitted_variance_is_rejected_and_recorded() -> None:
    expiration = date(2026, 8, 21)
    iv_data = pl.DataFrame(
        {
            "quote_date": [date(2026, 7, 1)],
            "underlying_symbol": ["SPY"],
            "expiration": [expiration],
            "fit_eligible": [True],
            "time_to_expiry": [0.14],
            "forward": [600.0],
            "interpolated_rate": [0.04],
        }
    )
    fit = SmileFit(
        model="test",
        weighting="equal",
        expiration=expiration,
        success=True,
        message="success",
        x_min=-0.1,
        x_max=0.1,
        parameters={},
        predictor=lambda values: np.full_like(values, -0.01, dtype=float),
    )

    controlled, adjustments = apply_surface_controls(iv_data, [fit])

    assert not controlled[0].success
    assert adjustments.height == 1
    assert adjustments["action"][0] == "surface_rejected"
    assert adjustments["check"][0] == "negative_total_variance"


def test_empty_diagnostic_schema_stays_stable() -> None:
    assert set(DIAGNOSTIC_SCHEMA) >= {
        "source",
        "check",
        "is_violation",
        "severity",
    }


def test_calendar_inconsistency_is_reported_for_market_variance() -> None:
    short_expiration = date(2026, 8, 21)
    long_expiration = date(2026, 9, 18)
    rows = []
    for expiration, time_to_expiry, variance in (
        (short_expiration, 0.14, 0.05),
        (long_expiration, 0.22, 0.03),
    ):
        for strike, moneyness in ((570.0, -0.05), (600.0, 0.0), (630.0, 0.05)):
            rows.append(
                {
                    "quote_date": date(2026, 7, 1),
                    "underlying_symbol": "SPY",
                    "expiration": expiration,
                    "option_type": "call",
                    "strike": strike,
                    "bid": 10.0,
                    "ask": 10.2,
                    "mid": 10.1,
                    "total_variance": variance,
                    "bid_total_variance": variance - 0.001,
                    "ask_total_variance": variance + 0.001,
                    "fit_eligible": True,
                    "time_to_expiry": time_to_expiry,
                    "forward_moneyness": moneyness,
                }
            )
    diagnostics = build_arbitrage_diagnostics(
        pl.DataFrame(rows),
        pl.DataFrame(schema=SMILE_FIT_DETAIL_SCHEMA),
        [],
    )
    calendar = diagnostics.filter(
        (pl.col("check") == "calendar_consistency") & pl.col("is_violation")
    )

    assert set(calendar["source"]) == {"midpoint", "executable_bid_ask"}


def test_unreasonable_fitted_extrapolation_is_rejected() -> None:
    expiration = date(2026, 8, 21)
    iv_data = pl.DataFrame(
        {
            "quote_date": [date(2026, 7, 1)],
            "underlying_symbol": ["SPY"],
            "expiration": [expiration],
            "fit_eligible": [True],
            "time_to_expiry": [0.14],
            "forward": [600.0],
            "interpolated_rate": [0.04],
        }
    )
    fit = SmileFit(
        model="test",
        weighting="equal",
        expiration=expiration,
        success=True,
        message="success",
        x_min=-0.01,
        x_max=0.01,
        parameters={},
        predictor=lambda values: 0.04 + 1_000.0 * np.asarray(values) ** 4,
    )

    controlled, adjustments = apply_surface_controls(iv_data, [fit])

    assert not controlled[0].success
    assert adjustments["check"][0] == "unreasonable_extrapolation"
