from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from vol_platform.event_study.history import build_event_history, detect_large_market_moves
from vol_platform.event_study.nonlinear import run_gated_polynomial_model


def test_detect_large_market_moves() -> None:
    underlying = pl.DataFrame(
        [
            {"timestamp": datetime(2026, 1, 2, tzinfo=UTC), "symbol": "SPY", "last": 100.0},
            {"timestamp": datetime(2026, 1, 5, tzinfo=UTC), "symbol": "SPY", "last": 104.0},
            {"timestamp": datetime(2026, 1, 6, tzinfo=UTC), "symbol": "SPY", "last": 103.0},
        ]
    )
    events = detect_large_market_moves(underlying, threshold=0.03, symbols=("SPY",))
    assert events.height == 1
    assert events["event_type"][0] == "large_market_move"
    assert not events["expected"][0]


def test_build_combined_event_history(tmp_path: Path) -> None:
    macro = tmp_path / "macro.csv"
    macro.write_text(
        "event_id,event_type,event_timestamp,known_timestamp,title,symbols,source,expected\n"
        "cpi-1,cpi,2026-01-13T13:30:00+00:00,2025-12-10T13:30:00+00:00,"
        "CPI,SPY,calendar,true\n",
        encoding="utf-8",
    )
    earnings = tmp_path / "earnings.csv"
    earnings.write_text(
        "event_id,event_type,event_timestamp,known_timestamp,title,symbols,source,expected\n"
        "earnings-1,earnings,2026-01-15T21:15:00+00:00,"
        "2026-01-02T14:00:00+00:00,Earnings,AAPL,calendar,true\n",
        encoding="utf-8",
    )
    underlying = tmp_path / "underlying.csv"
    underlying.write_text(
        "timestamp,symbol,bid,ask,last,volume,currency\n"
        "2026-01-12T21:00:00+00:00,SPY,99.99,100.01,100,1000,USD\n"
        "2026-01-13T21:00:00+00:00,SPY,103.99,104.01,104,1000,USD\n",
        encoding="utf-8",
    )

    events = build_event_history(
        macro_events_file=macro,
        earnings_events_file=earnings,
        underlying_file=underlying,
        market_symbols=("SPY",),
        large_move_threshold=0.03,
    )
    assert set(events["event_type"].to_list()) == {
        "cpi",
        "earnings",
        "large_market_move",
    }
    assert events["event_id"].n_unique() == events.height


def test_nonlinear_model_is_gated_by_baseline_stability() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    dataset = pl.DataFrame(
        [
            {
                "event_id": f"event-{index}",
                "event_timestamp": start + timedelta(days=30 * index),
                "period": "train" if index < 18 else "validation" if index < 24 else "test",
                "x": float(index),
                "expected_minus_realized_move": 0.001 * index,
            }
            for index in range(30)
        ]
    )
    performance = pl.DataFrame(
        [
            {
                "model": "linear",
                "period": "test",
                "directional_accuracy": 0.75,
            }
        ]
    )
    stability = pl.DataFrame(
        [
            {"model": "linear", "feature": "x", "sign_stable": True},
            {"model": "linear", "feature": "intercept", "sign_stable": True},
        ]
    )
    walk = pl.DataFrame(
        [
            {
                "period": "test",
                "linear_directional_accuracy": 0.70,
            }
        ]
    )
    output = run_gated_polynomial_model(
        dataset,
        performance,
        stability,
        walk,
        features=("x",),
    )
    assert output.status["status"][0] == "run"
    assert output.predictions["nonlinear_prediction"].null_count() == 0

    skipped = run_gated_polynomial_model(
        dataset,
        performance.with_columns(pl.lit(0.40).alias("directional_accuracy")),
        stability,
        walk,
        features=("x",),
    )
    assert skipped.status["status"][0] == "skipped"
