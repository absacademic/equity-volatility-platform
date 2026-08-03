from datetime import UTC, datetime, timedelta

import polars as pl

from vol_platform.event_study.backtest import run_event_strategy_backtest
from vol_platform.event_study.models import DEFAULT_FEATURES, run_baseline_models


def _dataset() -> pl.DataFrame:
    rows = []
    start = datetime(2023, 1, 1, tzinfo=UTC)
    for index in range(36):
        atm = 0.16 + 0.002 * index
        skew = 0.02 + 0.001 * (index % 5)
        dislocation = 0.2 + 0.05 * (index % 7)
        gap = 0.002 * (atm - 0.18) + 0.0007 * (dislocation - 0.35)
        long_gross = -gap
        rows.append(
            {
                "event_id": f"event-{index}",
                "event_timestamp": start + timedelta(days=30 * index),
                "event_type": "cpi" if index % 2 == 0 else "fomc",
                "atm_volatility": atm,
                "skew": skew,
                "term_structure": -0.05 + 0.002 * index,
                "volume_change": float(index * 1000),
                "open_interest_change": float(index * 400),
                "iv_percentile": index / 35,
                "surface_dislocation": dislocation,
                "expected_minus_realized_move": gap,
                "market_overestimated": gap > 0,
                "long_straddle_gross_return": long_gross,
                "estimated_transaction_cost": 0.0008,
                "realized_move_pnl": 0.008,
                "implied_move_carry": -0.008 - gap,
                "iv_change_pnl": 0.0,
            }
        )
    return pl.DataFrame(rows)


def test_chronological_models_and_backtest() -> None:
    outputs = run_baseline_models(
        _dataset(),
        features=DEFAULT_FEATURES,
        minimum_walk_forward_train=12,
    )
    assert outputs.dataset["period"].to_list()[:3] == ["train", "train", "train"]
    assert {"train", "validation", "test"}.issubset(set(outputs.performance["period"].to_list()))
    assert {"ci_lower_95", "ci_upper_95"}.issubset(outputs.coefficients.columns)
    assert outputs.walk_forward_predictions.height > 0
    assert outputs.stability["sign_stable"].null_count() == 0

    backtest, attribution, summary = run_event_strategy_backtest(outputs.dataset)
    assert backtest.height == 36
    assert attribution.height == 36
    assert "test" in summary["period"].to_list()
    assert "strategy_net_return" in backtest.columns
