# Event strategy backtests, transaction costs, and P&L attribution

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl


def run_event_strategy_backtest(
    dataset: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rows: list[dict[str, Any]] = []
    attribution: list[dict[str, Any]] = []
    cumulative = 0.0
    benchmark_cumulative = 0.0
    for event in dataset.sort("event_timestamp").iter_rows(named=True):
        prediction = float(event["linear_prediction"])
        position = -1.0 if prediction > 0.0 else 1.0
        gross = position * float(event["long_straddle_gross_return"])
        cost = float(event["estimated_transaction_cost"])
        net = gross - cost
        benchmark_gross = -float(event["long_straddle_gross_return"])
        benchmark_net = benchmark_gross - cost
        cumulative += net
        benchmark_cumulative += benchmark_net
        row = dict(event)
        row.update(
            {
                "strategy_position": position,
                "strategy_label": "short_straddle" if position < 0 else "long_straddle",
                "strategy_gross_return": gross,
                "strategy_net_return": net,
                "strategy_cumulative_return": cumulative,
                "always_short_net_return": benchmark_net,
                "always_short_cumulative_return": benchmark_cumulative,
            }
        )
        rows.append(row)
        attribution.append(
            {
                "event_id": event["event_id"],
                "event_timestamp": event["event_timestamp"],
                "period": event["period"],
                "strategy_position": position,
                "realized_move_component": position * float(event["realized_move_pnl"]),
                "implied_move_component": position * float(event["implied_move_carry"]),
                "iv_change_component": position * float(event["iv_change_pnl"]),
                "transaction_cost_component": -cost,
                "net_return": net,
            }
        )

    summary_rows = []
    for period in ("train", "validation", "test", "all"):
        selected = rows if period == "all" else [row for row in rows if row["period"] == period]
        if not selected:
            continue
        returns = np.asarray([float(row["strategy_net_return"]) for row in selected])
        standard_deviation = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        summary_rows.append(
            {
                "period": period,
                "observation_count": len(selected),
                "mean_net_return": float(np.mean(returns)),
                "median_net_return": float(np.median(returns)),
                "hit_rate": float(np.mean(returns > 0.0)),
                "cumulative_net_return": float(np.sum(returns)),
                "annualized_sharpe_12_events": (
                    float(np.mean(returns) / standard_deviation * math.sqrt(12.0))
                    if standard_deviation > 0.0
                    else None
                ),
                "mean_transaction_cost": float(
                    np.mean([float(row["estimated_transaction_cost"]) for row in selected])
                ),
            }
        )
    return pl.DataFrame(rows), pl.DataFrame(attribution), pl.DataFrame(summary_rows)
