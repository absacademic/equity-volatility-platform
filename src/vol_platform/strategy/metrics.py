from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import polars as pl


def _summary_row(
    rows: list[dict[str, Any]],
    *,
    label: str,
    value: str,
    annualization_events: float,
    portfolio_capital: float,
) -> dict[str, Any]:
    returns = np.asarray([float(row["net_return"]) for row in rows], dtype=float)
    gross_returns = np.asarray([float(row["gross_return"]) for row in rows], dtype=float)
    costs = np.asarray(
        [
            float(row["transaction_costs"]) / max(float(row["capital_at_risk"]), 1.0e-12)
            for row in rows
        ],
        dtype=float,
    )
    net_pnl = np.asarray([float(row["net_pnl"]) for row in rows], dtype=float)
    turnover = np.asarray([float(row["turnover_notional"]) for row in rows], dtype=float)
    compounded_trade_wealth = np.cumprod(1.0 + returns)
    portfolio_equity = portfolio_capital + np.cumsum(net_pnl)
    running_peak = np.maximum.accumulate(np.concatenate(([portfolio_capital], portfolio_equity)))[
        1:
    ]
    drawdown = portfolio_equity / np.maximum(running_peak, 1.0e-12) - 1.0
    standard_deviation = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    annualized_return = float(np.mean(returns) * annualization_events)
    annualized_volatility = standard_deviation * math.sqrt(annualization_events)
    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0.0 else None
    cutoff = float(np.quantile(returns, 0.05))
    tail = returns[returns <= cutoff]
    expected_shortfall = float(np.mean(tail)) if len(tail) else cutoff
    midpoint_returns = np.asarray(
        [float(row["midpoint_upper_bound_return"]) for row in rows], dtype=float
    )
    return {
        "group": label,
        "value": value,
        "trade_count": len(rows),
        "total_net_pnl": float(np.sum(net_pnl)),
        "total_return": float(np.sum(net_pnl) / portfolio_capital),
        "compounded_trade_return": float(compounded_trade_wealth[-1] - 1.0),
        "mean_net_return": float(np.mean(returns)),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": float(np.min(drawdown)),
        "win_rate": float(np.mean(returns > 0.0)),
        "turnover_to_portfolio_capital": float(np.sum(turnover) / portfolio_capital),
        "mean_cost_drag": float(np.mean(costs)),
        "cumulative_cost_drag": float(np.sum(costs)),
        "tail_loss_5pct": cutoff,
        "expected_shortfall_5pct": expected_shortfall,
        "mean_gross_return": float(np.mean(gross_returns)),
        "mean_midpoint_upper_bound_return": float(np.mean(midpoint_returns)),
        "midpoint_optimism": float(np.mean(midpoint_returns - returns)),
    }


def summarize_trades(
    trades: pl.DataFrame,
    *,
    annualization_events: float = 12.0,
    portfolio_capital: float = 1_000_000.0,
    group_columns: Iterable[str] = (
        "symbol",
        "underlying_type",
        "event_type",
        "period",
        "event_year",
        "volatility_regime",
    ),
) -> pl.DataFrame:
    if trades.is_empty():
        return pl.DataFrame(
            schema={
                "group": pl.String,
                "value": pl.String,
                "trade_count": pl.Int64,
                "total_net_pnl": pl.Float64,
                "total_return": pl.Float64,
            }
        )
    rows = list(trades.iter_rows(named=True))
    output = [
        _summary_row(
            rows,
            label="all",
            value="all",
            annualization_events=annualization_events,
            portfolio_capital=portfolio_capital,
        )
    ]
    for column in group_columns:
        if column not in trades.columns:
            continue
        values = sorted({str(row.get(column)) for row in rows if row.get(column) is not None})
        for value in values:
            selected = [row for row in rows if str(row.get(column)) == value]
            if selected:
                output.append(
                    _summary_row(
                        selected,
                        label=column,
                        value=value,
                        annualization_events=annualization_events,
                        portfolio_capital=portfolio_capital,
                    )
                )
    return pl.DataFrame(output)
