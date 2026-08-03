from __future__ import annotations

from itertools import product
from typing import Any

import polars as pl

from vol_platform.strategy.backtest import run_contract_backtest
from vol_platform.strategy.config import StrategyConfig
from vol_platform.strategy.metrics import summarize_trades


def _values(raw: dict[str, Any], name: str, default: object) -> list[object]:
    value = raw.get(name, [default])
    return list(value) if isinstance(value, (list, tuple)) else [value]


def run_sensitivity_grid(
    signals: pl.DataFrame,
    option_quotes: pl.DataFrame,
    underlying: pl.DataFrame,
    base_config: StrategyConfig,
    grid: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Evaluate execution and holding assumptions without using future data."""

    raw = grid or {}
    thresholds = _values(raw, "prediction_thresholds", base_config.prediction_threshold)
    holding_periods = _values(raw, "holding_period_days", base_config.holding_period_days)
    hedge_frequencies = _values(raw, "hedge_frequency_days", base_config.hedge_frequency_days)
    slippages = _values(
        raw,
        "option_slippage_fraction_of_spread",
        base_config.option_slippage_fraction_of_spread,
    )
    commissions = _values(
        raw,
        "commission_per_contract",
        base_config.commission_per_contract,
    )
    hedge_costs = _values(raw, "hedge_cost_bps", base_config.hedge_cost_bps)
    minimum_open_interests = _values(
        raw, "minimum_open_interest", base_config.minimum_open_interest
    )
    spread_limits = _values(raw, "maximum_relative_spread", base_config.maximum_relative_spread)
    year_ranges = raw.get("year_ranges", [[None, None]])
    regimes = _values(raw, "volatility_regimes", "all")

    rows: list[dict[str, Any]] = []
    scenarios = product(
        thresholds,
        holding_periods,
        hedge_frequencies,
        slippages,
        commissions,
        hedge_costs,
        minimum_open_interests,
        spread_limits,
        year_ranges,
        regimes,
    )
    for scenario_id, (
        threshold,
        holding,
        hedge_frequency,
        slippage,
        commission,
        hedge_cost,
        minimum_open_interest,
        spread_limit,
        year_range,
        regime,
    ) in enumerate(scenarios, start=1):
        year_start, year_end = year_range if len(year_range) == 2 else (None, None)
        selected = signals
        if year_start is not None and "reaction_date" in selected.columns:
            selected = selected.filter(pl.col("reaction_date").dt.year() >= int(year_start))
        if year_end is not None and "reaction_date" in selected.columns:
            selected = selected.filter(pl.col("reaction_date").dt.year() <= int(year_end))
        if regime != "all" and "volatility_regime" in selected.columns:
            selected = selected.filter(pl.col("volatility_regime") == str(regime))

        scenario = base_config.with_overrides(
            prediction_threshold=float(threshold),
            holding_period_days=int(holding),
            hedge_frequency_days=int(hedge_frequency),
            option_slippage_fraction_of_spread=float(slippage),
            commission_per_contract=float(commission),
            hedge_cost_bps=float(hedge_cost),
            minimum_open_interest=int(minimum_open_interest),
            maximum_relative_spread=float(spread_limit),
        )
        result = run_contract_backtest(selected, option_quotes, underlying, scenario)
        summary = summarize_trades(
            result.trades,
            annualization_events=scenario.annualization_events,
            portfolio_capital=scenario.portfolio_capital,
        )
        metrics = (
            summary.filter(pl.col("group") == "all").row(0, named=True)
            if not summary.is_empty()
            else {
                "trade_count": 0,
                "total_net_pnl": 0.0,
                "total_return": None,
                "annualized_return": None,
                "annualized_volatility": None,
                "sharpe_ratio": None,
                "maximum_drawdown": None,
                "win_rate": None,
                "turnover_to_portfolio_capital": None,
                "mean_cost_drag": None,
                "tail_loss_5pct": None,
                "expected_shortfall_5pct": None,
                "midpoint_optimism": None,
            }
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "prediction_threshold": float(threshold),
                "holding_period_days": int(holding),
                "hedge_frequency_days": int(hedge_frequency),
                "option_slippage_fraction_of_spread": float(slippage),
                "commission_per_contract": float(commission),
                "hedge_cost_bps": float(hedge_cost),
                "minimum_open_interest": int(minimum_open_interest),
                "maximum_relative_spread": float(spread_limit),
                "year_start": year_start,
                "year_end": year_end,
                "volatility_regime": str(regime),
                "rejection_count": result.rejections.height,
                **{key: value for key, value in metrics.items() if key not in {"group", "value"}},
            }
        )
    return pl.DataFrame(rows)
