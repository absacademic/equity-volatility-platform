from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def create_strategy_plots(
    trades: pl.DataFrame,
    attribution: pl.DataFrame,
    sensitivity: pl.DataFrame,
    output_dir: Path,
    *,
    portfolio_capital: float = 1_000_000.0,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if trades.is_empty():
        return paths

    ordered = trades.sort("event_timestamp")
    net_pnl = np.asarray(ordered["net_pnl"].to_list(), dtype=float)
    midpoint_pnl = np.asarray(ordered["midpoint_upper_bound_pnl"].to_list(), dtype=float)
    net_wealth = np.cumsum(net_pnl) / portfolio_capital
    midpoint_wealth = np.cumsum(midpoint_pnl) / portfolio_capital

    figure = Figure()
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(ordered["event_timestamp"].to_list(), net_wealth, label="Executable net")
    axis.plot(
        ordered["event_timestamp"].to_list(),
        midpoint_wealth,
        linestyle="--",
        label="Midpoint upper bound",
    )
    axis.axhline(0.0)
    axis.set_xlabel("Event date")
    axis.set_ylabel("Cumulative return")
    axis.set_title("Exact contract-level strategy results")
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    path = output_dir / "cumulative_strategy_return.png"
    figure.savefig(path, dpi=160)
    paths.append(path)

    wealth = 1.0 + net_wealth
    peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    drawdown = wealth / np.maximum(peaks, 1.0e-12) - 1.0
    figure = Figure()
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(ordered["event_timestamp"].to_list(), drawdown)
    axis.axhline(0.0)
    axis.set_xlabel("Event date")
    axis.set_ylabel("Drawdown")
    axis.set_title("Strategy drawdown")
    figure.autofmt_xdate()
    figure.tight_layout()
    path = output_dir / "strategy_drawdown.png"
    figure.savefig(path, dpi=160)
    paths.append(path)

    if not attribution.is_empty():
        names = [
            "delta_pnl",
            "gamma_pnl",
            "vega_pnl",
            "theta_pnl",
            "option_residual_pnl",
            "hedge_pnl",
            "transaction_cost_pnl",
            "financing_pnl",
        ]
        totals = [float(attribution[name].sum()) for name in names]
        figure = Figure()
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        positions = np.arange(len(names))
        axis.bar(positions, totals)
        axis.set_xticks(positions, [name.replace("_pnl", "") for name in names], rotation=35)
        axis.axhline(0.0)
        axis.set_ylabel("Aggregate P&L")
        axis.set_title("P&L and Greek attribution")
        figure.tight_layout()
        path = output_dir / "pnl_attribution.png"
        figure.savefig(path, dpi=160)
        paths.append(path)

    if not sensitivity.is_empty() and "sharpe_ratio" in sensitivity.columns:
        finite = sensitivity.filter(pl.col("sharpe_ratio").is_not_null())
        if not finite.is_empty():
            figure = Figure()
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            axis.scatter(
                finite["scenario_id"].to_list(),
                finite["sharpe_ratio"].to_list(),
            )
            axis.axhline(0.0)
            axis.set_xlabel("Sensitivity scenario")
            axis.set_ylabel("Sharpe ratio")
            axis.set_title("Strategy sensitivity grid")
            figure.tight_layout()
            path = output_dir / "sensitivity_sharpe.png"
            figure.savefig(path, dpi=160)
            paths.append(path)
    return paths
