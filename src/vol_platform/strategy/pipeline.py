from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from vol_platform.config import load_config
from vol_platform.data.adapters import UnderlyingPriceCSVAdapter
from vol_platform.event_study.datasets import read_table
from vol_platform.strategy.backtest import run_contract_backtest
from vol_platform.strategy.config import StrategyConfig
from vol_platform.strategy.metrics import summarize_trades
from vol_platform.strategy.plots import create_strategy_plots
from vol_platform.strategy.sensitivity import run_sensitivity_grid


@dataclass(frozen=True, slots=True)
class StrategyBacktestResult:
    output_dir: Path
    trades: Path
    metrics: Path
    attribution: Path
    hedge_log: Path
    rejections: Path
    sensitivity: Path
    report: Path
    database: Path
    plots: tuple[Path, ...]


def _settings(config: object, name: str) -> dict[str, Any]:
    value = getattr(config, name, {})
    return value if isinstance(value, dict) else {}


def _write_pair(frame: pl.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    frame.write_parquet(parquet_path)
    frame.write_csv(csv_path)


def _create_database(root: Path, tables: dict[str, Path]) -> Path:
    database = root / "strategy-backtest.duckdb"
    connection = duckdb.connect(str(database))
    try:
        for name, path in tables.items():
            escaped = str(path).replace("'", "''")
            connection.execute(
                f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{escaped}')"
            )
    finally:
        connection.close()
    return database


def _report_markdown(
    config: StrategyConfig,
    trades: pl.DataFrame,
    metrics: pl.DataFrame,
    rejections: pl.DataFrame,
) -> str:
    del trades
    overall = metrics.filter((pl.col("group") == "all") & (pl.col("value") == "all"))
    if overall.is_empty():
        result_text = "No trades passed the configured signal, liquidity, and risk filters."
    else:
        row = overall.row(0, named=True)
        sharpe = row.get("sharpe_ratio")
        sharpe_text = f"{float(sharpe):.2f}" if sharpe is not None else "not defined"
        result_text = (
            f"The backtest executed {int(row['trade_count'])} trades. Total return was "
            f"{float(row['total_return']):.2%}, annualized volatility was "
            f"{float(row['annualized_volatility']):.2%}, Sharpe ratio was {sharpe_text}, "
            f"maximum drawdown was {float(row['maximum_drawdown']):.2%}, and win rate "
            f"was {float(row['win_rate']):.1%}."
        )
    strategy_text = (
        "The implemented strategy is an event-conditioned, delta-hedged at-the-money "
        "straddle. A positive Week 5 prediction means the option-implied move is expected "
        "to exceed the realized move, so the strategy sells the straddle. A negative "
        "prediction buys the straddle. Signals inside the threshold are skipped."
    )
    timing_text = (
        f"Entry occurs {config.entry_days_before_event} trading day(s) before the event. "
        f"Exit occurs {config.holding_period_days} trading day(s) after the reaction date. "
        "The engine selects the liquid call-put pair nearest "
        f"{config.target_dte_days} days to expiration and closest to the underlying price."
    )
    execution_text = (
        "Long option trades execute near the ask and sales execute near the bid. The model "
        "includes option slippage, commissions, underlying hedge costs, and financing. "
        "Midpoint results are reported only as an upper bound. Contracts that fail volume, "
        "open-interest, spread, DTE, quote-completeness, overlap, or capital-limit checks "
        "are rejected and recorded."
    )
    sizing_text = (
        f"Position size is limited to {config.contracts_per_trade} requested contract(s), "
        f"{config.maximum_contracts} maximum contract(s), and "
        f"{config.maximum_capital_fraction_per_trade:.1%} of portfolio capital per trade. "
        f"Delta is hedged initially and every {config.hedge_frequency_days} trading day(s)."
    )
    interpretation = (
        "Synthetic inputs validate the workflow but are not evidence of a profitable "
        "real-market strategy. Real results depend on licensed historical option quotes, "
        "survivorship-safe reference data, timestamp quality, fill assumptions, margin "
        "rules, taxes, and market impact."
    )
    return f"""# Week 6 strategy backtest report

## Strategy

{strategy_text}

{timing_text}

## Execution and risk controls

{execution_text}

{sizing_text}

## Results

{result_text}

Rejected event count: {rejections.height}.

## Interpretation

{interpretation}
"""


def run_strategy_backtest(
    signals_file: str | Path,
    option_quotes_file: str | Path,
    underlying_file: str | Path,
    *,
    config_path: str | Path = "configs/base.yml",
    output_dir: str | Path | None = None,
) -> StrategyBacktestResult:
    config = load_config(config_path)
    strategy_settings = _settings(config, "strategy")
    sensitivity_settings = _settings(config, "sensitivity")
    strategy = StrategyConfig.from_mapping(strategy_settings)
    signals = read_table(signals_file)
    option_quotes = read_table(option_quotes_file)
    underlying = UnderlyingPriceCSVAdapter().read(underlying_file)

    result = run_contract_backtest(signals, option_quotes, underlying, strategy)
    metrics = summarize_trades(
        result.trades,
        annualization_events=strategy.annualization_events,
        portfolio_capital=strategy.portfolio_capital,
    )
    sensitivity = run_sensitivity_grid(
        signals,
        option_quotes,
        underlying,
        strategy,
        sensitivity_settings,
    )

    default_root = Path(config.paths["processed_data"]) / "strategies" / "week6"
    root = Path(output_dir or default_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "trades": root / "strategy-trades.parquet",
        "attribution": root / "pnl-attribution.parquet",
        "hedge_log": root / "hedge-log.parquet",
        "rejections": root / "trade-rejections.parquet",
        "metrics": root / "strategy-metrics.parquet",
        "sensitivity": root / "sensitivity-results.parquet",
    }
    _write_pair(result.trades, paths["trades"], root / "strategy-trades.csv")
    _write_pair(result.attribution, paths["attribution"], root / "pnl-attribution.csv")
    _write_pair(result.hedge_log, paths["hedge_log"], root / "hedge-log.csv")
    _write_pair(result.rejections, paths["rejections"], root / "trade-rejections.csv")
    _write_pair(metrics, paths["metrics"], root / "strategy-metrics.csv")
    _write_pair(sensitivity, paths["sensitivity"], root / "sensitivity-results.csv")

    plots = create_strategy_plots(
        result.trades,
        result.attribution,
        sensitivity,
        root / "plots",
        portfolio_capital=strategy.portfolio_capital,
    )
    database = _create_database(root, paths)
    report_path = root / "strategy-report.md"
    report_path.write_text(
        _report_markdown(strategy, result.trades, metrics, result.rejections),
        encoding="utf-8",
    )
    manifest = {
        "strategy": asdict(strategy),
        "trade_count": result.trades.height,
        "rejection_count": result.rejections.height,
        "pnl_method": "exact_contract_level_bid_ask_delta_hedged",
        "midpoint_treatment": "upper_bound_only",
        "inputs": {
            "signals": str(signals_file),
            "option_quotes": str(option_quotes_file),
            "underlying": str(underlying_file),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "plots": [str(path) for path in plots],
    }
    (root / "strategy-manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    return StrategyBacktestResult(
        output_dir=root,
        trades=paths["trades"],
        metrics=paths["metrics"],
        attribution=paths["attribution"],
        hedge_log=paths["hedge_log"],
        rejections=paths["rejections"],
        sensitivity=paths["sensitivity"],
        report=report_path,
        database=database,
        plots=tuple(plots),
    )
