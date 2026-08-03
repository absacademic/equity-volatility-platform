from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

from vol_platform.pricing.black_scholes import price
from vol_platform.strategy.backtest import run_contract_backtest
from vol_platform.strategy.config import StrategyConfig
from vol_platform.strategy.pipeline import run_strategy_backtest
from vol_platform.strategy.synthetic import write_synthetic_week6_inputs


def _timestamp(value: date) -> datetime:
    return datetime(value.year, value.month, value.day, 20, 0, tzinfo=UTC)


def _small_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    dates = [date(2026, 1, 5) + timedelta(days=index) for index in range(4)]
    spots = [100.0, 100.0, 104.0, 103.0]
    underlying = pl.DataFrame(
        [
            {"timestamp": _timestamp(current), "symbol": "SPY", "last": spot}
            for current, spot in zip(dates, spots, strict=True)
        ]
    )
    expiration = date(2026, 2, 20)
    quotes = []
    for current, spot in zip(dates, spots, strict=True):
        tte = (expiration - current).days / 365.0
        for option_type in ("call", "put"):
            midpoint = price(spot, 100.0, tte, 0.04, 0.22, option_type, 0.01)
            quotes.append(
                {
                    "quote_timestamp": _timestamp(current),
                    "symbol": f"SPY-{expiration}-{option_type}",
                    "underlying_symbol": "SPY",
                    "expiration": expiration,
                    "strike": 100.0,
                    "option_type": option_type,
                    "bid": max(midpoint - 0.05, 0.01),
                    "ask": midpoint + 0.05,
                    "volume": 1000,
                    "open_interest": 5000,
                    "multiplier": 100,
                    "implied_volatility": 0.22,
                }
            )
    signals = pl.DataFrame(
        [
            {
                "event_id": "cpi-test",
                "event_type": "cpi",
                "event_timestamp": datetime(2026, 1, 6, 13, 30, tzinfo=UTC),
                "reaction_date": dates[1],
                "symbol": "SPY",
                "underlying_type": "etf",
                "period": "test",
                "linear_prediction": -0.001,
                "volatility_regime": "normal",
            }
        ]
    )
    return signals, pl.DataFrame(quotes), underlying


def test_exact_contract_pnl_and_hedging() -> None:
    signals, quotes, underlying = _small_frames()
    config = StrategyConfig(
        holding_period_days=1,
        minimum_open_interest=100,
        maximum_relative_spread=0.50,
        prediction_threshold=0.0001,
    )
    result = run_contract_backtest(signals, quotes, underlying, config)

    assert result.trades.height == 1
    trade = result.trades.row(0, named=True)
    assert trade["pnl_method"] == "exact_contract_level_bid_ask_delta_hedged"
    assert trade["midpoint_upper_bound_pnl"] >= trade["net_pnl"]
    assert trade["transaction_costs"] > 0.0
    assert result.attribution.height == 1
    assert result.hedge_log["action"].to_list()[0] == "initial_hedge"


def test_liquidity_filter_records_rejection() -> None:
    signals, quotes, underlying = _small_frames()
    result = run_contract_backtest(
        signals,
        quotes,
        underlying,
        StrategyConfig(minimum_open_interest=100_000),
    )
    assert result.trades.is_empty()
    assert result.rejections["reason"].to_list() == ["no_liquid_atm_call_put_pair"]


def _week6_config(tmp_path: Path) -> Path:
    path = tmp_path / "week6.yml"
    path.write_text(
        f"""project:
  name: week6-test
  timezone: America/New_York
universe:
  symbols: [AAPL, SPY, XSP]
pricing:
  default_model: black_scholes
  default_rate: 0.04
  default_dividend_yield: 0.012
  iv: {{}}
paths:
  raw_data: {tmp_path / "raw"}
  interim_data: {tmp_path / "interim"}
  processed_data: {tmp_path / "processed"}
  reports: {tmp_path / "reports"}
strategy:
  prediction_threshold: 0.0001
  holding_period_days: 1
  hedge_frequency_days: 1
  minimum_open_interest: 100
  maximum_relative_spread: 0.35
  tradable_event_types: [cpi, fomc, earnings]
sensitivity:
  prediction_thresholds: [0.0001]
  holding_period_days: [1]
  hedge_frequency_days: [1]
  option_slippage_fraction_of_spread: [0.10]
  commission_per_contract: [0.65]
  hedge_cost_bps: [1.0]
  minimum_open_interest: [100]
  maximum_relative_spread: [0.35]
  year_ranges: [[2018, 2025]]
  volatility_regimes: [all]
""",
        encoding="utf-8",
    )
    return path


def test_week6_completion_criterion(tmp_path: Path) -> None:
    inputs = write_synthetic_week6_inputs(tmp_path / "inputs")
    result = run_strategy_backtest(
        inputs["signals"],
        inputs["option_quotes"],
        inputs["underlying"],
        config_path=_week6_config(tmp_path),
        output_dir=tmp_path / "output",
    )
    for path in (
        result.trades,
        result.metrics,
        result.attribution,
        result.hedge_log,
        result.rejections,
        result.sensitivity,
        result.report,
        result.database,
    ):
        assert path.exists()

    trades = pl.read_parquet(result.trades)
    metrics = pl.read_parquet(result.metrics)
    assert trades.height >= 50
    assert {"equity", "etf", "index"}.issubset(set(trades["underlying_type"].to_list()))
    assert trades["midpoint_upper_bound_return"].mean() >= trades["net_return"].mean()
    assert "sharpe_ratio" in metrics.columns
    assert len(result.plots) == 4

    connection = duckdb.connect(str(result.database), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == trades.height
        assert connection.execute("SELECT COUNT(*) FROM attribution").fetchone()[0] == trades.height
    finally:
        connection.close()
