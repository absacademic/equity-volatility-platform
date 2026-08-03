from __future__ import annotations

import math
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from vol_platform.pricing.black_scholes import price as black_scholes_price
from vol_platform.types import OptionType


def _business_dates(start: date, end: date) -> list[date]:
    values: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            values.append(current)
        current += timedelta(days=1)
    return values


def _local_timestamp(value: date, clock: time) -> datetime:
    local = datetime.combine(value, clock, tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(UTC)


def _next_business_date(dates: list[date], index: int) -> date:
    return dates[min(index + 1, len(dates) - 1)]


def _strike(spot: float, symbol: str) -> float:
    increment = 5.0 if symbol in {"AAPL", "SPY"} else 1.0
    return round(spot / increment) * increment


def synthetic_week6_inputs() -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    """Create a deterministic multi-asset 2018-2025 research sample."""

    dates = _business_dates(date(2018, 1, 2), date(2025, 12, 31))
    assets = {
        "AAPL": {"start": 42.0, "type": "equity", "base_iv": 0.30, "dividend": 0.006},
        "SPY": {"start": 270.0, "type": "etf", "base_iv": 0.19, "dividend": 0.013},
        "XSP": {"start": 270.0, "type": "index", "base_iv": 0.18, "dividend": 0.013},
    }

    events: list[dict[str, Any]] = []
    signal_specs: list[dict[str, Any]] = []
    reaction_moves: dict[tuple[str, date], float] = {}

    macro_indices = list(range(70, len(dates) - 10, 42))
    for number, index in enumerate(macro_indices):
        event_date = dates[index]
        event_type = "cpi" if number % 2 == 0 else "fomc"
        event_clock = time(8, 30) if event_type == "cpi" else time(14, 0)
        event_timestamp = _local_timestamp(event_date, event_clock)
        event_id = f"{event_type}-{event_date.isoformat()}"
        events.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "event_timestamp": event_timestamp,
                "known_timestamp": event_timestamp - timedelta(days=30),
                "title": "US CPI release" if event_type == "cpi" else "FOMC decision",
                "symbols": "SPY,XSP",
                "source": "synthetic_week6_long_history",
                "expected": True,
            }
        )
        for asset_number, symbol in enumerate(("SPY", "XSP")):
            sign = 1.0 if (number + asset_number) % 3 != 0 else -1.0
            prediction = sign * (0.0010 + 0.00025 * math.sin(number / 3.0))
            direction = -1.0 if prediction > 0.0 else 1.0
            base_iv = float(assets[symbol]["base_iv"])
            expected_move = base_iv * math.sqrt(2.0 / math.pi) / math.sqrt(252.0)
            realized = expected_move * (0.58 if prediction > 0.0 else 1.55)
            signed_move = realized * (-1.0 if (number + asset_number) % 2 else 1.0)
            reaction_moves[(symbol, event_date)] = signed_move
            signal_specs.append(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "event_timestamp": event_timestamp,
                    "reaction_date": event_date,
                    "symbol": symbol,
                    "symbols": symbol,
                    "underlying_type": assets[symbol]["type"],
                    "linear_prediction": prediction,
                    "intended_position": direction,
                }
            )

    earnings_indices = list(range(90, len(dates) - 10, 63))
    for number, index in enumerate(earnings_indices):
        event_date = dates[index]
        reaction_date = _next_business_date(dates, index)
        event_timestamp = _local_timestamp(event_date, time(16, 15))
        event_id = f"earnings-aapl-{event_date.isoformat()}"
        events.append(
            {
                "event_id": event_id,
                "event_type": "earnings",
                "event_timestamp": event_timestamp,
                "known_timestamp": event_timestamp - timedelta(days=14),
                "title": "AAPL earnings release",
                "symbols": "AAPL",
                "source": "synthetic_week6_long_history",
                "expected": True,
            }
        )
        prediction = (1.0 if number % 3 else -1.0) * (0.0015 + 0.00035 * math.cos(number / 4.0))
        expected_move = (
            float(assets["AAPL"]["base_iv"]) * math.sqrt(2.0 / math.pi) / math.sqrt(252.0)
        )
        realized = expected_move * (0.60 if prediction > 0.0 else 1.65)
        reaction_moves[("AAPL", reaction_date)] = realized * (-1.0 if number % 2 else 1.0)
        signal_specs.append(
            {
                "event_id": event_id,
                "event_type": "earnings",
                "event_timestamp": event_timestamp,
                "reaction_date": reaction_date,
                "symbol": "AAPL",
                "symbols": "AAPL",
                "underlying_type": "equity",
                "linear_prediction": prediction,
                "intended_position": -1.0 if prediction > 0.0 else 1.0,
            }
        )

    prices: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    price_lookup: dict[tuple[str, date], float] = {}
    feature_lookup: dict[tuple[str, date], dict[str, Any]] = {}
    for asset_index, (symbol, metadata) in enumerate(assets.items()):
        current_price = float(metadata["start"])
        base_iv = float(metadata["base_iv"])
        for index, current_date in enumerate(dates):
            ordinary_return = 0.00022 + 0.0032 * math.sin(
                (index + 7 * asset_index) / 17.0
            ) * math.cos((index + 3 * asset_index) / 41.0)
            daily_return = reaction_moves.get((symbol, current_date), ordinary_return)
            current_price *= math.exp(daily_return)
            price_lookup[(symbol, current_date)] = current_price
            prices.append(
                {
                    "timestamp": _local_timestamp(current_date, time(16, 0)),
                    "symbol": symbol,
                    "bid": current_price * 0.99995,
                    "ask": current_price * 1.00005,
                    "last": current_price,
                    "volume": 10_000_000,
                    "currency": "USD",
                }
            )
            cycle = math.sin((index + asset_index * 11) / 45.0)
            shorter = math.cos((index + asset_index * 5) / 19.0)
            event_move = reaction_moves.get((symbol, current_date))
            event_iv_change = 0.0
            if event_move is not None:
                related = [
                    item
                    for item in signal_specs
                    if item["symbol"] == symbol and item["reaction_date"] == current_date
                ]
                prediction = float(related[0]["linear_prediction"]) if related else 0.0
                event_iv_change = -0.045 if prediction > 0.0 else 0.025
            atm = max(base_iv + 0.025 * cycle + 0.010 * shorter + event_iv_change, 0.08)
            skew = 0.032 + 0.009 * math.sin((index + asset_index) / 23.0)
            percentile = min(max(0.5 + 0.45 * cycle, 0.01), 0.99)
            atm_z = 0.9 * cycle
            skew_z = 0.75 * math.cos(index / 27.0)
            residual_z = 0.55 * math.sin(index / 13.0)
            feature = {
                "quote_date": current_date,
                "quote_timestamp": _local_timestamp(current_date, time(16, 0)),
                "symbol": symbol,
                "expiration": current_date + timedelta(days=30),
                "time_to_expiry": 30.0 / 365.0,
                "forward": current_price,
                "atm_implied_volatility": atm,
                "downside_skew_25": skew,
                "risk_reversal_25": -2.0 * skew,
                "butterfly_25": 0.006 + 0.001 * shorter,
                "wing_curvature_10_25": 0.004,
                "atm_term_structure_slope": -0.04 + 0.07 * shorter,
                "skew_term_structure_slope": 0.018 * cycle,
                "iv_bid_ask_width": 0.018 + 0.005 * abs(shorter),
                "surface_residual_rmse": 0.0015 + 0.0005 * abs(residual_z),
                "realized_volatility_20d": max(
                    base_iv - 0.035 + 0.015 * math.sin(index / 31.0),
                    0.05,
                ),
                "vrp_variance_20d": atm**2
                - max(base_iv - 0.035 + 0.015 * math.sin(index / 31.0), 0.05) ** 2,
                "total_option_volume": 800_000.0 * (1.0 + 0.20 * math.sin(index / 9.0)),
                "total_open_interest": 3_000_000.0 * (1.0 + 0.10 * math.cos(index / 25.0)),
                "total_option_volume_change": 35_000.0 * math.sin(index / 7.0),
                "total_open_interest_change": 15_000.0 * math.cos(index / 12.0),
                "atm_implied_volatility_expanding_percentile": percentile,
                "atm_implied_volatility_rolling_zscore": atm_z,
                "downside_skew_25_rolling_zscore": skew_z,
                "surface_residual_rmse_rolling_zscore": residual_z,
                "material_arbitrage_violation_count": 0,
                "standardized_points_complete": True,
                "chain_valid": True,
            }
            feature_rows.append(feature)
            feature_lookup[(symbol, current_date)] = feature

    large_move_number = 0
    for symbol in assets:
        prior: float | None = None
        for current_date in dates:
            current = price_lookup[(symbol, current_date)]
            if prior is not None:
                move = current / prior - 1.0
                if abs(move) >= 0.022:
                    large_move_number += 1
                    timestamp = _local_timestamp(current_date, time(16, 0))
                    events.append(
                        {
                            "event_id": (
                                f"large-move-{symbol.lower()}-"
                                f"{current_date.isoformat()}-{large_move_number}"
                            ),
                            "event_type": "large_market_move",
                            "event_timestamp": timestamp,
                            "known_timestamp": timestamp,
                            "title": f"{symbol} daily move {move:.2%}",
                            "symbols": symbol,
                            "source": "synthetic_derived_returns",
                            "expected": False,
                        }
                    )
            prior = current

    sorted_signals = sorted(signal_specs, key=lambda row: (row["event_timestamp"], row["symbol"]))
    count = len(sorted_signals)
    train_end = int(count * 0.60)
    validation_end = int(count * 0.80)
    for index, row in enumerate(sorted_signals):
        if index < train_end:
            row["period"] = "train"
        elif index < validation_end:
            row["period"] = "validation"
        else:
            row["period"] = "test"
        feature = feature_lookup[(row["symbol"], dates[dates.index(row["reaction_date"]) - 1])]
        row["atm_volatility"] = feature["atm_implied_volatility"]
        row["iv_percentile"] = feature["atm_implied_volatility_expanding_percentile"]
        row["surface_dislocation"] = math.sqrt(
            (
                float(feature["atm_implied_volatility_rolling_zscore"]) ** 2
                + float(feature["downside_skew_25_rolling_zscore"]) ** 2
                + float(feature["surface_residual_rmse_rolling_zscore"]) ** 2
            )
            / 3.0
        )
        row["volatility_regime"] = "high_vol" if float(row["iv_percentile"]) >= 0.70 else "normal"

    option_rows: list[dict[str, Any]] = []
    for signal in sorted_signals:
        symbol = str(signal["symbol"])
        reaction_date = signal["reaction_date"]
        reaction_index = dates.index(reaction_date)
        entry_index = reaction_index - 1
        expiration = dates[entry_index] + timedelta(days=35)
        entry_spot = price_lookup[(symbol, dates[entry_index])]
        strike = _strike(entry_spot, symbol)
        for index in range(entry_index, min(reaction_index + 6, len(dates))):
            current_date = dates[index]
            spot = price_lookup[(symbol, current_date)]
            volatility = float(feature_lookup[(symbol, current_date)]["atm_implied_volatility"])
            tte = max((expiration - current_date).days / 365.0, 1.0 / 365.0)
            for option_type in (OptionType.CALL, OptionType.PUT):
                theoretical = black_scholes_price(
                    spot,
                    strike,
                    tte,
                    0.04,
                    volatility,
                    option_type,
                    float(assets[symbol]["dividend"]),
                )
                minimum_spread = 0.04 if symbol != "AAPL" else 0.03
                spread = max(minimum_spread, 0.07 * max(theoretical, 0.10))
                bid = max(theoretical - 0.5 * spread, 0.01)
                ask = max(theoretical + 0.5 * spread, bid + 0.01)
                contract_symbol = (
                    f"{symbol}-{expiration.isoformat()}-{strike:.2f}-{option_type.value[0].upper()}"
                )
                option_rows.append(
                    {
                        "quote_timestamp": _local_timestamp(current_date, time(16, 0)),
                        "symbol": contract_symbol,
                        "underlying_symbol": symbol,
                        "expiration": expiration,
                        "strike": strike,
                        "option_type": option_type.value,
                        "bid": bid,
                        "ask": ask,
                        "last": theoretical,
                        "bid_size": 100,
                        "ask_size": 100,
                        "volume": 2500,
                        "open_interest": 20_000,
                        "exchange": "SYNTHETIC",
                        "currency": "USD",
                        "multiplier": 100,
                        "implied_volatility": volatility,
                        "underlying_price": spot,
                        "source": "synthetic_week6",
                    }
                )

    return (
        pl.DataFrame(events).sort("event_timestamp"),
        pl.DataFrame(prices).sort(["symbol", "timestamp"]),
        pl.DataFrame(feature_rows).sort(["symbol", "quote_timestamp"]),
        pl.DataFrame(option_rows).sort(["underlying_symbol", "quote_timestamp", "symbol"]),
        pl.DataFrame(sorted_signals).sort(["event_timestamp", "symbol"]),
    )


def write_synthetic_week6_inputs(output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events, underlying, features, options, signals = synthetic_week6_inputs()
    paths = {
        "events": output / "synthetic-week6-events.csv",
        "underlying": output / "synthetic-week6-underlying.csv",
        "surface_features": output / "synthetic-week6-surface-features.csv",
        "option_quotes": output / "synthetic-week6-option-quotes.csv",
        "signals": output / "synthetic-week6-signals.csv",
    }
    events.write_csv(paths["events"])
    underlying.write_csv(paths["underlying"])
    features.write_csv(paths["surface_features"])
    options.write_csv(paths["option_quotes"])
    signals.write_csv(paths["signals"])
    return paths
