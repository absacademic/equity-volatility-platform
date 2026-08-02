# Deterministic event-study inputs

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl


def _business_dates(start: date, end: date) -> list[date]:
    dates: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def _local_timestamp(value: date, clock: time) -> datetime:
    local = datetime.combine(value, clock, tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(UTC)


def synthetic_event_study_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    dates = _business_dates(date(2022, 1, 3), date(2026, 6, 30))
    event_indices = list(range(35, len(dates) - 7, 22))
    base_features: list[dict[str, object]] = []
    for index, current in enumerate(dates):
        cycle = math.sin(index / 31.0)
        shorter_cycle = math.cos(index / 13.0)
        atm = 0.185 + 0.020 * cycle + 0.007 * shorter_cycle
        skew = 0.030 + 0.007 * math.sin(index / 17.0)
        residual_z = 0.55 * math.sin(index / 9.0)
        atm_z = 0.85 * cycle
        skew_z = 0.70 * math.cos(index / 19.0)
        dislocation = math.sqrt((atm_z**2 + skew_z**2 + residual_z**2) / 3.0)
        percentile = min(max(0.5 + 0.45 * cycle, 0.01), 0.99)
        volume = 1_000_000.0 * (1.0 + 0.20 * math.sin(index / 8.0))
        open_interest = 4_500_000.0 * (1.0 + 0.08 * math.cos(index / 21.0))
        base_features.append(
            {
                "quote_date": current,
                "quote_timestamp": _local_timestamp(current, time(16, 0)),
                "symbol": "SPY",
                "expiration": current + timedelta(days=30),
                "time_to_expiry": 30.0 / 365.0,
                "forward": 0.0,
                "atm_implied_volatility": atm,
                "downside_skew_25": skew,
                "risk_reversal_25": -2.0 * skew,
                "butterfly_25": 0.006 + 0.001 * shorter_cycle,
                "wing_curvature_10_25": 0.004,
                "atm_term_structure_slope": -0.05 + 0.08 * shorter_cycle,
                "skew_term_structure_slope": 0.02 * cycle,
                "iv_bid_ask_width": 0.018 + 0.004 * abs(shorter_cycle),
                "surface_residual_rmse": 0.0015 + 0.0007 * abs(residual_z),
                "realized_volatility_20d": 0.165 + 0.016 * math.sin(index / 29.0),
                "vrp_variance_20d": atm**2 - (0.165 + 0.016 * math.sin(index / 29.0)) ** 2,
                "total_option_volume": volume,
                "total_open_interest": open_interest,
                "total_option_volume_change": 45_000.0 * math.sin(index / 5.0),
                "total_open_interest_change": 18_000.0 * math.cos(index / 11.0),
                "atm_implied_volatility_expanding_percentile": percentile,
                "atm_implied_volatility_rolling_zscore": atm_z,
                "downside_skew_25_rolling_zscore": skew_z,
                "surface_residual_rmse_rolling_zscore": residual_z,
                "material_arbitrage_violation_count": 0,
                "standardized_points_complete": True,
                "chain_valid": True,
                "synthetic_surface_dislocation": dislocation,
            }
        )

    events: list[dict[str, object]] = []
    event_returns: dict[int, float] = {}
    for event_number, event_index in enumerate(event_indices):
        event_date = dates[event_index]
        event_type = "cpi" if event_number % 2 == 0 else "fomc"
        event_clock = time(8, 30) if event_type == "cpi" else time(14, 0)
        event_timestamp = _local_timestamp(event_date, event_clock)
        pre = base_features[event_index - 1]
        atm = float(pre["atm_implied_volatility"])
        expected_move = atm * math.sqrt(2.0 / math.pi) / math.sqrt(252.0)
        dislocation = float(pre["synthetic_surface_dislocation"])
        percentile = float(pre["atm_implied_volatility_expanding_percentile"])
        predictable_gap = 0.00055 * (dislocation - 0.45) + 0.00035 * (percentile - 0.5)
        noise = 0.00065 * math.sin(event_number * 1.7)
        realized_move = max(expected_move - predictable_gap + noise, 0.0010)
        event_returns[event_index] = realized_move * (-1.0 if event_number % 3 == 0 else 1.0)
        base_features[event_index]["atm_implied_volatility"] = max(atm - 0.018, 0.08)
        events.append(
            {
                "event_id": f"{event_type}-{event_date.isoformat()}",
                "event_type": event_type,
                "event_timestamp": event_timestamp,
                "known_timestamp": event_timestamp - timedelta(days=30),
                "title": "US CPI release" if event_type == "cpi" else "FOMC decision",
                "symbols": "SPY",
                "source": "synthetic_week5",
                "expected": True,
            }
        )

    price = 450.0
    prices: list[dict[str, object]] = []
    for index, current in enumerate(dates):
        normal_return = 0.00015 + 0.0035 * math.sin(index / 7.0) * math.cos(index / 23.0)
        daily_return = event_returns.get(index, normal_return)
        price *= math.exp(daily_return)
        base_features[index]["forward"] = price
        prices.append(
            {
                "timestamp": _local_timestamp(current, time(16, 0)),
                "symbol": "SPY",
                "last": price,
                "currency": "USD",
            }
        )

    return pl.DataFrame(events), pl.DataFrame(prices), pl.DataFrame(base_features)


def write_synthetic_event_study_inputs(output_dir: str | Path) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events, prices, features = synthetic_event_study_inputs()
    event_path = output / "synthetic-week5-events.csv"
    price_path = output / "synthetic-week5-underlying.csv"
    feature_path = output / "synthetic-week5-surface-features.csv"
    events.write_csv(event_path)
    prices.write_csv(price_path)
    features.write_csv(feature_path)
    return event_path, price_path, feature_path
