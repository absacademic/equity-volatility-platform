# Point-in-time event tables, event windows, and pre-event feature snapshots

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl


def read_table(path: str | Path) -> pl.DataFrame:
    selected = Path(path)
    if selected.suffix.lower() == ".parquet":
        return pl.read_parquet(selected)
    if selected.suffix.lower() == ".csv":
        return pl.read_csv(selected, try_parse_dates=True)
    raise ValueError(f"unsupported table format: {selected.suffix}")


def _next_weekday(value: date) -> date:
    current = value + timedelta(days=1)
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def classify_market_session(
    event_timestamp: datetime,
    *,
    timezone: str = "America/New_York",
) -> str:
    local = event_timestamp.astimezone(ZoneInfo(timezone))
    if local.weekday() >= 5:
        return "market_closed"
    clock = local.timetz().replace(tzinfo=None)
    if clock < time(9, 30):
        return "pre_market"
    if clock <= time(16, 0):
        return "regular_session"
    return "after_hours"


def normalize_point_in_time_events(
    events: pl.DataFrame,
    *,
    timezone: str = "America/New_York",
) -> pl.DataFrame:
    # Add local dates, session labels, reaction dates, and look-ahead checks

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in events.iter_rows(named=True):
        event_id = str(raw.get("event_id") or "").strip()
        timestamp = raw.get("event_timestamp")
        if not event_id or not isinstance(timestamp, datetime) or event_id in seen:
            continue
        seen.add(event_id)
        local = timestamp.astimezone(ZoneInfo(timezone))
        session = classify_market_session(timestamp, timezone=timezone)
        event_date = local.date()
        reaction_date = (
            _next_weekday(event_date) if session in {"after_hours", "market_closed"} else event_date
        )
        known = raw.get("known_timestamp")
        valid = known is None or (isinstance(known, datetime) and known <= timestamp)
        rows.append(
            {
                "event_id": event_id,
                "event_type": str(raw.get("event_type") or "other").lower(),
                "event_timestamp": timestamp,
                "known_timestamp": known,
                "market_session": session,
                "event_date": event_date,
                "reaction_date": reaction_date,
                "title": raw.get("title"),
                "symbols": raw.get("symbols"),
                "source": raw.get("source"),
                "expected": bool(raw.get("expected", True)),
                "point_in_time_valid": valid,
            }
        )
    return pl.DataFrame(rows).sort("event_timestamp") if rows else pl.DataFrame()


def build_event_windows(
    events: pl.DataFrame,
    trading_dates: Iterable[date],
    *,
    pre_days: int = 20,
    post_days: int = 5,
) -> pl.DataFrame:
    # Create one row per event and trading-day offset

    dates = sorted(set(trading_dates))
    rows: list[dict[str, Any]] = []
    for event in events.iter_rows(named=True):
        if not event.get("point_in_time_valid", False):
            continue
        reaction_date = event["reaction_date"]
        index = bisect_left(dates, reaction_date)
        if index >= len(dates):
            continue
        complete = index >= pre_days and index + post_days < len(dates)
        for offset in range(-pre_days, post_days + 1):
            position = index + offset
            if not 0 <= position < len(dates):
                continue
            rows.append(
                {
                    "event_id": event["event_id"],
                    "event_type": event["event_type"],
                    "event_timestamp": event["event_timestamp"],
                    "market_session": event["market_session"],
                    "event_date": event["event_date"],
                    "reaction_date": dates[index],
                    "window_date": dates[position],
                    "trading_day_offset": offset,
                    "window_phase": (
                        "pre_event" if offset < 0 else "event" if offset == 0 else "post_event"
                    ),
                    "window_complete": complete,
                    "source": event.get("source"),
                }
            )
    return (
        pl.DataFrame(rows).sort(["event_timestamp", "trading_day_offset"])
        if rows
        else pl.DataFrame()
    )


def _symbol_matches(value: object, symbol: str) -> bool:
    if value is None:
        return True
    symbols = {
        part.strip().upper() for part in str(value).replace(";", ",").split(",") if part.strip()
    }
    return not symbols or symbol.upper() in symbols or bool(symbols & {"ALL", "MARKET"})


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _surface_dislocation(row: dict[str, Any]) -> float | None:
    candidates = (
        row.get("atm_implied_volatility_rolling_zscore"),
        row.get("downside_skew_25_rolling_zscore"),
        row.get("surface_residual_rmse_rolling_zscore"),
    )
    values = [value for value in (_finite(item) for item in candidates) if value is not None]
    if not values:
        return None
    return float(math.sqrt(sum(value * value for value in values) / len(values)))


def select_pre_event_surface_features(
    events: pl.DataFrame,
    daily_features: pl.DataFrame,
    *,
    symbol: str,
    target_dte_days: int = 30,
    annualization_days: int = 252,
) -> pl.DataFrame:
    # Select the latest valid surface timestamp strictly before each event

    feature_rows = list(daily_features.iter_rows(named=True))
    output: list[dict[str, Any]] = []
    for event in events.iter_rows(named=True):
        if not event.get("point_in_time_valid", False) or not _symbol_matches(
            event.get("symbols"), symbol
        ):
            continue
        timestamp = event["event_timestamp"]
        candidates = [
            row
            for row in feature_rows
            if str(row.get("symbol", "")).upper() == symbol.upper()
            and row.get("quote_timestamp") is not None
            and row["quote_timestamp"] < timestamp
            and row.get("chain_valid", True) is not False
        ]
        if not candidates:
            continue
        latest_timestamp = max(row["quote_timestamp"] for row in candidates)
        latest = [row for row in candidates if row["quote_timestamp"] == latest_timestamp]
        selected = min(
            latest,
            key=lambda row: abs(
                (_finite(row.get("time_to_expiry")) or target_dte_days / 365.0) * 365.0
                - target_dte_days
            ),
        )
        atm = _finite(selected.get("atm_implied_volatility"))
        expected_move = (
            atm * math.sqrt(2.0 / math.pi) / math.sqrt(annualization_days)
            if atm is not None
            else None
        )
        event_row = dict(event)
        event_row.update(
            {
                "symbol": symbol.upper(),
                "pre_event_quote_date": selected.get("quote_date"),
                "pre_event_quote_timestamp": selected.get("quote_timestamp"),
                "feature_expiration": selected.get("expiration"),
                "feature_time_to_expiry": _finite(selected.get("time_to_expiry")),
                "snapshot_lag_hours": (timestamp - selected["quote_timestamp"]).total_seconds()
                / 3600.0,
                "atm_volatility": atm,
                "skew": _finite(selected.get("downside_skew_25")),
                "term_structure": _finite(selected.get("atm_term_structure_slope")),
                "expected_move": expected_move,
                "volume_change": _finite(selected.get("total_option_volume_change")),
                "open_interest_change": _finite(selected.get("total_open_interest_change")),
                "iv_percentile": _finite(
                    selected.get("atm_implied_volatility_expanding_percentile")
                ),
                "surface_dislocation": _surface_dislocation(selected),
                "iv_bid_ask_width": _finite(selected.get("iv_bid_ask_width")),
                "surface_residual_rmse": _finite(selected.get("surface_residual_rmse")),
                "realized_volatility_20d": _finite(selected.get("realized_volatility_20d")),
                "vrp_variance_20d": _finite(selected.get("vrp_variance_20d")),
            }
        )
        output.append(event_row)
    return pl.DataFrame(output).sort("event_timestamp") if output else pl.DataFrame()
