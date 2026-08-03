# Combines CPI, FOMC, earnings, and other supplied event calenders
# Derives impactful events from underlying returns
# Preserve point-in-time information; prevent duplicate identifiers
# Export to CSV or Parquet

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from vol_platform.data.adapters import EventCSVAdapter, UnderlyingPriceCSVAdapter


def _event_frame(path: str | Path | None, source: str) -> pl.DataFrame:
    if path is None:
        return pl.DataFrame()
    return (
        EventCSVAdapter()
        .read(path, source=source)
        .select(
            [
                "event_id",
                "event_type",
                "event_timestamp",
                "known_timestamp",
                "title",
                "symbols",
                "source",
                "expected",
            ]
        )
    )


def _identifier(symbol: str, current_date: date, move: float) -> str:
    digest = hashlib.sha1(f"{symbol}|{current_date}|{move:.10f}".encode()).hexdigest()[:10]
    return f"large-move-{symbol.lower()}-{current_date.isoformat()}-{digest}"


def detect_large_market_moves(
    underlying: pl.DataFrame,
    *,
    threshold: float = 0.03,
    symbols: tuple[str, ...] = ("SPY",),
) -> pl.DataFrame:
    if threshold <= 0.0:
        raise ValueError("large-move threshold must be positive")
    selected_symbols = {symbol.upper() for symbol in symbols}
    rows: list[dict[str, Any]] = []
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in underlying.sort("timestamp").iter_rows(named=True):
        symbol = str(row.get("symbol") or "").upper()
        if symbol in selected_symbols:
            by_symbol.setdefault(symbol, []).append(row)
    for symbol, observations in by_symbol.items():
        prior_price: float | None = None
        for row in observations:
            timestamp = row.get("timestamp")
            price = row.get("underlying_price", row.get("last"))
            if not isinstance(timestamp, datetime) or price is None:
                continue
            current_price = float(price)
            if prior_price is not None and prior_price > 0.0:
                move = current_price / prior_price - 1.0
                if abs(move) >= threshold:
                    current_date = timestamp.date()
                    event_timestamp = timestamp
                    rows.append(
                        {
                            "event_id": _identifier(symbol, current_date, move),
                            "event_type": "large_market_move",
                            "event_timestamp": event_timestamp,
                            "known_timestamp": event_timestamp,
                            "title": f"{symbol} daily move {move:.2%}",
                            "symbols": symbol,
                            "source": "derived_underlying_returns",
                            "expected": False,
                            "signed_move": move,
                        }
                    )
            prior_price = current_price
    if rows:
        return pl.DataFrame(rows).sort("event_timestamp")
    return pl.DataFrame(
        schema={
            "event_id": pl.String,
            "event_type": pl.String,
            "event_timestamp": pl.Datetime(time_zone="UTC"),
            "known_timestamp": pl.Datetime(time_zone="UTC"),
            "title": pl.String,
            "symbols": pl.String,
            "source": pl.String,
            "expected": pl.Boolean,
            "signed_move": pl.Float64,
        }
    )


def build_event_history(
    *,
    macro_events_file: str | Path | None = None,
    earnings_events_file: str | Path | None = None,
    underlying_file: str | Path | None = None,
    large_move_threshold: float = 0.03,
    market_symbols: tuple[str, ...] = ("SPY",),
) -> pl.DataFrame:
    # Combine CPI, FOMC, earnings, and derived large-move events

    frames = [
        _event_frame(macro_events_file, "macro_calendar"),
        _event_frame(earnings_events_file, "earnings_calendar"),
    ]
    if underlying_file is not None:
        underlying = UnderlyingPriceCSVAdapter().read(underlying_file)
        frames.append(
            detect_large_market_moves(
                underlying,
                threshold=large_move_threshold,
                symbols=market_symbols,
            )
        )
    nonempty = [frame for frame in frames if not frame.is_empty()]
    if not nonempty:
        raise ValueError("at least one event or underlying input is required")
    combined = pl.concat(nonempty, how="diagonal_relaxed")
    return combined.unique(subset=["event_id"], keep="last").sort("event_timestamp")
