# Aligns each quote with underlying observation and records the delay and staleness

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from datetime import date, datetime
from typing import Any

import polars as pl


def align_underlying_prices(
    quotes: pl.DataFrame,
    underlying: pl.DataFrame,
    max_staleness_seconds: float,
) -> pl.DataFrame:
    # Attach latest underlying observation on or before each quote

    history: dict[str, list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    for row in underlying.sort(["symbol", "timestamp"]).iter_rows(named=True):
        if row["symbol"] and row["timestamp"] and row["underlying_price"] is not None:
            history[row["symbol"]].append((row["timestamp"], row))

    columns: dict[str, list[Any]] = defaultdict(list)
    for quote in quotes.iter_rows(named=True):
        symbol = quote.get("underlying_symbol")
        timestamp = quote.get("quote_timestamp")
        match = None
        if symbol and timestamp and history.get(symbol):
            times = [item[0] for item in history[symbol]]
            position = bisect_right(times, timestamp) - 1
            if position >= 0:
                match = history[symbol][position][1]

        if match is None:
            columns["underlying_timestamp"].append(None)
            columns["underlying_price"].append(None)
            columns["underlying_source"].append(None)
            columns["alignment_delay_seconds"].append(None)
            columns["underlying_is_stale"].append(True)
        else:
            delay = (timestamp - match["timestamp"]).total_seconds()
            columns["underlying_timestamp"].append(match["timestamp"])
            columns["underlying_price"].append(match["underlying_price"])
            columns["underlying_source"].append(match["source"])
            columns["alignment_delay_seconds"].append(delay)
            columns["underlying_is_stale"].append(delay > max_staleness_seconds)

    return quotes.with_columns([pl.Series(name, values) for name, values in columns.items()])


def align_rates(
    quotes: pl.DataFrame,
    rates: pl.DataFrame | None,
    default_rate: float,
) -> pl.DataFrame:
    # Attach latest available curve point nearest the option expiration

    rows = rates.iter_rows(named=True) if rates is not None else []
    available = [row for row in rows if row["as_of_date"] and row["maturity_date"]]
    out: dict[str, list[Any]] = defaultdict(list)

    for quote in quotes.iter_rows(named=True):
        timestamp: datetime | None = quote.get("quote_timestamp")
        expiration: date | None = quote.get("expiration")
        quote_date = timestamp.date() if timestamp else None
        candidates = [
            row
            for row in available
            if quote_date is not None
            and expiration is not None
            and row["as_of_date"] <= quote_date
            and row["currency"] == quote.get("currency", "USD")
        ]
        if candidates:
            latest_date = max(row["as_of_date"] for row in candidates)
            latest = [row for row in candidates if row["as_of_date"] == latest_date]
            match = min(latest, key=lambda row: abs((row["maturity_date"] - expiration).days))
            out["risk_free_rate"].append(match["rate"])
            out["rate_as_of_date"].append(match["as_of_date"])
            out["rate_maturity_date"].append(match["maturity_date"])
            out["rate_source"].append(match["source"])
        else:
            out["risk_free_rate"].append(default_rate)
            out["rate_as_of_date"].append(None)
            out["rate_maturity_date"].append(None)
            out["rate_source"].append("config_default")

    return quotes.with_columns([pl.Series(name, values) for name, values in out.items()])
