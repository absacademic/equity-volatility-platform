# Event outcomes and transparent daily straddle P&L approximation

from __future__ import annotations

import math
from datetime import date
from typing import Any

import polars as pl


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _price_map(underlying: pl.DataFrame) -> dict[tuple[str, date], float]:
    result: dict[tuple[str, date], float] = {}
    for row in underlying.sort("timestamp").iter_rows(named=True):
        timestamp = row.get("timestamp")
        price = _finite(row.get("underlying_price", row.get("last")))
        if timestamp is None or price is None:
            continue
        result[(str(row.get("symbol", "")).upper(), timestamp.date())] = price
    return result


def _post_event_surface(
    event: dict[str, Any],
    daily_features: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in daily_features
        if str(row.get("symbol", "")).upper() == str(event["symbol"]).upper()
        and row.get("quote_timestamp") is not None
        and row["quote_timestamp"] > event["event_timestamp"]
        and row.get("chain_valid", True) is not False
    ]
    if not candidates:
        return None
    earliest = min(row["quote_timestamp"] for row in candidates)
    same_time = [row for row in candidates if row["quote_timestamp"] == earliest]
    target = (_finite(event.get("feature_time_to_expiry")) or 30.0 / 365.0) * 365.0
    return min(
        same_time,
        key=lambda row: abs((_finite(row.get("time_to_expiry")) or target / 365.0) * 365 - target),
    )


def calculate_event_outcomes(
    event_features: pl.DataFrame,
    event_windows: pl.DataFrame,
    underlying: pl.DataFrame,
    daily_features: pl.DataFrame,
    *,
    annualization_days: int = 252,
    fixed_cost_bps: float = 8.0,
    spread_cost_multiplier: float = 0.5,
    vega_scale: float = 0.25,
) -> pl.DataFrame:
    # Estimate event returns, IV changes, and a normalized delta-hedged straddle result

    prices = _price_map(underlying)
    window_rows = list(event_windows.iter_rows(named=True))
    windows = {
        (str(row["event_id"]), int(row["trading_day_offset"])): row["window_date"]
        for row in window_rows
    }
    complete_events = {
        str(row["event_id"]) for row in window_rows if bool(row.get("window_complete"))
    }
    surfaces = list(daily_features.iter_rows(named=True))
    rows: list[dict[str, Any]] = []
    for event in event_features.iter_rows(named=True):
        event_id = str(event["event_id"])
        if event_id not in complete_events:
            continue
        symbol = str(event["symbol"]).upper()
        start_date = windows.get((event_id, -1))
        end_date = windows.get((event_id, 0))
        start = prices.get((symbol, start_date)) if start_date else None
        end = prices.get((symbol, end_date)) if end_date else None
        expected_move = _finite(event.get("expected_move"))
        atm = _finite(event.get("atm_volatility"))
        if start is None or end is None or expected_move is None or atm is None:
            continue
        signed_return = end / start - 1.0
        absolute_return = abs(signed_return)
        post = _post_event_surface(event, surfaces)
        post_atm = _finite(post.get("atm_implied_volatility")) if post else None
        post_skew = _finite(post.get("downside_skew_25")) if post else None
        pre_skew = _finite(event.get("skew"))
        atm_change = post_atm - atm if post_atm is not None else None
        skew_change = (
            post_skew - pre_skew if post_skew is not None and pre_skew is not None else None
        )
        iv_change_pnl = (
            vega_scale * atm_change / math.sqrt(annualization_days)
            if atm_change is not None
            else 0.0
        )
        spread = max(_finite(event.get("iv_bid_ask_width")) or 0.0, 0.0)
        transaction_cost = fixed_cost_bps / 10_000.0 + (
            spread_cost_multiplier
            * spread
            * math.sqrt(2.0 / math.pi)
            / math.sqrt(annualization_days)
        )
        realized_move_pnl = absolute_return
        implied_move_carry = -expected_move
        long_gross = realized_move_pnl + implied_move_carry + iv_change_pnl
        long_net = long_gross - transaction_cost
        row = dict(event)
        row.update(
            {
                "event_start_date": start_date,
                "event_end_date": end_date,
                "start_price": start,
                "end_price": end,
                "signed_return": signed_return,
                "absolute_return": absolute_return,
                "expected_minus_realized_move": expected_move - absolute_return,
                "market_overestimated": expected_move > absolute_return,
                "post_event_atm_volatility": post_atm,
                "post_event_iv_collapse": atm - post_atm if post_atm is not None else None,
                "atm_volatility_change": atm_change,
                "skew_change": skew_change,
                "realized_move_pnl": realized_move_pnl,
                "implied_move_carry": implied_move_carry,
                "iv_change_pnl": iv_change_pnl,
                "estimated_transaction_cost": transaction_cost,
                "long_straddle_gross_return": long_gross,
                "long_straddle_net_return": long_net,
                "delta_hedged_straddle_return": long_net / max(expected_move, 1.0e-8),
                "pnl_method": "daily_atm_straddle_approximation",
            }
        )
        rows.append(row)
    return pl.DataFrame(rows).sort("event_timestamp") if rows else pl.DataFrame()
