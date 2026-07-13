# Validates quotes while retaining every rejected row

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class CleaningRules:
    max_relative_spread: float = 0.50
    min_volume: int = 1
    min_open_interest: int = 1
    min_moneyness: float = 0.50
    max_moneyness: float = 1.50


_REQUIRED = (
    "quote_timestamp",
    "symbol",
    "underlying_symbol",
    "expiration",
    "strike",
    "option_type",
    "bid",
    "ask",
)


def clean_option_quotes(frame: pl.DataFrame, rules: CleaningRules) -> pl.DataFrame:
    # Flag invalid quotes, retain all input rows

    seen: set[tuple[Any, ...]] = set()
    results: dict[str, list[Any]] = {
        "mid": [],
        "spread": [],
        "relative_spread": [],
        "moneyness": [],
        "is_duplicate": [],
        "is_wide_spread": [],
        "is_low_volume": [],
        "is_low_open_interest": [],
        "is_implausible_moneyness": [],
        "is_valid": [],
        "rejection_reason": [],
        "quote_quality_score": [],
    }

    for row in frame.iter_rows(named=True):
        reasons: list[str] = []
        missing = [name for name in _REQUIRED if row.get(name) is None]
        if row.get("option_type") not in {"call", "put"}:
            missing.append("option_type")
        if missing:
            reasons.append("missing_or_invalid:" + ",".join(sorted(set(missing))))

        timestamp = row.get("quote_timestamp")
        expiration = row.get("expiration")
        if timestamp and expiration and expiration < timestamp.date():
            reasons.append("expired_contract")

        bid, ask, last = row.get("bid"), row.get("ask"), row.get("last")
        if bid is not None and bid <= 0:
            reasons.append("nonpositive_bid")
        if ask is not None and ask <= 0:
            reasons.append("nonpositive_ask")
        if last is not None and last < 0:
            reasons.append("negative_last")
        if bid is not None and ask is not None and ask < bid:
            reasons.append("crossed_market")

        mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
        spread = ask - bid if bid is not None and ask is not None else None
        relative_spread = spread / mid if mid is not None and mid > 0 else None
        wide = relative_spread is not None and relative_spread > rules.max_relative_spread
        if wide:
            reasons.append("wide_spread")

        low_volume = row.get("volume") is not None and row["volume"] < rules.min_volume
        low_oi = (
            row.get("open_interest") is not None and row["open_interest"] < rules.min_open_interest
        )
        if low_volume:
            reasons.append("low_volume")
        if low_oi:
            reasons.append("low_open_interest")

        underlying = row.get("underlying_price")
        strike = row.get("strike")
        moneyness = strike / underlying if strike and underlying and underlying > 0 else None
        implausible = moneyness is not None and not (
            rules.min_moneyness <= moneyness <= rules.max_moneyness
        )
        if implausible:
            reasons.append("implausible_moneyness")
        if underlying is None:
            reasons.append("missing_underlying_alignment")
        elif row.get("underlying_is_stale"):
            reasons.append("stale_underlying_alignment")

        key = (
            timestamp,
            row.get("symbol"),
            expiration,
            strike,
            row.get("option_type"),
        )
        duplicate = all(value is not None for value in key) and key in seen
        if all(value is not None for value in key):
            seen.add(key)
        if duplicate:
            reasons.append("duplicate_observation")

        score = _quality_score(reasons)
        results["mid"].append(mid)
        results["spread"].append(spread)
        results["relative_spread"].append(relative_spread)
        results["moneyness"].append(moneyness)
        results["is_duplicate"].append(duplicate)
        results["is_wide_spread"].append(wide)
        results["is_low_volume"].append(low_volume)
        results["is_low_open_interest"].append(low_oi)
        results["is_implausible_moneyness"].append(implausible)
        results["is_valid"].append(not reasons)
        results["rejection_reason"].append(";".join(reasons) if reasons else None)
        results["quote_quality_score"].append(score)

    return frame.with_columns(
        [pl.Series(name, values) for name, values in results.items()]
    ).with_columns(pl.col("quote_timestamp").dt.date().alias("quote_date"))


def _quality_score(reasons: list[str]) -> float:
    if any(
        reason.startswith("missing_or_invalid")
        or reason in {"expired_contract", "nonpositive_bid", "nonpositive_ask", "crossed_market"}
        for reason in reasons
    ):
        return 0.0
    penalties = {
        "negative_last": 0.15,
        "duplicate_observation": 0.40,
        "wide_spread": 0.20,
        "low_volume": 0.10,
        "low_open_interest": 0.10,
        "implausible_moneyness": 0.20,
        "missing_underlying_alignment": 0.40,
        "stale_underlying_alignment": 0.20,
    }
    return round(max(0.0, 1.0 - sum(penalties.get(reason, 0.0) for reason in reasons)), 4)
