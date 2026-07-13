# Put-call-parity forward estimation from near-ATM option pairs

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class ForwardSettings:
    max_pairs: int = 5
    maximum_atm_distance: float = 0.10
    reliable_relative_dispersion: float = 0.005
    minimum_average_quality: float = 0.50


def _reliability(
    pair_count: int,
    relative_dispersion: float,
    average_quality: float,
    settings: ForwardSettings,
) -> str:
    if pair_count < 2 or not math.isfinite(relative_dispersion):
        return "unreliable"
    high_dispersion = 0.5 * settings.reliable_relative_dispersion
    high_quality = max(settings.minimum_average_quality, 0.75)
    if (
        relative_dispersion <= high_dispersion
        and average_quality >= high_quality
        and pair_count >= 3
    ):
        return "high"
    if (
        relative_dispersion <= settings.reliable_relative_dispersion
        and average_quality >= settings.minimum_average_quality
    ):
        return "medium"
    if relative_dispersion <= 0.015:
        return "low"
    return "unreliable"


def estimate_forwards(
    quotes: pl.DataFrame,
    settings: ForwardSettings | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Estimate one forward per expiration and return summary and pair details

    settings = settings or ForwardSettings()
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for expiration_frame in quotes.partition_by("expiration", maintain_order=True):
        expiration: date = expiration_frame["expiration"][0]
        spot = float(expiration_frame["underlying_price"].median())
        rate = float(expiration_frame["interpolated_rate"].median())
        time_to_expiry = float(expiration_frame["time_to_expiry"].median())
        discount = float(expiration_frame["discount_factor"].median())

        candidates: list[dict[str, Any]] = []
        for strike_frame in expiration_frame.partition_by("strike", maintain_order=True):
            calls = strike_frame.filter(pl.col("option_type") == "call")
            puts = strike_frame.filter(pl.col("option_type") == "put")
            if calls.is_empty() or puts.is_empty():
                continue
            call = calls.sort("quote_quality_score", descending=True).row(0, named=True)
            put = puts.sort("quote_quality_score", descending=True).row(0, named=True)
            strike = float(call["strike"])
            distance = abs(strike / spot - 1.0)
            if distance > settings.maximum_atm_distance:
                continue
            call_mid = float(call["mid"])
            put_mid = float(put["mid"])
            forward = strike + (call_mid - put_mid) / discount
            pair_spread = float(call["spread"] + put["spread"])
            quality = 0.5 * (float(call["quote_quality_score"]) + float(put["quote_quality_score"]))
            weight = max(quality, 0.05) / max(pair_spread, 1e-4)
            candidates.append(
                {
                    "expiration": expiration,
                    "strike": strike,
                    "spot": spot,
                    "time_to_expiry": time_to_expiry,
                    "interpolated_rate": rate,
                    "discount_factor": discount,
                    "call_mid": call_mid,
                    "put_mid": put_mid,
                    "pair_spread": pair_spread,
                    "pair_quality": quality,
                    "atm_distance": distance,
                    "pair_forward": forward,
                    "pair_weight": weight,
                }
            )

        selected = sorted(candidates, key=lambda row: (row["atm_distance"], row["pair_spread"]))[
            : settings.max_pairs
        ]
        if not selected:
            summary_rows.append(
                {
                    "expiration": expiration,
                    "time_to_expiry": time_to_expiry,
                    "interpolated_rate": rate,
                    "discount_factor": discount,
                    "forward": None,
                    "pair_count": 0,
                    "forward_std": None,
                    "relative_dispersion": None,
                    "forward_range": None,
                    "average_pair_quality": None,
                    "reliability": "unreliable",
                }
            )
            continue

        forwards = np.array([row["pair_forward"] for row in selected], dtype=float)
        weights = np.array([row["pair_weight"] for row in selected], dtype=float)
        weights = weights / weights.sum()
        estimate = float(np.sum(weights * forwards))
        standard_deviation = float(np.sqrt(np.sum(weights * (forwards - estimate) ** 2)))
        relative_dispersion = standard_deviation / estimate
        average_quality = float(np.mean([row["pair_quality"] for row in selected]))
        reliability = _reliability(len(selected), relative_dispersion, average_quality, settings)

        for row in selected:
            row["selected"] = True
            row["forward_estimate"] = estimate
            row["forward_residual"] = row["pair_forward"] - estimate
            pair_rows.append(row)

        summary_rows.append(
            {
                "expiration": expiration,
                "time_to_expiry": time_to_expiry,
                "interpolated_rate": rate,
                "discount_factor": discount,
                "forward": estimate,
                "pair_count": len(selected),
                "forward_std": standard_deviation,
                "relative_dispersion": relative_dispersion,
                "forward_range": float(forwards.max() - forwards.min()),
                "average_pair_quality": average_quality,
                "reliability": reliability,
            }
        )

    summary = pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame()
    pairs = pl.DataFrame(pair_rows) if pair_rows else pl.DataFrame()
    return summary, pairs
