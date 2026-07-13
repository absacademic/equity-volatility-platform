# IV and moneyness feature construction

from __future__ import annotations

import math
from typing import Any

import polars as pl

from vol_platform.pricing.greeks import black76_greeks
from vol_platform.pricing.implied_vol import solve_implied_volatility


def _solve(
    price: float,
    row: dict[str, Any],
    solver_options: dict[str, Any],
) -> tuple[float | None, str, int, float | None]:
    result = solve_implied_volatility(
        price,
        row["forward"],
        row["strike"],
        row["time_to_expiry"],
        row["interpolated_rate"],
        row["option_type"],
        model="black_76",
        **solver_options,
    )
    return result.volatility, str(result.status), result.iterations, result.residual


def build_implied_volatility_dataset(
    quotes: pl.DataFrame,
    forwards: pl.DataFrame,
    *,
    solver_options: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Attach forward features and bid/mid/ask Black-76 implied volatilities."""

    solver_options = solver_options or {}
    forward_map = {row["expiration"]: row for row in forwards.iter_rows(named=True)}
    output: list[dict[str, Any]] = []
    for row in quotes.iter_rows(named=True):
        estimate = forward_map.get(row["expiration"])
        if estimate is None or estimate.get("forward") is None:
            continue
        enriched = dict(row)
        enriched.update(
            {
                "forward": float(estimate["forward"]),
                "forward_pair_count": int(estimate["pair_count"]),
                "forward_relative_dispersion": estimate["relative_dispersion"],
                "forward_reliability": estimate["reliability"],
            }
        )
        bid_iv, bid_status, _, _ = _solve(float(row["bid"]), enriched, solver_options)
        mid_iv, mid_status, iterations, residual = _solve(
            float(row["mid"]), enriched, solver_options
        )
        ask_iv, ask_status, _, _ = _solve(float(row["ask"]), enriched, solver_options)

        strike = float(row["strike"])
        spot = float(row["underlying_price"])
        forward = float(enriched["forward"])
        time_to_expiry = float(row["time_to_expiry"])
        delta = None
        vega = None
        total_variance = None
        if mid_iv is not None and mid_iv > 0.0:
            greeks = black76_greeks(
                forward,
                strike,
                time_to_expiry,
                float(row["interpolated_rate"]),
                mid_iv,
                row["option_type"],
            )
            delta = greeks.delta
            vega = greeks.vega
            total_variance = mid_iv**2 * time_to_expiry

        is_otm = (row["option_type"] == "call" and strike >= forward) or (
            row["option_type"] == "put" and strike < forward
        )
        enriched.update(
            {
                "pricing_model": "black_76",
                "bid_implied_volatility": bid_iv,
                "mid_implied_volatility": mid_iv,
                "ask_implied_volatility": ask_iv,
                "bid_iv_status": bid_status,
                "mid_iv_status": mid_status,
                "ask_iv_status": ask_status,
                "iv_iterations": iterations,
                "iv_price_residual": residual,
                "log_moneyness": math.log(strike / spot),
                "forward_moneyness": math.log(strike / forward),
                "delta": delta,
                "vega": vega,
                "total_variance": total_variance,
                "bid_total_variance": bid_iv**2 * time_to_expiry if bid_iv is not None else None,
                "ask_total_variance": ask_iv**2 * time_to_expiry if ask_iv is not None else None,
                "is_otm": is_otm,
                "fit_eligible": bool(
                    is_otm
                    and mid_iv is not None
                    and total_variance is not None
                    and estimate["reliability"] != "unreliable"
                ),
            }
        )
        output.append(enriched)
    return pl.DataFrame(output) if output else pl.DataFrame()
