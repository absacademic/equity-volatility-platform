# Deterministic synthetic option chains for tests

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from vol_platform.pricing.black76 import price
from vol_platform.surface.expiration import exact_time_to_expiry


def synthetic_rate_curve(as_of_date: date = date(2026, 7, 1)) -> pl.DataFrame:
    maturities = [
        date(2026, 7, 15),
        date(2026, 8, 15),
        date(2026, 10, 1),
        date(2027, 1, 1),
    ]
    rates = [0.0400, 0.0410, 0.0420, 0.0430]
    return pl.DataFrame(
        {
            "as_of_date": [as_of_date] * len(maturities),
            "maturity_date": maturities,
            "rate": rates,
            "currency": ["USD"] * len(maturities),
            "source": ["synthetic_curve"] * len(maturities),
        }
    )


def synthetic_clean_chain() -> pl.DataFrame:
    # Build a small clean SPY chain with three expirations and known smiles

    quote_timestamp = datetime(2026, 7, 1, 14, 30, tzinfo=UTC)
    spot = 600.0
    dividend_yield = 0.012
    expirations = [date(2026, 8, 21), date(2026, 9, 18), date(2026, 12, 18)]
    strikes = [540.0, 555.0, 570.0, 585.0, 600.0, 615.0, 630.0, 645.0, 660.0]
    rows = []
    row_number = 1

    for expiration_index, expiration in enumerate(expirations):
        time_to_expiry = exact_time_to_expiry(quote_timestamp, expiration)
        rate = 0.0405 + 0.0008 * expiration_index
        forward = spot * math.exp((rate - dividend_yield) * time_to_expiry)
        base_volatility = 0.205 + 0.012 * expiration_index
        for strike in strikes:
            log_moneyness = math.log(strike / forward)
            volatility = base_volatility - 0.18 * log_moneyness + 0.85 * log_moneyness**2
            parity_noise = 0.008 * math.sin(strike / 13.0)
            for option_type in ("call", "put"):
                theoretical = price(
                    forward,
                    strike,
                    time_to_expiry,
                    rate,
                    volatility,
                    option_type,
                )
                signed_noise = parity_noise if option_type == "call" else -parity_noise
                mid = max(theoretical + signed_noise, 0.015)
                half_spread = min(0.08, max(0.015, 0.008 * mid))
                bid = max(mid - half_spread, 0.005)
                ask = mid + half_spread
                actual_mid = 0.5 * (bid + ask)
                spread = ask - bid
                rows.append(
                    {
                        "quote_timestamp": quote_timestamp,
                        "symbol": f"SPY-{expiration}-{int(strike)}-{option_type[0].upper()}",
                        "underlying_symbol": "SPY",
                        "expiration": expiration,
                        "strike": strike,
                        "option_type": option_type,
                        "bid": bid,
                        "ask": ask,
                        "last": actual_mid,
                        "bid_size": 50,
                        "ask_size": 50,
                        "volume": 100,
                        "open_interest": 1_000,
                        "exchange": "SYNTHETIC",
                        "currency": "USD",
                        "multiplier": 100,
                        "source_row": row_number,
                        "source": "synthetic",
                        "source_file": "generated",
                        "underlying_timestamp": quote_timestamp,
                        "underlying_price": spot,
                        "underlying_source": "synthetic",
                        "alignment_delay_seconds": 0.0,
                        "underlying_is_stale": False,
                        "risk_free_rate": rate,
                        "rate_as_of_date": quote_timestamp.date(),
                        "rate_maturity_date": expiration,
                        "rate_source": "synthetic_curve",
                        "mid": actual_mid,
                        "spread": spread,
                        "relative_spread": spread / actual_mid,
                        "moneyness": strike / spot,
                        "is_duplicate": False,
                        "is_wide_spread": False,
                        "is_low_volume": False,
                        "is_low_open_interest": False,
                        "is_implausible_moneyness": False,
                        "is_valid": True,
                        "rejection_reason": None,
                        "quote_quality_score": 0.95,
                        "quote_date": quote_timestamp.date(),
                    }
                )
                row_number += 1
    return pl.DataFrame(rows)


def write_synthetic_inputs(output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    chain_path = output / "synthetic-clean-chain.parquet"
    rates_path = output / "synthetic-rates.csv"
    synthetic_clean_chain().write_parquet(chain_path)
    synthetic_rate_curve().write_csv(rates_path)
    return chain_path, rates_path
