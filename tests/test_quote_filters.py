from pathlib import Path

import polars as pl

from vol_platform.data.adapters import OptionQuoteCSVAdapter, UnderlyingPriceCSVAdapter
from vol_platform.data.alignment import align_underlying_prices
from vol_platform.data.cleaning import CleaningRules, clean_option_quotes


def test_quote_filters_retain_rows_and_record_reasons(tmp_path: Path) -> None:
    path = tmp_path / "quotes.csv"
    path.write_text(
        "quote_timestamp,option_symbol,underlying_symbol,expiration,strike,option_type,"
        "bid,ask,volume,open_interest,underlying_timestamp,underlying_last\n"
        "2026-07-01T14:30:00Z,A,SPY,2026-07-17,600,call,4.0,4.2,10,20,"
        "2026-07-01T14:29:59Z,600\n"
        "2026-07-01T14:30:01Z,B,SPY,2026-07-17,610,call,0.0,0.3,0,0,"
        "2026-07-01T14:29:59Z,600\n"
        "2026-07-01T14:30:02Z,C,SPY,2026-07-17,620,put,2.0,1.0,10,20,"
        "2026-07-01T14:29:59Z,600\n",
        encoding="utf-8",
    )
    quotes = OptionQuoteCSVAdapter().read(path)
    underlying = UnderlyingPriceCSVAdapter().read_embedded(path)
    aligned = align_underlying_prices(quotes, underlying, max_staleness_seconds=300)
    cleaned = clean_option_quotes(aligned, CleaningRules())

    assert cleaned.height == 3
    assert cleaned.filter(pl.col("is_valid")).height == 1
    reasons = cleaned.filter(~pl.col("is_valid"))["rejection_reason"].to_list()
    assert any("nonpositive_bid" in reason and "low_volume" in reason for reason in reasons)
    assert any("crossed_market" in reason for reason in reasons)
    assert cleaned["quote_quality_score"].to_list()[0] == 1.0


def test_duplicate_and_stale_quotes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "quotes.csv"
    path.write_text(
        "quote_timestamp,option_symbol,underlying_symbol,expiration,strike,option_type,"
        "bid,ask,volume,open_interest,underlying_timestamp,underlying_last\n"
        "2026-07-01T14:40:00Z,A,SPY,2026-07-17,600,call,4.0,4.2,10,20,"
        "2026-07-01T14:30:00Z,600\n"
        "2026-07-01T14:40:00Z,A,SPY,2026-07-17,600,call,4.0,4.2,10,20,"
        "2026-07-01T14:30:00Z,600\n",
        encoding="utf-8",
    )
    quotes = OptionQuoteCSVAdapter().read(path)
    underlying = UnderlyingPriceCSVAdapter().read_embedded(path)
    cleaned = clean_option_quotes(
        align_underlying_prices(quotes, underlying, max_staleness_seconds=60),
        CleaningRules(),
    )

    assert cleaned.filter(pl.col("is_valid")).is_empty()
    assert "stale_underlying_alignment" in cleaned["rejection_reason"][0]
    assert "duplicate_observation" in cleaned["rejection_reason"][1]
