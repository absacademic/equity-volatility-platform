# Modular CSV adapters for option quotes, underlying prices, interest rates, and events

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

_NULLS = ["", "NA", "N/A", "null", "None"]


def _read_csv(path: str | Path) -> pl.DataFrame:
    return pl.read_csv(
        Path(path),
        infer_schema_length=10_000,
        null_values=_NULLS,
        try_parse_dates=False,
    )


def _rename_aliases(
    frame: pl.DataFrame,
    aliases: dict[str, tuple[str, ...]],
    *,
    prefer_aliases: bool = False,
    include_target: bool = True,
) -> pl.DataFrame:
    by_lower = {name.lower(): name for name in frame.columns}
    rename: dict[str, str] = {}
    for target, candidates in aliases.items():
        if include_target:
            search_order = (*candidates, target) if prefer_aliases else (target, *candidates)
        else:
            search_order = candidates
        for candidate in search_order:
            actual = by_lower.get(candidate.lower())
            if actual is not None:
                rename[actual] = target
                break
    return frame.rename(rename)


def _add_missing(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    missing = [pl.lit(None).alias(name) for name in columns if name not in frame.columns]
    return frame.with_columns(missing) if missing else frame


def _datetime(name: str) -> pl.Expr:
    return (
        pl.col(name)
        .cast(pl.String, strict=False)
        .str.to_datetime(format="%+", time_zone="UTC", strict=False)
        .alias(name)
    )


def _date(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.String, strict=False).str.to_date(strict=False).alias(name)


def _float(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Float64, strict=False).alias(name)


def _integer(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Int64, strict=False).alias(name)


class OptionQuoteCSVAdapter:
    # Normalize a CSV option file into the project column names

    aliases: ClassVar[dict[str, tuple[str, ...]]] = {
        "quote_timestamp": ("timestamp", "datetime", "quote_time"),
        "symbol": ("option_symbol", "contract_symbol"),
        "underlying_symbol": ("underlying", "root"),
        "expiration": ("expiry", "expiration_date"),
        "strike": ("strike_price",),
        "option_type": ("type", "right", "put_call"),
        "bid": ("bid_price",),
        "ask": ("ask_price",),
        "last": ("last_price",),
        "bid_size": ("bidsize",),
        "ask_size": ("asksize",),
        "volume": ("trade_volume",),
        "open_interest": ("oi",),
    }
    columns: ClassVar[list[str]] = [
        "quote_timestamp",
        "symbol",
        "underlying_symbol",
        "expiration",
        "strike",
        "option_type",
        "bid",
        "ask",
        "last",
        "bid_size",
        "ask_size",
        "volume",
        "open_interest",
        "exchange",
        "currency",
        "multiplier",
    ]

    def read(self, path: str | Path, source: str = "local_csv") -> pl.DataFrame:
        frame = _add_missing(_rename_aliases(_read_csv(path), self.aliases), self.columns)
        frame = frame.with_row_index("source_row", offset=1).with_columns(
            _datetime("quote_timestamp"),
            _date("expiration"),
            *[_float(name) for name in ("strike", "bid", "ask", "last")],
            *[_integer(name) for name in ("bid_size", "ask_size", "volume", "open_interest")],
            pl.col("multiplier").cast(pl.Int64, strict=False).fill_null(100),
            pl.col("currency").cast(pl.String, strict=False).fill_null("USD").str.to_uppercase(),
            pl.col("option_type").cast(pl.String, strict=False).str.to_lowercase(),
            pl.col("symbol").cast(pl.String, strict=False).str.strip_chars(),
            pl.col("underlying_symbol").cast(pl.String, strict=False).str.to_uppercase(),
            pl.lit(source).alias("source"),
            pl.lit(str(Path(path))).alias("source_file"),
        )
        return frame.select([*self.columns, "source_row", "source", "source_file"])


class UnderlyingPriceCSVAdapter:
    # Normalize standalone or option-file-embedded underlying prices

    aliases: ClassVar[dict[str, tuple[str, ...]]] = {
        "timestamp": ("underlying_timestamp", "price_timestamp"),
        "symbol": ("underlying_symbol", "ticker"),
        "bid": ("underlying_bid",),
        "ask": ("underlying_ask",),
        "last": ("underlying_last", "price", "underlying_price"),
        "volume": ("underlying_volume",),
    }
    columns: ClassVar[list[str]] = [
        "timestamp",
        "symbol",
        "bid",
        "ask",
        "last",
        "volume",
        "currency",
    ]

    def read(self, path: str | Path, source: str = "local_csv") -> pl.DataFrame:
        frame = _add_missing(_rename_aliases(_read_csv(path), self.aliases), self.columns)
        return self._normalize(frame, path, source)

    def read_embedded(self, path: str | Path, source: str = "embedded") -> pl.DataFrame:
        raw = _read_csv(path)
        embedded_aliases = {
            "timestamp": ("underlying_timestamp",),
            "symbol": ("underlying_symbol", "underlying"),
            "bid": ("underlying_bid",),
            "ask": ("underlying_ask",),
            "last": ("underlying_last", "underlying_price"),
            "volume": ("underlying_volume",),
            "currency": ("underlying_currency", "currency"),
        }
        by_lower = {name.lower(): name for name in raw.columns}
        expressions: list[pl.Expr] = []
        for target, candidates in embedded_aliases.items():
            actual = next(
                (by_lower[name.lower()] for name in candidates if name.lower() in by_lower),
                None,
            )
            expressions.append(
                pl.col(actual).alias(target) if actual is not None else pl.lit(None).alias(target)
            )
        return self._normalize(raw.select(expressions), path, source)

    def _normalize(self, frame: pl.DataFrame, path: str | Path, source: str) -> pl.DataFrame:
        frame = frame.with_columns(
            _datetime("timestamp"),
            *[_float(name) for name in ("bid", "ask", "last")],
            _integer("volume"),
            pl.col("symbol").cast(pl.String, strict=False).str.to_uppercase(),
            pl.col("currency").cast(pl.String, strict=False).fill_null("USD").str.to_uppercase(),
            pl.lit(source).alias("source"),
            pl.lit(str(Path(path))).alias("source_file"),
        )
        frame = frame.with_columns(
            pl.when(pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
            .then((pl.col("bid") + pl.col("ask")) / 2.0)
            .otherwise(pl.coalesce(["last", "bid", "ask"]))
            .alias("underlying_price")
        )
        return frame.select([*self.columns, "underlying_price", "source", "source_file"]).unique(
            subset=["timestamp", "symbol"], keep="last", maintain_order=True
        )


class RateCSVAdapter:
    aliases: ClassVar[dict[str, tuple[str, ...]]] = {
        "as_of_date": ("date", "observation_date"),
        "maturity_date": ("maturity", "tenor_date"),
        "rate": ("value", "yield"),
    }
    columns: ClassVar[list[str]] = ["as_of_date", "maturity_date", "rate", "currency", "source"]

    def read(self, path: str | Path, source: str = "local_csv") -> pl.DataFrame:
        frame = _add_missing(_rename_aliases(_read_csv(path), self.aliases), self.columns)
        return frame.with_columns(
            _date("as_of_date"),
            _date("maturity_date"),
            _float("rate"),
            pl.col("currency").cast(pl.String, strict=False).fill_null("USD").str.to_uppercase(),
            pl.col("source").cast(pl.String, strict=False).fill_null(source),
            pl.lit(str(Path(path))).alias("source_file"),
        ).select([*self.columns, "source_file"])


class EventCSVAdapter:
    aliases: ClassVar[dict[str, tuple[str, ...]]] = {
        "event_timestamp": ("timestamp", "datetime"),
        "event_id": ("id",),
        "event_type": ("type",),
        "symbols": ("symbol", "tickers"),
    }
    columns: ClassVar[list[str]] = [
        "event_id",
        "event_type",
        "event_timestamp",
        "title",
        "symbols",
        "source",
        "expected",
    ]

    def read(self, path: str | Path, source: str = "local_csv") -> pl.DataFrame:
        frame = _add_missing(_rename_aliases(_read_csv(path), self.aliases), self.columns)
        return frame.with_columns(
            _datetime("event_timestamp"),
            pl.col("event_id").cast(pl.String, strict=False),
            pl.col("event_type").cast(pl.String, strict=False).str.to_lowercase(),
            pl.col("title").cast(pl.String, strict=False),
            pl.col("symbols").cast(pl.String, strict=False),
            pl.col("source").cast(pl.String, strict=False).fill_null(source),
            pl.col("expected").cast(pl.Boolean, strict=False).fill_null(True),
            pl.lit(str(Path(path))).alias("source_file"),
        ).select([*self.columns, "source_file"])