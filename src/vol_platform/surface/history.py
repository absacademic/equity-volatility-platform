# Point-in-time realized volatility, event links, and historical comparisons

from __future__ import annotations

import math
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

HISTORICAL_COLUMNS = (
    "atm_implied_volatility",
    "downside_skew_25",
    "risk_reversal_25",
    "butterfly_25",
    "iv_bid_ask_width",
    "surface_residual_rmse",
    "total_option_volume",
    "total_open_interest",
    "realized_volatility_20d",
    "vrp_variance_20d",
)


def build_underlying_history(
    quotes: pl.DataFrame,
    external_history: pl.DataFrame | None = None,
) -> pl.DataFrame:
    # Return one observed underlying price per symbol and date

    quote_prices = quotes.select(
        pl.col("quote_date"),
        pl.col("underlying_symbol").alias("symbol"),
        pl.col("underlying_price").cast(pl.Float64),
    )
    frames = [quote_prices]
    if external_history is not None and not external_history.is_empty():
        frames.append(
            external_history.select(
                pl.col("timestamp").dt.date().alias("quote_date"),
                pl.col("symbol"),
                pl.col("underlying_price").cast(pl.Float64),
            )
        )
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .drop_nulls(["quote_date", "symbol", "underlying_price"])
        .group_by(["quote_date", "symbol"])
        .agg(pl.col("underlying_price").last())
        .sort(["symbol", "quote_date"])
    )


def add_realized_volatility_features(
    features: pl.DataFrame,
    underlying_history: pl.DataFrame,
    *,
    annualization_days: int = 252,
    windows: tuple[int, ...] = (5, 20, 60),
) -> pl.DataFrame:
    # Add trailing realized volatility using prices strictly before each quote date

    if features.is_empty():
        return features
    history_map: dict[str, list[tuple[object, float]]] = {}
    for row in underlying_history.iter_rows(named=True):
        history_map.setdefault(str(row["symbol"]), []).append(
            (row["quote_date"], float(row["underlying_price"]))
        )

    output: list[dict[str, Any]] = []
    for row in features.iter_rows(named=True):
        observations = [
            (observation_date, price)
            for observation_date, price in history_map.get(str(row["symbol"]), [])
            if observation_date < row["quote_date"]
        ]
        prices = np.asarray([price for _, price in observations], dtype=float)
        returns = np.diff(np.log(prices)) if prices.size >= 2 else np.asarray([], dtype=float)
        for window in windows:
            value = None
            if returns.size >= window:
                sample = returns[-window:]
                value = float(np.std(sample, ddof=1) * math.sqrt(annualization_days))
            row[f"realized_volatility_{window}d"] = value
        realized_20 = row.get("realized_volatility_20d")
        atm = row.get("atm_implied_volatility")
        row["vrp_volatility_20d"] = (
            atm - realized_20 if atm is not None and realized_20 is not None else None
        )
        row["vrp_variance_20d"] = (
            atm**2 - realized_20**2 if atm is not None and realized_20 is not None else None
        )
        output.append(row)
    return pl.DataFrame(output)


def _event_matches_symbol(event_symbols: object, symbol: str) -> bool:
    if event_symbols is None:
        return True
    values = {part.strip().upper() for part in str(event_symbols).replace(";", ",").split(",")}
    return symbol.upper() in values or "ALL" in values or "MARKET" in values


def add_event_linked_features(
    features: pl.DataFrame,
    events: pl.DataFrame | None,
    *,
    timezone: str = "America/New_York",
) -> pl.DataFrame:
    # Attach nearest known past and scheduled future events without using later outcomes

    if features.is_empty():
        return features
    event_rows = list(events.iter_rows(named=True)) if events is not None else []
    output: list[dict[str, Any]] = []
    for row in features.iter_rows(named=True):
        quote_timestamp = row["quote_timestamp"]
        if quote_timestamp.tzinfo is None:
            quote_timestamp = quote_timestamp.replace(tzinfo=ZoneInfo(timezone))
        expiry_timestamp = datetime.combine(
            row["expiration"], time(16, 0), tzinfo=ZoneInfo(timezone)
        )
        matching = [
            event
            for event in event_rows
            if event.get("event_timestamp") is not None
            and _event_matches_symbol(event.get("symbols"), str(row["symbol"]))
            and (
                event.get("known_timestamp") is None
                or event["known_timestamp"] <= quote_timestamp
            )
        ]
        past = [event for event in matching if event["event_timestamp"] <= quote_timestamp]
        future = [
            event
            for event in matching
            if quote_timestamp < event["event_timestamp"] <= expiry_timestamp
            and bool(event.get("expected", True))
        ]
        latest = max(past, key=lambda event: event["event_timestamp"], default=None)
        next_event = min(future, key=lambda event: event["event_timestamp"], default=None)
        days_to_next = (
            (next_event["event_timestamp"] - quote_timestamp).total_seconds() / 86_400.0
            if next_event is not None
            else None
        )
        days_since = (
            (quote_timestamp - latest["event_timestamp"]).total_seconds() / 86_400.0
            if latest is not None
            else None
        )
        row.update(
            {
                "event_count_to_expiry": len(future),
                "next_event_id": next_event.get("event_id") if next_event else None,
                "next_event_type": next_event.get("event_type") if next_event else None,
                "days_to_next_event": days_to_next,
                "event_within_1d": bool(days_to_next is not None and days_to_next <= 1.0),
                "event_within_3d": bool(days_to_next is not None and days_to_next <= 3.0),
                "event_within_7d": bool(days_to_next is not None and days_to_next <= 7.0),
                "latest_event_id": latest.get("event_id") if latest else None,
                "latest_event_type": latest.get("event_type") if latest else None,
                "days_since_latest_event": days_since,
                "event_variance_density": (
                    row["atm_implied_volatility"] ** 2 / row["time_to_expiry"]
                    if next_event is not None
                    and row.get("atm_implied_volatility") is not None
                    and row["time_to_expiry"] > 0.0
                    else None
                ),
            }
        )
        output.append(row)
    return pl.DataFrame(output)


def _zscore(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    standard_deviation = float(np.std(finite, ddof=1))
    if standard_deviation <= 0.0:
        return 0.0
    return float((finite[-1] - float(np.mean(finite))) / standard_deviation)


def _percentile(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    current = finite[-1]
    return float(np.mean(finite <= current))


def add_historical_comparisons(
    features: pl.DataFrame,
    *,
    rolling_window: int = 20,
) -> pl.DataFrame:
    # Add lagged changes, rolling/expanding z-scores, percentiles, and ranks

    if features.is_empty():
        return features
    output: list[dict[str, Any]] = []
    for frame in features.partition_by(["symbol", "expiration"], maintain_order=True):
        ordered = frame.sort("quote_date")
        history: dict[str, list[float]] = {column: [] for column in HISTORICAL_COLUMNS}
        previous: dict[str, float | None] = {column: None for column in HISTORICAL_COLUMNS}
        for row in ordered.iter_rows(named=True):
            for column in HISTORICAL_COLUMNS:
                value = row.get(column)
                current = (
                    float(value)
                    if value is not None and math.isfinite(float(value))
                    else None
                )
                row[f"{column}_change"] = (
                    current - previous[column]
                    if current is not None and previous[column] is not None
                    else None
                )
                history[column].append(current if current is not None else math.nan)
                expanding = np.asarray(history[column], dtype=float)
                rolling = expanding[-rolling_window:]
                row[f"{column}_rolling_zscore"] = (
                    _zscore(rolling) if current is not None else None
                )
                row[f"{column}_expanding_zscore"] = (
                    _zscore(expanding) if current is not None else None
                )
                row[f"{column}_expanding_percentile"] = (
                    _percentile(expanding) if current is not None else None
                )
                previous[column] = current
            output.append(row)

    result = pl.DataFrame(output)
    rank_rows: list[dict[str, Any]] = []
    for frame in result.partition_by(["quote_date", "expiration"], maintain_order=True):
        rows = list(frame.iter_rows(named=True))
        for column in (
            "atm_implied_volatility",
            "downside_skew_25",
            "vrp_variance_20d",
        ):
            valid_values = np.asarray(
                [
                    float(row[column])
                    for row in rows
                    if row.get(column) is not None
                    and math.isfinite(float(row[column]))
                ],
                dtype=float,
            )
            for row in rows:
                value = row.get(column)
                finite_value = (
                    float(value)
                    if value is not None and math.isfinite(float(value))
                    else None
                )
                row[f"{column}_cross_sectional_rank"] = (
                    float(np.mean(valid_values <= finite_value))
                    if finite_value is not None and valid_values.size
                    else None
                )
        rank_rows.extend(rows)
    return pl.DataFrame(rank_rows).sort(["quote_date", "symbol", "expiration"])


def create_historical_plots(features: pl.DataFrame, output_dir: Path) -> list[Path]:
    # Create compact historical ATM, skew, and volatility-risk-premium charts

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if features.is_empty():
        return paths

    def save(column: str, ylabel: str, title: str, filename: str) -> None:
        figure = Figure()
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for frame in features.partition_by(["symbol", "expiration"], maintain_order=True):
            ordered = frame.sort("quote_date").drop_nulls([column])
            if ordered.is_empty():
                continue
            label = f"{ordered['symbol'][0]} {ordered['expiration'][0]}"
            axis.plot(
                ordered["quote_date"].to_list(),
                ordered[column].to_list(),
                marker="o",
                label=label,
            )
        axis.set_xlabel("Quote date")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if axis.lines:
            axis.legend(fontsize="small")
        figure.autofmt_xdate()
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        figure.clear()
        paths.append(path)

    save(
        "atm_implied_volatility",
        "ATM implied volatility",
        "Historical ATM volatility",
        "historical_atm_iv.png",
    )
    save(
        "downside_skew_25",
        "25-delta put IV minus ATM IV",
        "Historical downside skew",
        "historical_skew.png",
    )
    save(
        "vrp_variance_20d",
        "ATM variance minus realized variance",
        "Historical volatility risk premium",
        "historical_vrp.png",
    )
    return paths
