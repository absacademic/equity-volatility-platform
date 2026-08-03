# Longer real-market history

History interfaces support CPI, FOMC, earnings, and large-market-move histories over any date range covered by the supplied files.

## Required event fields

```text
event_id,event_type,event_timestamp,known_timestamp,title,symbols,source,expected
```

`known_timestamp` is required for point-in-time validation. Scheduled CPI, FOMC, and earnings events should use the timestamp at which the date or announcement was known. Large-market-move events are generated from underlying closes and are marked as unexpected.

Combine event sources with:

```bash
vol-platform build-event-history \
  --macro-events data/raw/macro-events.csv \
  --earnings-events data/raw/earnings-events.csv \
  --underlying data/raw/underlying-history.csv \
  --market-symbols SPY,XSP \
  --large-move-threshold 0.03 \
  --output data/processed/events/combined-event-history.csv
```

## Required option-history fields

```text
quote_timestamp,symbol,underlying_symbol,expiration,strike,option_type,bid,ask,
volume,open_interest,multiplier,implied_volatility
```

The exact strategy requires historical contract quotes through each entry, hedge, and exit date. End-of-day data supports daily hedging. Intraday data can support finer hedge rules after the date-selection layer is extended.

## Panel comparison

Use a consistent timestamp convention and map each symbol to an asset type such as `equity`, `etf`, or `index`. The strategy metrics then compare results across these groups. Index options require an explicit, economically valid hedge instrument or hedge ratio in real research; the synthetic XSP example uses the index level directly to demonstrate the mechanics.

## Data availability limitation

Reliable historical option bid/ask quotes, open interest, and contract histories are commonly licensed. The repository therefore includes a deterministic long-history sample and vendor-neutral adapters.
