## Initial universe

The first analytical universe remains **SPY**. It is a liquid ETF with a broad option chain and frequent expirations. Standard ETF options are American-style, physically settled, and normally use a 100-share contract multiplier.

## Expected data availability

| Dataset | Expected frequency | Minimum fields | Main limitations |
|---|---|---|---|
| Option quotes | Tick, snapshot, or end-of-day | timestamp, contract, underlying, expiration, strike, type, bid, ask | Vendor symbology differs; historical NBBO data may be expensive; volume and open interest can be delayed or missing. |
| Underlying prices | Tick, minute, or daily | timestamp, symbol, bid/ask or last | A delayed underlying observation can distort moneyness and implied volatility. |
| Interest rates | Daily | as-of date, maturity date, rate | A simple curve point is only an approximation for the exact option maturity. |
| Event data | Event time or daily calendar | event id, type, timestamp, title | Event timestamps and expected/actual status vary by source. |

## Cleaning policy

Every input quote is retained. Clean rows have `is_valid = true`. Rejected rows have `is_valid = false`, a semicolon-separated `rejection_reason`, and a `quote_quality_score` from 0 to 1.

The current rules reject missing or invalid required fields, expired contracts, nonpositive or crossed quotes, duplicate observations, wide spreads, low volume, low open interest, implausible moneyness, missing underlying matches, and stale underlying matches. Thresholds are in `configs/base.yml`.

## Storage layout

Clean and rejected options are written as partitioned Parquet files under:

```text
data/processed/options/clean/underlying_symbol=SPY/quote_date=YYYY-MM-DD/
data/processed/options/rejected/underlying_symbol=SPY/quote_date=YYYY-MM-DD/
```

Each run also creates a metadata manifest, a Markdown data-quality report, a CSV rejection summary, normalized reference tables, and a DuckDB database with these views:

- `clean_quotes`
- `rejected_quotes`
- `daily_summaries`
- `expiration_chains`

## Source references

Contract-style assumptions follow these official references:

- [OCC ETF Options](https://www.theocc.com/clearance-and-settlement/clearing/etf-options)
- [Cboe: European Style versus SPY ETF Options](https://www.cboe.com/tradable_products/sp_500/mini_spx_options/european_style/)