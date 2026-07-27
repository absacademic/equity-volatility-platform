from pathlib import Path

from vol_platform.data.adapters import (
    DividendCSVAdapter,
    EventCSVAdapter,
    RateCSVAdapter,
    UnderlyingPriceCSVAdapter,
)


def test_reference_adapters_normalize_csv_files(tmp_path: Path) -> None:
    underlying_path = tmp_path / "underlying.csv"
    underlying_path.write_text(
        "timestamp,ticker,bid,ask\n2026-07-01T14:30:00Z,spy,599.9,600.1\n",
        encoding="utf-8",
    )
    rates_path = tmp_path / "rates.csv"
    rates_path.write_text(
        "date,maturity,value\n2026-07-01,2026-08-01,0.04\n",
        encoding="utf-8",
    )
    dividends_path = tmp_path / "dividends.csv"
    dividends_path.write_text(
        "ticker,ex_dividend_date,cash_amount\nspy,2026-07-17,1.75\n",
        encoding="utf-8",
    )
    events_path = tmp_path / "events.csv"
    events_path.write_text(
        "id,type,timestamp,title,symbol\n1,macro,2026-07-29T18:00:00Z,FOMC,SPY\n",
        encoding="utf-8",
    )

    underlying = UnderlyingPriceCSVAdapter().read(underlying_path)
    rates = RateCSVAdapter().read(rates_path)
    dividends = DividendCSVAdapter().read(dividends_path)
    events = EventCSVAdapter().read(events_path)

    assert underlying["symbol"][0] == "SPY"
    assert underlying["underlying_price"][0] == 600.0
    assert rates["rate"][0] == 0.04
    assert rates["source"][0] == "local_csv"
    assert dividends["symbol"][0] == "SPY"
    assert dividends["amount"][0] == 1.75
    assert events["event_type"][0] == "macro"
    assert events["expected"][0]
