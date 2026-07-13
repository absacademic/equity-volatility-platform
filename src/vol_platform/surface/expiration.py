# Gives option-expiration timestamps and year fractions

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

_SECONDS_PER_DAY = 86_400.0


def expiration_timestamp(
    expiration: date,
    *,
    timezone: str = "America/New_York",
    close_time: time = time(16, 0),
) -> datetime:
    # Returns the timezone expiration timestamp for contract date

    return datetime.combine(expiration, close_time, tzinfo=ZoneInfo(timezone))


def exact_time_to_expiry(
    quote_timestamp: datetime,
    expiration: date,
    *,
    timezone: str = "America/New_York",
    close_time: time = time(16, 0),
    day_count_basis: float = 365.0,
) -> float:
    # Calculate time to expiration from elapsed seconds

    if day_count_basis <= 0.0:
        raise ValueError("day_count_basis must be positive")
    local_zone = ZoneInfo(timezone)
    if quote_timestamp.tzinfo is None:
        quote_timestamp = quote_timestamp.replace(tzinfo=local_zone)
    expiry = expiration_timestamp(expiration, timezone=timezone, close_time=close_time)
    expiry_utc = expiry.astimezone(ZoneInfo("UTC"))
    quote_utc = quote_timestamp.astimezone(ZoneInfo("UTC"))
    seconds = (expiry_utc - quote_utc).total_seconds()
    return seconds / (_SECONDS_PER_DAY * day_count_basis)
