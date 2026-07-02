from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vol_platform.pricing.implied_vol import IVMethod, IVStatus
from vol_platform.types import OptionType, PricingModel


class StrictRecord(BaseModel):
    # Used for immutable ingestion records

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="before")
    @classmethod
    def strip_strings(cls, value: object) -> object:
        # Strip whitespace from str
        if isinstance(value, str):
            return value.strip()
        return value


class DividendType(StrEnum):
    REGULAR = "regular"
    SPECIAL = "special"
    ESTIMATED = "estimated"


class EventType(StrEnum):
    EARNINGS = "earnings"
    MACRO = "macro"
    CORPORATE_ACTION = "corporate_action"
    INDEX_REBALANCE = "index_rebalance"
    OTHER = "other"


class OptionQuote(StrictRecord):
    # Option quote used as the raw option-chain record

    timestamp: datetime
    symbol: str = Field(min_length=1)
    underlying_symbol: str = Field(min_length=1)
    expiration: date
    strike: float = Field(gt=0.0)
    option_type: OptionType
    bid: float = Field(ge=0.0)
    ask: float = Field(ge=0.0)
    last: float | None = Field(default=None, ge=0.0)
    bid_size: int | None = Field(default=None, ge=0)
    ask_size: int | None = Field(default=None, ge=0)
    volume: int | None = Field(default=None, ge=0)
    open_interest: int | None = Field(default=None, ge=0)
    exchange: str | None = None
    currency: str = Field(default="USD", min_length=3, max_length=3)
    multiplier: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def validate_market(self) -> OptionQuote:
        if self.ask < self.bid:
            raise ValueError("ask must be greater than or equal to bid")
        if self.expiration < self.timestamp.date():
            raise ValueError("expiration cannot precede quote timestamp")
        return self

    @property
    def mid(self) -> float:
        # Returns (artihmetic) bid-ask midpoint
        return 0.5 * (self.bid + self.ask)


class UnderlyingPrice(StrictRecord):
    # Underlying top-of-book or last-trade observation

    timestamp: datetime
    symbol: str = Field(min_length=1)
    bid: float | None = Field(default=None, gt=0.0)
    ask: float | None = Field(default=None, gt=0.0)
    last: float | None = Field(default=None, gt=0.0)
    volume: int | None = Field(default=None, ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_prices(self) -> UnderlyingPrice:
        if self.bid is None and self.ask is None and self.last is None:
            raise ValueError("at least one of bid, ask, or last must be supplied")
        if self.bid is not None and self.ask is not None and self.ask < self.bid:
            raise ValueError("ask must be greater than or equal to bid")
        return self

    @property
    def reference_price(self) -> float:
        # Prefer the midpoint to last, then whichever side is available
        if self.bid is not None and self.ask is not None:
            return 0.5 * (self.bid + self.ask)
        if self.last is not None:
            return self.last
        return self.bid if self.bid is not None else float(self.ask)


class RateCurvePoint(StrictRecord):
    # Continuously compuounded risk-free rate at one maturity

    as_of_date: date
    maturity_date: date
    rate: float
    currency: str = Field(default="USD", min_length=3, max_length=3)
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_maturity(self) -> RateCurvePoint:
        if self.maturity_date <= self.as_of_date:
            raise ValueError("maturity_date must be after as_of_date")
        return self


class DividendRecord(StrictRecord):
    # Known or est. cash dividend

    symbol: str = Field(min_length=1)
    ex_date: date
    amount: float = Field(ge=0.0)
    payment_date: date | None = None
    dividend_type: DividendType = DividendType.REGULAR
    currency: str = Field(default="USD", min_length=3, max_length=3)
    source: str | None = None

    @model_validator(mode="after")
    def validate_payment_date(self) -> DividendRecord:
        if self.payment_date is not None and self.payment_date < self.ex_date:
            raise ValueError("payment_date cannot precede ex_date")
        return self


class EventRecord(StrictRecord):
    # Timestamped event to be used for event-window studies

    event_id: str = Field(min_length=1)
    event_type: EventType
    event_timestamp: datetime
    title: str = Field(min_length=1)
    symbols: tuple[str, ...] = ()
    source: str | None = None
    expected: bool = True
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ImpliedVolatilityRecord(StrictRecord):
    # Output record from quote normalization and implied-volatilty inversion

    quote_timestamp: datetime
    symbol: str = Field(min_length=1)
    expiration: date
    strike: float = Field(gt=0.0)
    option_type: OptionType
    model: PricingModel
    option_price: float = Field(ge=0.0)
    spot: float | None = Field(default=None, gt=0.0)
    forward: float | None = Field(default=None, gt=0.0)
    rate: float
    dividend_yield: float = 0.0
    implied_volatility: float | None = Field(default=None, ge=0.0)
    status: IVStatus
    method: IVMethod = IVMethod.NONE
    iterations: int = Field(default=0, ge=0)
    residual: float | None = None

    @model_validator(mode="after")
    def validate_model_inputs(self) -> ImpliedVolatilityRecord:
        if self.expiration < self.quote_timestamp.date():
            raise ValueError("expiration cannot precede quote timestamp")
        if self.model is PricingModel.BLACK_SCHOLES and self.spot is None:
            raise ValueError("spot is required for Black-Scholes records")
        if self.model is PricingModel.BLACK_76 and self.forward is None:
            raise ValueError("forward is required for Black-76 records")
        if (
            self.status in {IVStatus.SUCCESS, IVStatus.AT_LOWER_BOUND}
            and self.implied_volatility is None
        ):
            raise ValueError("successful records require implied_volatility")
        return self
