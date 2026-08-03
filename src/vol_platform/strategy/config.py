from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    strategy_name: str = "event_conditioned_delta_hedged_straddle"
    target_dte_days: int = 30
    minimum_dte_days: int = 7
    maximum_dte_days: int = 60
    entry_days_before_event: int = 1
    holding_period_days: int = 1
    hedge_frequency_days: int = 1
    prediction_threshold: float = 0.0
    contracts_per_trade: int = 1
    maximum_contracts: int = 5
    portfolio_capital: float = 1_000_000.0
    maximum_capital_fraction_per_trade: float = 0.10
    short_margin_fraction: float = 0.20
    minimum_volume: int = 1
    minimum_open_interest: int = 100
    maximum_relative_spread: float = 0.35
    option_slippage_fraction_of_spread: float = 0.10
    commission_per_contract: float = 0.65
    hedge_cost_bps: float = 1.0
    financing_rate: float = 0.04
    dividend_yield: float = 0.0
    annualization_events: float = 12.0
    contract_multiplier: int = 100
    allow_long: bool = True
    allow_short: bool = True
    tradable_event_types: tuple[str, ...] = ("cpi", "fomc", "earnings")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> StrategyConfig:
        if not raw:
            return cls()
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in raw.items() if key in allowed}
        if "tradable_event_types" in values:
            values["tradable_event_types"] = tuple(
                str(item).lower() for item in values["tradable_event_types"]
            )
        config = cls(**values)
        config.validate()
        return config

    def with_overrides(self, **changes: Any) -> StrategyConfig:
        result = replace(self, **changes)
        result.validate()
        return result

    def validate(self) -> None:
        if self.minimum_dte_days < 0 or self.maximum_dte_days < self.minimum_dte_days:
            raise ValueError("DTE limits are invalid")
        if not self.minimum_dte_days <= self.target_dte_days <= self.maximum_dte_days:
            raise ValueError("target_dte_days must be within the configured DTE range")
        if self.entry_days_before_event < 0 or self.holding_period_days < 0:
            raise ValueError("entry and holding periods must be nonnegative")
        if not math.isfinite(self.prediction_threshold) or self.prediction_threshold < 0.0:
            raise ValueError("prediction_threshold must be finite and nonnegative")
        if self.hedge_frequency_days < 1:
            raise ValueError("hedge_frequency_days must be positive")
        if self.contracts_per_trade < 1 or self.maximum_contracts < 1:
            raise ValueError("contract counts must be positive")
        if self.portfolio_capital <= 0.0:
            raise ValueError("portfolio_capital must be positive")
        if not 0.0 < self.maximum_capital_fraction_per_trade <= 1.0:
            raise ValueError("maximum_capital_fraction_per_trade must be in (0, 1]")
        if self.short_margin_fraction <= 0.0:
            raise ValueError("short_margin_fraction must be positive")
        if self.minimum_volume < 0 or self.minimum_open_interest < 0:
            raise ValueError("liquidity limits must be nonnegative")
        if not 0.0 <= self.maximum_relative_spread <= 2.0:
            raise ValueError("maximum_relative_spread must be between zero and two")
        if self.option_slippage_fraction_of_spread < 0.0:
            raise ValueError("option slippage must be nonnegative")
        if self.commission_per_contract < 0.0 or self.hedge_cost_bps < 0.0:
            raise ValueError("transaction costs must be nonnegative")
        for name, value in (
            ("financing_rate", self.financing_rate),
            ("dividend_yield", self.dividend_yield),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.annualization_events <= 0.0:
            raise ValueError("annualization_events must be positive")
        if self.contract_multiplier < 1:
            raise ValueError("contract_multiplier must be positive")
        if not self.tradable_event_types:
            raise ValueError("tradable_event_types cannot be empty")
