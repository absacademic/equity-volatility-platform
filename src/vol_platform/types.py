from __future__ import annotations

from enum import StrEnum


class OptionType(StrEnum):
    # European option payoff types

    CALL = "call"
    PUT = "put"


class PricingModel(StrEnum):
    # Current pricing models
    BLACK_SCHOLES = "black_scholes"
    BLACK_76 = "black_76"
