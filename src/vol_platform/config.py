# Project configuration with YAML; environment-based

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from vol_platform.types import PricingModel


class IVSolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_volatility: float = Field(
        default=0.20,
        gt=0.0,
        validation_alias=AliasChoices("initial_volatility", "inital_volatility"),
    )
    minimum_volatility: float = Field(default=1e-8, gt=0.0)
    maximum_volatility: float = Field(default=10.0, gt=0.0)
    price_tolerance: float = Field(default=1e-10, gt=0.0)
    volatility_tolerance: float = Field(default=1e-10, gt=0.0)
    max_iterations: int = Field(default=100, gt=0)


class PricingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_model: PricingModel = PricingModel.BLACK_SCHOLES
    default_rate: float = 0.04
    default_dividend_yield: float = 0.0
    iv: IVSolverConfig = Field(default_factory=IVSolverConfig)


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: dict[str, Any]
    universe: dict[str, Any]
    pricing: PricingConfig
    paths: dict[str, str]


def load_config(path: str | Path | None = None) -> ProjectConfig:
    # Load the YAML config selected by argument or environment variable
    selected = Path(path or os.getenv("VOL_PLATFORM_CONFIG_PATH", "configs/base.yml")).expanduser()
    if not selected.exists():
        raise FileNotFoundError(f"configuration file not found: {selected}")
    with selected.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return ProjectConfig.model_validate(raw)
