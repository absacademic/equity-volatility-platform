from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from vol_platform.pricing._validation import coerce_option_type, require_finite
from vol_platform.types import OptionType


class BarrierType(StrEnum):
    UP_AND_OUT = "up_and_out"
    DOWN_AND_OUT = "down_and_out"
    UP_AND_IN = "up_and_in"
    DOWN_AND_IN = "down_and_in"


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_lower_95: float
    confidence_upper_95: float
    path_count: int
    step_count: int
    seed: int
    barrier_type: str
    barrier: float
    knock_probability: float
    antithetic: bool


def price_barrier_option(
    spot: float,
    strike: float,
    barrier: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType | str = OptionType.CALL,
    barrier_type: BarrierType | str = BarrierType.UP_AND_OUT,
    *,
    dividend_yield: float = 0.0,
    rebate: float = 0.0,
    paths: int = 100_000,
    steps: int = 252,
    seed: int = 7,
    antithetic: bool = True,
) -> MonteCarloResult:
    """Price a discretely monitored European barrier option under GBM.

    The simulation uses antithetic normal draws by default. Barrier monitoring is
    performed at every simulated time step, including the initial spot
    """

    spot = require_finite("spot", spot)
    strike = require_finite("strike", strike)
    barrier = require_finite("barrier", barrier)
    time_to_expiry = require_finite("time_to_expiry", time_to_expiry)
    rate = require_finite("rate", rate)
    volatility = require_finite("volatility", volatility)
    dividend_yield = require_finite("dividend_yield", dividend_yield)
    rebate = require_finite("rebate", rebate)
    option_type = coerce_option_type(option_type)
    barrier_type = BarrierType(barrier_type)

    if spot <= 0.0 or strike <= 0.0 or barrier <= 0.0:
        raise ValueError("spot, strike, and barrier must be positive")
    if time_to_expiry <= 0.0:
        raise ValueError("time_to_expiry must be positive")
    if volatility < 0.0:
        raise ValueError("volatility must be nonnegative")
    if paths < 2:
        raise ValueError("paths must be at least two")
    if steps < 1:
        raise ValueError("steps must be positive")
    if rebate < 0.0:
        raise ValueError("rebate must be nonnegative")

    simulated_paths = paths if not antithetic else (paths + 1) // 2
    rng = np.random.default_rng(seed)
    dt = time_to_expiry / steps
    drift = (rate - dividend_yield - 0.5 * volatility**2) * dt
    diffusion = volatility * math.sqrt(dt)

    log_spot = np.full(simulated_paths, math.log(spot), dtype=float)
    running_max = np.full(simulated_paths, spot, dtype=float)
    running_min = np.full(simulated_paths, spot, dtype=float)
    if antithetic:
        anti_log_spot = log_spot.copy()
        anti_max = running_max.copy()
        anti_min = running_min.copy()

    for _ in range(steps):
        draws = rng.standard_normal(simulated_paths)
        log_spot += drift + diffusion * draws
        current = np.exp(log_spot)
        running_max = np.maximum(running_max, current)
        running_min = np.minimum(running_min, current)
        if antithetic:
            anti_log_spot += drift - diffusion * draws
            anti_current = np.exp(anti_log_spot)
            anti_max = np.maximum(anti_max, anti_current)
            anti_min = np.minimum(anti_min, anti_current)

    terminal = np.exp(log_spot)
    maxima = running_max
    minima = running_min
    if antithetic:
        terminal = np.concatenate([terminal, np.exp(anti_log_spot)])[:paths]
        maxima = np.concatenate([maxima, anti_max])[:paths]
        minima = np.concatenate([minima, anti_min])[:paths]
    else:
        terminal = terminal[:paths]
        maxima = maxima[:paths]
        minima = minima[:paths]

    if option_type is OptionType.CALL:
        vanilla_payoff = np.maximum(terminal - strike, 0.0)
    else:
        vanilla_payoff = np.maximum(strike - terminal, 0.0)

    hit = maxima >= barrier if barrier_type.value.startswith("up") else minima <= barrier
    active = hit if barrier_type.value.endswith("in") else ~hit
    payoff = np.where(active, vanilla_payoff, rebate)
    discounted = math.exp(-rate * time_to_expiry) * payoff
    price = float(np.mean(discounted))
    if antithetic and paths >= 4:
        base_count = (paths + 1) // 2
        pair_count = paths // 2
        pair_means = 0.5 * (
            discounted[:pair_count] + discounted[base_count : base_count + pair_count]
        )
        standard_error = float(np.std(pair_means, ddof=1) / math.sqrt(len(pair_means)))
    else:
        standard_error = float(np.std(discounted, ddof=1) / math.sqrt(len(discounted)))
    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_lower_95=max(price - 1.96 * standard_error, 0.0),
        confidence_upper_95=price + 1.96 * standard_error,
        path_count=len(discounted),
        step_count=steps,
        seed=seed,
        barrier_type=barrier_type.value,
        barrier=barrier,
        knock_probability=float(np.mean(hit)),
        antithetic=antithetic,
    )
