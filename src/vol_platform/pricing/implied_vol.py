# Scalar implied-volatility inversion with explicit failure statuses

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from scipy.optimize import brentq

from vol_platform.pricing import black76, black_scholes
from vol_platform.pricing._validation import coerce_option_type, require_finite
from vol_platform.pricing.bounds import black76_bounds, black_scholes_bounds
from vol_platform.pricing.greeks import black76_greeks, black_scholes_greeks
from vol_platform.types import OptionType, PricingModel


class IVStatus(StrEnum):
    # Outcome codes returned by the implied-volatility solver

    SUCCESS = "success"
    AT_LOWER_BOUND = "at_lower_bound"
    AT_UPPER_BOUND = "at_upper_bound"
    PRICE_BELOW_LOWER_BOUND = "price_below_lower_bound"
    PRICE_ABOVE_UPPER_BOUND = "price_above_upper_bound"
    EXPIRED = "expired"
    INVALID_INPUT = "invalid_input"
    NON_CONVERGENCE = "non_convergence"


class IVMethod(StrEnum):
    # Gives numerical method that produced the first result

    NEWTON = "newton"
    BRENTQ = "brentq"
    BISECTION = "bisection"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    # Implied-volatility result

    volatility: float | None
    status: IVStatus
    method: IVMethod
    iterations: int
    residual: float | None
    message: str

    @property
    def converged(self) -> bool:
        # Does the result contain a useable implied-volatility?
        return self.status in {IVStatus.SUCCESS, IVStatus.AT_LOWER_BOUND}


def _invalid(message: str) -> ImpliedVolResult:
    return ImpliedVolResult(
        volatility=None,
        status=IVStatus.INVALID_INPUT,
        method=IVMethod.NONE,
        iterations=0,
        residual=None,
        message=message,
    )


def solve_implied_volatility(
    option_price: float,
    underlying: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: OptionType | str,
    model: PricingModel | str = PricingModel.BLACK_SCHOLES,
    dividend_yield: float = 0.0,
    *,
    initial_volatility: float = 0.20,
    minimum_volatility: float = 1e-8,
    maximum_volatility: float = 10.0,
    price_tolerance: float = 1e-10,
    volatility_tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> ImpliedVolResult:
    # Recover implied volatility from a European option price
    # Validates no-arbitrage bounds, attempts Newton iterations
    # Falls back to Brent's method, then to pure bisection
    # Expected failures returned as statuses

    try:
        option_price = require_finite("option_price", option_price)
        underlying = require_finite("underlying", underlying)
        strike = require_finite("strike", strike)
        time_to_expiry = require_finite("time_to_expiry", time_to_expiry)
        rate = require_finite("rate", rate)
        dividend_yield = require_finite("dividend_yield", dividend_yield)
        initial_volatility = require_finite("initial_volatility", initial_volatility)
        minimum_volatility = require_finite("minimum_volatility", minimum_volatility)
        maximum_volatility = require_finite("maximum_volatility", maximum_volatility)
        price_tolerance = require_finite("price_tolerance", price_tolerance)
        volatility_tolerance = require_finite("volatility_tolerance", volatility_tolerance)
        option_type = coerce_option_type(option_type)
        model = PricingModel(model)
    except (TypeError, ValueError) as exc:
        return _invalid(str(exc))

    if option_price < 0.0:
        return _invalid("option_price cannot be negative")
    if underlying <= 0.0 or strike <= 0.0:
        return _invalid("underlying and strike must be positive")
    if time_to_expiry < 0.0:
        return _invalid("time_to_expiry cannot be negative")
    if minimum_volatility <= 0.0 or maximum_volatility <= minimum_volatility:
        return _invalid("volatility bracket must satisfy 0 < minimum < maximum")
    if initial_volatility <= 0.0:
        return _invalid("initial_volatility must be positive")
    if price_tolerance <= 0.0 or volatility_tolerance <= 0.0:
        return _invalid("solver tolerances must be positive")
    if max_iterations < 1:
        return _invalid("max_iterations must be positive")
    if time_to_expiry == 0.0:
        return ImpliedVolResult(
            volatility=None,
            status=IVStatus.EXPIRED,
            method=IVMethod.NONE,
            iterations=0,
            residual=None,
            message="implied volatility is not identifiable at expiry",
        )

    if model is PricingModel.BLACK_SCHOLES:
        lower, upper = black_scholes_bounds(
            underlying, strike, time_to_expiry, rate, option_type, dividend_yield
        )

        def price_function(volatility: float) -> float:
            return black_scholes.price(
                underlying,
                strike,
                time_to_expiry,
                rate,
                volatility,
                option_type,
                dividend_yield,
            )

        def vega_function(volatility: float) -> float:
            return black_scholes_greeks(
                underlying,
                strike,
                time_to_expiry,
                rate,
                volatility,
                option_type,
                dividend_yield,
            ).vega

    else:
        lower, upper = black76_bounds(underlying, strike, time_to_expiry, rate, option_type)

        def price_function(volatility: float) -> float:
            return black76.price(
                underlying,
                strike,
                time_to_expiry,
                rate,
                volatility,
                option_type,
            )

        def vega_function(volatility: float) -> float:
            return black76_greeks(
                underlying,
                strike,
                time_to_expiry,
                rate,
                volatility,
                option_type,
            ).vega

    if option_price < lower - price_tolerance:
        return ImpliedVolResult(
            None,
            IVStatus.PRICE_BELOW_LOWER_BOUND,
            IVMethod.NONE,
            0,
            option_price - lower,
            f"price {option_price:.12g} is blow lower bound {lower:.12g}",
        )
    if option_price > upper + price_tolerance:
        return ImpliedVolResult(
            None,
            IVStatus.PRICE_ABOVE_UPPER_BOUND,
            IVMethod.NONE,
            0,
            option_price - upper,
            f"price {option_price:.12g} is above upper bound {upper:.12g}",
        )
    if abs(option_price - lower) <= price_tolerance:
        return ImpliedVolResult(
            0.0,
            IVStatus.AT_LOWER_BOUND,
            IVMethod.NONE,
            0,
            option_price - lower,
            "price is at the zero-volatility lower bound",
        )
    if abs(option_price - upper) <= price_tolerance:
        return ImpliedVolResult(
            None,
            IVStatus.AT_UPPER_BOUND,
            IVMethod.NONE,
            0,
            option_price - upper,
            "price is at the model upper bound; no finite volatility exists",
        )

    target = min(max(option_price, lower), upper)

    def objective(volatility: float) -> float:
        return price_function(volatility) - target

    # Newton: fast for regular quotes, not used when vega is flat
    # or a proposed step leaves allowed volatility bracket

    sigma = min(max(initial_volatility, minimum_volatility), maximum_volatility)
    for iteration in range(1, min(max_iterations, 25) + 1):
        residual = objective(sigma)
        if abs(residual) <= price_tolerance:
            return ImpliedVolResult(
                sigma,
                IVStatus.SUCCESS,
                IVMethod.NEWTON,
                iteration,
                residual,
                "Newton iteration converged",
            )
        try:
            vega = vega_function(sigma)
        except ValueError:
            break
        if not math.isfinite(vega) or abs(vega) < 1e-12:
            break
        candidate = sigma - residual / vega
        if not math.isfinite(candidate) or not (
            minimum_volatility < candidate < maximum_volatility
        ):
            break
        if abs(candidate - sigma) <= volatility_tolerance:
            final_residual = objective(candidate)
            if abs(final_residual) <= price_tolerance:
                return ImpliedVolResult(
                    candidate,
                    IVStatus.SUCCESS,
                    IVMethod.NEWTON,
                    iteration,
                    final_residual,
                    "Newton iteration converged",
                )
            break
        sigma = candidate

    f_low = objective(minimum_volatility)
    f_high = objective(maximum_volatility)
    if f_low > 0.0 or f_high < 0.0:
        return ImpliedVolResult(
            None,
            IVStatus.NON_CONVERGENCE,
            IVMethod.NONE,
            0,
            min(abs(f_low), abs(f_high)),
            "configured volatiity bracket does not have the root",
        )

    try:
        root, details = brentq(
            objective,
            minimum_volatility,
            maximum_volatility,
            xtol=volatility_tolerance,
            rtol=max(4.0 * float.fromhex("0x1.0000000000000p-52"), volatility_tolerance),
            maxiter=max_iterations,
            full_output=True,
            disp=False,
        )
        residual = objective(root)
        if details.converged and abs(residual) <= max(price_tolerance, 1e-8):
            return ImpliedVolResult(
                float(root),
                IVStatus.SUCCESS,
                IVMethod.BRENTQ,
                int(details.iterations),
                float(residual),
                "Brent root finder converged",
            )
    except (RuntimeError, ValueError, OverflowError):
        pass

    # Dependency-indep. option
    low = minimum_volatility
    high = maximum_volatility
    f_low = objective(low)
    for iteration in range(1, max_iterations + 1):
        midpoint = 0.5 * (low + high)
        f_mid = objective(midpoint)
        if abs(f_mid) <= price_tolerance or (high - low) <= volatility_tolerance:
            return ImpliedVolResult(
                midpoint,
                IVStatus.SUCCESS,
                IVMethod.BISECTION,
                iteration,
                f_mid,
                "Bisection fallback converged",
            )
        if f_low * f_mid <= 0.0:
            high = midpoint
        else:
            low = midpoint
            f_low = f_mid

    midpoint = 0.5 * (low + high)
    return ImpliedVolResult(
        None,
        IVStatus.NON_CONVERGENCE,
        IVMethod.BISECTION,
        max_iterations,
        objective(midpoint),
        "all implied-volatility methods failed to converge",
    )
