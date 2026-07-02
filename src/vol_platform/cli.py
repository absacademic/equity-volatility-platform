# Command-line tools for pricing options & solving implied volatility

import json
from dataclasses import asdict

import typer

from vol_platform.pricing import black76, black_scholes
from vol_platform.pricing.greeks import calculate_greeks
from vol_platform.pricing.implied_vol import solve_implied_volatility
from vol_platform.types import OptionType, PricingModel

app = typer.Typer(
    name="vol-platform",
    help="Option-pricing and implied-volatility tools.",
)


@app.command("price")
def price_command(
    underlying: float = typer.Option(
        ...,
        help="Spot for Black-Scholes; forward for Black-76",
    ),
    strike: float = typer.Option(...),
    time_to_expiry: float = typer.Option(..., "--time", help="Years to expiry"),
    rate: float = typer.Option(...),
    volatility: float = typer.Option(..., "--vol"),
    option_type: OptionType = typer.Option(OptionType.CALL, "--type"),
    model: PricingModel = typer.Option(PricingModel.BLACK_SCHOLES),
    dividend_yield: float = typer.Option(0.0, "--dividend-yield"),
) -> None:
    # Price a single option and print core greeks
    if model is PricingModel.BLACK_SCHOLES:
        option_price = black_scholes.price(
            underlying,
            strike,
            time_to_expiry,
            rate,
            volatility,
            option_type,
            dividend_yield,
        )
    else:
        option_price = black76.price(
            underlying,
            strike,
            time_to_expiry,
            rate,
            volatility,
            option_type,
        )

    greek_values = calculate_greeks(
        model,
        underlying,
        strike,
        time_to_expiry,
        rate,
        volatility,
        option_type,
        dividend_yield,
    )

    output = {"price": option_price, "greeks": asdict(greek_values)}
    typer.echo(json.dumps(output, indent=2))  # JSON string


@app.command("implied-vol")
def implied_vol_command(
    option_price: float = typer.Option(..., "--price"),
    underlying: float = typer.Option(
        ...,
        help="Spot for Black-Scholes; forward for Black-76.",
    ),
    strike: float = typer.Option(...),
    time_to_expiry: float = typer.Option(..., "--time"),
    rate: float = typer.Option(...),
    option_type: OptionType = typer.Option(OptionType.CALL, "--type"),
    model: PricingModel = typer.Option(PricingModel.BLACK_SCHOLES),
    dividend_yield: float = typer.Option(0.0, "--dividend-yield"),
) -> None:
    # Recover a single implied volatility and print solver diagnostics
    result = solve_implied_volatility(
        option_price,
        underlying,
        strike,
        time_to_expiry,
        rate,
        option_type,
        model,
        dividend_yield,
    )

    typer.echo(json.dumps(asdict(result), indent=2, default=str))
    if not result.converged:
        raise typer.Exit(code=1)
