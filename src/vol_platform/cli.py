# Command-line tools for pricing options & solving implied volatility

import json
from dataclasses import asdict
from pathlib import Path

import typer

from vol_platform.data.pipeline import run_ingestion
from vol_platform.pricing import black76, black_scholes
from vol_platform.pricing.greeks import calculate_greeks
from vol_platform.pricing.implied_vol import solve_implied_volatility
from vol_platform.surface.pipeline import run_surface_analysis
from vol_platform.surface.synthetic import write_synthetic_inputs
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


@app.command("ingest")
def ingest_command(
    input_file: Path = typer.Argument(..., exists=True, dir_okay=False),
    underlying_file: Path | None = typer.Option(None, "--underlying", exists=True, dir_okay=False),
    rates_file: Path | None = typer.Option(None, "--rates", exists=True, dir_okay=False),
    events_file: Path | None = typer.Option(None, "--events", exists=True, dir_okay=False),
    config_path: Path = typer.Option(Path("configs/base.yml"), "--config"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    database_path: Path | None = typer.Option(None, "--database"),
    source: str = typer.Option("local_csv", "--source"),
) -> None:
    # Convert raw option CSV data into clean and rejected analytical datasets

    result = run_ingestion(
        input_file,
        underlying_file=underlying_file,
        rates_file=rates_file,
        events_file=events_file,
        config_path=config_path,
        output_dir=output_dir,
        database_path=database_path,
        source=source,
    )
    typer.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "input_rows": result.input_rows,
                "clean_rows": result.clean_rows,
                "rejected_rows": result.rejected_rows,
                "output_dir": str(result.output_dir),
                "database": str(result.database),
                "report": str(result.report),
                "manifest": str(result.manifest),
            },
            indent=2,
        )
    )


@app.command("surface")
def surface_command(
    clean_quotes: Path = typer.Argument(..., exists=True),
    rates_file: Path | None = typer.Option(None, "--rates", exists=True, dir_okay=False),
    dividends_file: Path | None = typer.Option(None, "--dividends", exists=True, dir_okay=False),
    events_file: Path | None = typer.Option(None, "--events", exists=True, dir_okay=False),
    underlying_history_file: Path | None = typer.Option(
        None, "--underlying-history", exists=True, dir_okay=False
    ),
    quote_date: str | None = typer.Option(None, "--date"),
    config_path: Path = typer.Option(Path("configs/base.yml"), "--config"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    # Build current surfaces plus point-in-time historical features
    result = run_surface_analysis(
        clean_quotes,
        rates_file=rates_file,
        dividends_file=dividends_file,
        events_file=events_file,
        underlying_history_file=underlying_history_file,
        quote_date=quote_date,
        config_path=config_path,
        output_dir=output_dir,
    )
    typer.echo(
        json.dumps(
            {
                "quote_date": str(result.quote_date),
                "input_rows": result.input_rows,
                "iv_rows": result.iv_rows,
                "successful_fits": result.successful_fits,
                "failed_fits": result.failed_fits,
                "output_dir": str(result.output_dir),
                "implied_volatility_dataset": str(result.implied_volatility_dataset),
                "forward_summary": str(result.forward_summary),
                "model_comparison": str(result.model_comparison),
                "report": str(result.report),
                "arbitrage_report": str(result.arbitrage_report),
                "arbitrage_diagnostics": str(result.arbitrage_diagnostics),
                "surface_adjustments": str(result.surface_adjustments),
                "standardized_delta_points": str(result.standardized_delta_points),
                "daily_feature_table": str(result.daily_feature_table),
                "database": str(result.database),
                "plots": [str(path) for path in result.plots],
                "historical_plots": [str(path) for path in result.historical_plots],
            },
            indent=2,
        )
    )


@app.command("synthetic-chain")
def synthetic_chain_comamnd(
    output_dir: Path = typer.Option(Path("data/interim/week4-demo"), "--output-dir"),
) -> None:
    # Write deterministic chain, rate, dividend, event, and history inputs
    chain, rates = write_synthetic_inputs(output_dir)
    typer.echo(
        json.dumps(
            {
                "clean_chain": str(chain),
                "rates": str(rates),
                "dividends": str(output_dir / "synthetic-dividends.csv"),
                "events": str(output_dir / "synthetic-events.csv"),
                "underlying_history": str(output_dir / "synthetic-underlying-history.csv"),
            },
            indent=2,
        )
    )
