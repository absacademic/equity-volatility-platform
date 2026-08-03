# Command-line tools for pricing options & solving implied volatility

import json
from dataclasses import asdict
from pathlib import Path

import typer

from vol_platform.data.pipeline import run_ingestion
from vol_platform.event_study.history import build_event_history
from vol_platform.event_study.pipeline import run_event_study
from vol_platform.event_study.synthetic import write_synthetic_event_study_inputs
from vol_platform.pricing import black76, black_scholes
from vol_platform.pricing.greeks import calculate_greeks
from vol_platform.pricing.implied_vol import solve_implied_volatility
from vol_platform.pricing.monte_carlo import BarrierType, price_barrier_option
from vol_platform.strategy.pipeline import run_strategy_backtest
from vol_platform.strategy.synthetic import write_synthetic_week6_inputs
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


@app.command("monte-carlo-barrier")
def monte_carlo_barrier_command(
    spot: float = typer.Option(..., "--spot"),
    strike: float = typer.Option(..., "--strike"),
    barrier: float = typer.Option(..., "--barrier"),
    time_to_expiry: float = typer.Option(..., "--time", help="Years to expiry"),
    rate: float = typer.Option(..., "--rate"),
    volatility: float = typer.Option(..., "--vol"),
    option_type: OptionType = typer.Option(OptionType.CALL, "--type"),
    barrier_type: BarrierType = typer.Option(BarrierType.UP_AND_OUT, "--barrier-type"),
    dividend_yield: float = typer.Option(0.0, "--dividend-yield"),
    rebate: float = typer.Option(0.0, "--rebate"),
    paths: int = typer.Option(100_000, "--paths"),
    steps: int = typer.Option(252, "--steps"),
    seed: int = typer.Option(7, "--seed"),
    antithetic: bool = typer.Option(True, "--antithetic/--no-antithetic"),
) -> None:
    # Price a discretely monitored barrier option under geometric Brownian motion
    result = price_barrier_option(
        spot,
        strike,
        barrier,
        time_to_expiry,
        rate,
        volatility,
        option_type,
        barrier_type,
        dividend_yield=dividend_yield,
        rebate=rebate,
        paths=paths,
        steps=steps,
        seed=seed,
        antithetic=antithetic,
    )
    typer.echo(json.dumps(asdict(result), indent=2))


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


@app.command("synthetic-event-study")
def synthetic_event_study_command(
    output_dir: Path = typer.Option(Path("data/interim/week5-demo"), "--output-dir"),
) -> None:
    # Write deterministic event, underlying, and surface-feature inputs
    events, underlying, features = write_synthetic_event_study_inputs(output_dir)
    typer.echo(
        json.dumps(
            {
                "events": str(events),
                "underlying": str(underlying),
                "surface_features": str(features),
            },
            indent=2,
        )
    )


@app.command("event-study")
def event_study_command(
    daily_features: Path = typer.Argument(..., exists=True, dir_okay=False),
    events_file: Path = typer.Option(..., "--events", exists=True, dir_okay=False),
    underlying_file: Path = typer.Option(..., "--underlying", exists=True, dir_okay=False),
    symbol: str = typer.Option("SPY", "--symbol"),
    config_path: Path = typer.Option(Path("configs/base.yml"), "--config"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    # Run point-in-time event analysis, baseline models, and cost-aware backtests
    result = run_event_study(
        daily_features,
        events_file,
        underlying_file,
        symbol=symbol,
        config_path=config_path,
        output_dir=output_dir,
    )
    typer.echo(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "point_in_time_events": str(result.point_in_time_events),
                "event_windows": str(result.event_windows),
                "event_dataset": str(result.event_dataset),
                "summary_analysis": str(result.summary_analysis),
                "regime_comparison": str(result.regime_comparison),
                "model_coefficients": str(result.model_coefficients),
                "model_performance": str(result.model_performance),
                "coefficient_stability": str(result.coefficient_stability),
                "walk_forward_results": str(result.walk_forward_results),
                "walk_forward_performance": str(result.walk_forward_performance),
                "strategy_backtest": str(result.strategy_backtest),
                "strategy_summary": str(result.strategy_summary),
                "nonlinear_predictions": str(result.nonlinear_predictions),
                "nonlinear_performance": str(result.nonlinear_performance),
                "nonlinear_status": str(result.nonlinear_status),
                "pnl_attribution": str(result.pnl_attribution),
                "conclusion": str(result.conclusion),
                "report": str(result.report),
                "database": str(result.database),
                "plots": [str(path) for path in result.plots],
            },
            indent=2,
        )
    )


@app.command("build-event-history")
def build_event_history_command(
    macro_events_file: Path | None = typer.Option(
        None, "--macro-events", exists=True, dir_okay=False
    ),
    earnings_events_file: Path | None = typer.Option(
        None, "--earnings-events", exists=True, dir_okay=False
    ),
    underlying_file: Path | None = typer.Option(None, "--underlying", exists=True, dir_okay=False),
    large_move_threshold: float = typer.Option(0.03, "--large-move-threshold"),
    market_symbols: str = typer.Option("SPY", "--market-symbols"),
    output_file: Path = typer.Option(
        Path("data/processed/events/combined-event-history.csv"), "--output"
    ),
) -> None:
    # Combine supplied macro and earnings calendars with derived large-move events
    events = build_event_history(
        macro_events_file=macro_events_file,
        earnings_events_file=earnings_events_file,
        underlying_file=underlying_file,
        large_move_threshold=large_move_threshold,
        market_symbols=tuple(
            item.strip().upper() for item in market_symbols.split(",") if item.strip()
        ),
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.suffix.lower() == ".parquet":
        events.write_parquet(output_file)
    else:
        events.write_csv(output_file)
    typer.echo(json.dumps({"event_count": events.height, "output": str(output_file)}, indent=2))


@app.command("synthetic-week6")
def synthetic_week6_command(
    output_dir: Path = typer.Option(Path("data/interim/week6-demo"), "--output-dir"),
) -> None:
    # Write deterministic long-history, multi-asset, contract-level sample inputs
    paths = write_synthetic_week6_inputs(output_dir)
    typer.echo(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


@app.command("strategy-backtest")
def strategy_backtest_command(
    signals_file: Path = typer.Argument(..., exists=True, dir_okay=False),
    option_quotes_file: Path = typer.Option(..., "--option-quotes", exists=True, dir_okay=False),
    underlying_file: Path = typer.Option(..., "--underlying", exists=True, dir_okay=False),
    config_path: Path = typer.Option(Path("configs/base.yml"), "--config"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    # Run exact contract-level option and periodic delta-hedge P&L
    result = run_strategy_backtest(
        signals_file,
        option_quotes_file,
        underlying_file,
        config_path=config_path,
        output_dir=output_dir,
    )
    typer.echo(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "trades": str(result.trades),
                "metrics": str(result.metrics),
                "attribution": str(result.attribution),
                "hedge_log": str(result.hedge_log),
                "rejections": str(result.rejections),
                "sensitivity": str(result.sensitivity),
                "report": str(result.report),
                "database": str(result.database),
                "plots": [str(path) for path in result.plots],
            },
            indent=2,
        )
    )
