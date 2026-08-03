# Compact exploratory and backtest plots for event studies

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def create_event_study_plots(
    dataset: pl.DataFrame,
    coefficients: pl.DataFrame,
    backtest: pl.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if dataset.is_empty():
        return paths

    figure = Figure()
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.scatter(dataset["expected_move"].to_list(), dataset["absolute_return"].to_list())
    maximum = max(dataset["expected_move"].max(), dataset["absolute_return"].max())
    axis.plot([0.0, maximum], [0.0, maximum], linestyle="--")
    axis.set_xlabel("Pre-event expected move")
    axis.set_ylabel("Realized absolute return")
    axis.set_title("Expected versus realized event move")
    figure.tight_layout()
    path = output_dir / "expected_vs_realized.png"
    figure.savefig(path, dpi=160)
    paths.append(path)

    figure = Figure()
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.scatter(
        dataset["surface_dislocation"].fill_null(0.0).to_list(),
        dataset["expected_minus_realized_move"].to_list(),
    )
    axis.axhline(0.0, linestyle="--")
    axis.set_xlabel("Surface dislocation score")
    axis.set_ylabel("Expected minus realized move")
    axis.set_title("Surface dislocation and move estimation error")
    figure.tight_layout()
    path = output_dir / "surface_dislocation.png"
    figure.savefig(path, dpi=160)
    paths.append(path)

    linear = coefficients.filter((pl.col("model") == "linear") & (pl.col("feature") != "intercept"))
    if not linear.is_empty():
        figure = Figure()
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        positions = np.arange(linear.height)
        axis.barh(positions, linear["coefficient"].to_list())
        axis.set_yticks(positions, linear["feature"].to_list())
        axis.axvline(0.0)
        axis.set_xlabel("Standardized coefficient")
        axis.set_title("Baseline linear model coefficients")
        figure.tight_layout()
        path = output_dir / "linear_coefficients.png"
        figure.savefig(path, dpi=160)
        paths.append(path)

    if not backtest.is_empty():
        figure = Figure()
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.plot(
            backtest["event_timestamp"].to_list(),
            backtest["strategy_cumulative_return"].to_list(),
            label="Model strategy",
        )
        axis.plot(
            backtest["event_timestamp"].to_list(),
            backtest["always_short_cumulative_return"].to_list(),
            label="Always short",
        )
        axis.axhline(0.0)
        axis.set_xlabel("Event date")
        axis.set_ylabel("Cumulative return")
        axis.set_title("Cost-adjusted event-strategy backtest")
        axis.legend()
        figure.autofmt_xdate()
        figure.tight_layout()
        path = output_dir / "strategy_backtest.png"
        figure.savefig(path, dpi=160)
        paths.append(path)
    return paths
