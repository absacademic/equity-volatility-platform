# Static diagnostic plots for smiles and surfaces

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from vol_platform.surface.evaluation import best_fits_by_expiration_and_model
from vol_platform.surface.models import SmileFit


def _new_figure() -> tuple[Figure, object]:
    figure = Figure()
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    figure.clear()
    return path


def create_surface_plots(
    iv_data: pl.DataFrame,
    details: pl.DataFrame,
    fits: list[SmileFit],
    output_dir: Path,
) -> list[Path]:
    # Create smile, residual, surface, bid-ask, and ATM term-structure plots

    paths: list[Path] = []
    eligible = iv_data.filter(pl.col("fit_eligible"))
    if eligible.is_empty():
        return paths
    selected = best_fits_by_expiration_and_model(details, fits)
    first_expiration = eligible["expiration"].unique().sort()[0]
    frame = eligible.filter(pl.col("expiration") == first_expiration).sort("forward_moneyness")
    x = np.asarray(frame["forward_moneyness"], dtype=float)
    observed = np.asarray(frame["mid_implied_volatility"], dtype=float)
    time_to_expiry = float(frame["time_to_expiry"].median())
    grid = np.linspace(float(x.min()), float(x.max()), 151)

    figure, axis = _new_figure()
    axis.scatter(x, observed, label="mid IV")
    for model in ("cubic_spline", "svi"):
        fit = selected.get((first_expiration, model))
        if fit is not None:
            fitted_iv = np.sqrt(np.maximum(fit.predict_total_variance(grid), 0.0) / time_to_expiry)
            axis.plot(grid, fitted_iv, label=f"{model}: {fit.weighting}")
    axis.set_xlabel("Log forward moneyness, log(K/F)")
    axis.set_ylabel("Implied volatility")
    axis.set_title(f"Smile: {first_expiration}")
    axis.legend()
    paths.append(_save(figure, output_dir / "smile.png"))

    figure, axis = _new_figure()
    for model in ("cubic_spline", "svi"):
        fit = selected.get((first_expiration, model))
        if fit is not None:
            fitted_iv = np.sqrt(np.maximum(fit.predict_total_variance(x), 0.0) / time_to_expiry)
            axis.scatter(x, fitted_iv - observed, label=f"{model}: {fit.weighting}")
    axis.axhline(0.0)
    axis.set_xlabel("Log forward moneyness, log(K/F)")
    axis.set_ylabel("Fitted IV minus observed IV")
    axis.set_title(f"Smile residuals: {first_expiration}")
    axis.legend()
    paths.append(_save(figure, output_dir / "residuals.png"))

    band = iv_data.filter(
        (pl.col("expiration") == first_expiration)
        & pl.col("is_otm")
        & pl.col("bid_implied_volatility").is_not_null()
        & pl.col("ask_implied_volatility").is_not_null()
    ).sort("forward_moneyness")
    if not band.is_empty():
        band_x = np.asarray(band["forward_moneyness"], dtype=float)
        bid = np.asarray(band["bid_implied_volatility"], dtype=float)
        ask = np.asarray(band["ask_implied_volatility"], dtype=float)
        mid = np.asarray(band["mid_implied_volatility"], dtype=float)
        figure, axis = _new_figure()
        axis.fill_between(band_x, bid, ask, alpha=0.25, label="bid-ask IV band")
        axis.plot(band_x, mid, marker="o", label="mid IV")
        axis.set_xlabel("Log forward moneyness, log(K/F)")
        axis.set_ylabel("Implied volatility")
        axis.set_title(f"Bid-ask volatility band: {first_expiration}")
        axis.legend()
        paths.append(_save(figure, output_dir / "bid_ask_band.png"))

    surface_x: list[float] = []
    surface_t: list[float] = []
    surface_iv: list[float] = []
    for expiration in eligible["expiration"].unique().sort():
        expiry_frame = eligible.filter(pl.col("expiration") == expiration)
        fit = selected.get((expiration, "svi")) or selected.get((expiration, "cubic_spline"))
        if fit is None:
            continue
        local_x = np.linspace(
            float(expiry_frame["forward_moneyness"].min()),
            float(expiry_frame["forward_moneyness"].max()),
            41,
        )
        maturity = float(expiry_frame["time_to_expiry"].median())
        local_iv = np.sqrt(np.maximum(fit.predict_total_variance(local_x), 0.0) / maturity)
        surface_x.extend(local_x.tolist())
        surface_t.extend([maturity] * local_x.size)
        surface_iv.extend(local_iv.tolist())
    if len(set(surface_t)) >= 2:
        figure = Figure()
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111, projection="3d")
        axis.plot_trisurf(surface_x, surface_t, surface_iv, linewidth=0.2)
        axis.set_xlabel("Log forward moneyness")
        axis.set_ylabel("Time to expiry")
        axis.set_zlabel("Implied volatility")
        axis.set_title("Implied-volatility surface")
        paths.append(_save(figure, output_dir / "surface.png"))

    atm_rows = []
    for expiry_frame in eligible.partition_by("expiration", maintain_order=True):
        atm_rows.append(expiry_frame.sort(pl.col("forward_moneyness").abs()).row(0, named=True))
    atm = pl.DataFrame(atm_rows).sort("time_to_expiry")
    figure, axis = _new_figure()
    axis.plot(
        np.asarray(atm["time_to_expiry"], dtype=float),
        np.asarray(atm["mid_implied_volatility"], dtype=float),
        marker="o",
    )
    axis.set_xlabel("Time to expiry (years)")
    axis.set_ylabel("ATM implied volatility")
    axis.set_title("ATM term structure")
    paths.append(_save(figure, output_dir / "atm_term_structure.png"))
    return paths
