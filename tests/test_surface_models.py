import polars as pl

from vol_platform.surface.evaluation import evaluate_fits
from vol_platform.surface.features import build_implied_volatility_dataset
from vol_platform.surface.forward import estimate_forwards
from vol_platform.surface.models import fit_all_smiles
from vol_platform.surface.pipeline import _add_rate_and_expiration_features
from vol_platform.surface.synthetic import synthetic_clean_chain, synthetic_rate_curve


def _iv_data() -> pl.DataFrame:
    enriched = _add_rate_and_expiration_features(
        synthetic_clean_chain(),
        synthetic_rate_curve(),
        timezone="America/New_York",
        day_count_basis=365.0,
        default_rate=0.04,
    )
    forwards, _ = estimate_forwards(enriched)
    return build_implied_volatility_dataset(enriched, forwards)


def test_spline_and_svi_fit_all_weighting_schemes() -> None:
    iv_data = _iv_data()
    fits = fit_all_smiles(iv_data)
    details, comparison = evaluate_fits(iv_data, fits)

    assert len(fits) == 24
    assert sum(fit.success for fit in fits) == 24
    assert details.height == 24
    assert comparison.height == 8
    assert set(comparison["model"]) == {"cubic_spline", "svi"}
    assert set(comparison["weighting"]) == {"equal", "vega", "spread", "quote_quality"}
    assert comparison["failed_fit_rate"].max() == 0.0
    assert comparison["average_coverage"].min() == 1.0
    assert comparison["average_rmse"].max() < 0.02
