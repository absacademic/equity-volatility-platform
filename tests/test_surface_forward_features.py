import polars as pl
import pytest

from vol_platform.surface.features import build_implied_volatility_dataset
from vol_platform.surface.forward import estimate_forwards
from vol_platform.surface.pipeline import _add_rate_and_expiration_features
from vol_platform.surface.synthetic import synthetic_clean_chain, synthetic_rate_curve


def _enriched() -> pl.DataFrame:
    return _add_rate_and_expiration_features(
        synthetic_clean_chain(),
        synthetic_rate_curve(),
        timezone="America/New_York",
        day_count_basis=365.0,
        default_rate=0.04,
    )


def test_forward_estimator_uses_multiple_near_atm_pairs() -> None:
    quotes = _enriched()
    forwards, pairs = estimate_forwards(quotes)
    assert forwards.height == 3
    assert pairs.height == 15
    assert forwards["pair_count"].to_list() == [5, 5, 5]
    assert set(forwards["reliability"].to_list()) <= {"high", "medium"}
    assert max(forwards["relative_dispersion"]) < 0.001


def test_implied_volatility_dataset_has_week_three_features() -> None:
    quotes = _enriched()
    forwards, _ = estimate_forwards(quotes)
    iv_data = build_implied_volatility_dataset(quotes, forwards)
    assert iv_data.height == quotes.height
    assert iv_data["mid_implied_volatility"].null_count() == 0
    assert iv_data["total_variance"].null_count() == 0
    assert iv_data.filter(pl.col("fit_eligible")).height == 27
    assert iv_data["delta"].drop_nulls().len() == iv_data.height
    assert iv_data["forward_moneyness"].abs().min() < 0.02
    assert iv_data["mid_implied_volatility"].mean() == pytest.approx(0.22, abs=0.03)
