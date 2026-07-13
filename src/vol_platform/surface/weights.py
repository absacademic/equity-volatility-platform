# Smile-fitting weight schemes

from __future__ import annotations

from enum import StrEnum

import numpy as np
import polars as pl


class Weighting(StrEnum):
    EQUAL = "equal"
    VEGA = "vega"
    SPREAD = "spread"
    QUOTE_QUALITY = "quote_quality"


def build_weights(frame: pl.DataFrame, weighting: Weighting | str) -> np.ndarray:
    weighting = Weighting(weighting)
    if weighting is Weighting.EQUAL:
        raw = np.ones(frame.height, dtype=float)
    elif weighting is Weighting.VEGA:
        raw = np.asarray(frame["vega"], dtype=float)
    elif weighting is Weighting.SPREAD:
        spreads = np.asarray(frame["relative_spread"], dtype=float)
        raw = 1.0 / np.maximum(spreads, 1e-4)
    else:
        raw = np.asarray(frame["quote_quality_score"], dtype=float)

    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 1e-8)
    return raw / np.mean(raw)
