from pathlib import Path

import duckdb


def test_duckdb_schema_builds_all_canonical_tables() -> None:
    sql = Path("sql/schema.sql").read_text(encoding="utf-8")
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(sql)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
    finally:
        connection.close()

    assert tables == {
        "arbitrage_diagnostics",
        "daily_volatility_features",
        "dividends",
        "events",
        "event_study_events",
        "event_study_observations",
        "event_strategy_results",
        "forward_estimates",
        "implied_volatilities",
        "option_quotes",
        "rate_curve",
        "smile_model_comparison",
        "standardized_delta_points",
        "surface_adjustments",
        "underlying_prices",
    }
