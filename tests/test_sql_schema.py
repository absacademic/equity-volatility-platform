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
        "dividends",
        "events",
        "implied_volatilities",
        "option_quotes",
        "rate_curve",
        "underlying_prices",
    }
