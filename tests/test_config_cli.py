import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vol_platform.cli import app
from vol_platform.config import load_config

runner = CliRunner()


def test_load_base_config() -> None:
    config = load_config("configs/base.yml")
    assert config.universe["symbols"] == ["SPY"]
    assert config.pricing.default_model == "black_scholes"
    assert config.pricing.iv.maximum_volatility == 10.0


def test_load_config_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(
        """
project: {name: test}
universe: {symbols: [QQQ]}
pricing:
  default_model: black_76
  default_rate: 0.03
  default_dividend_yield: 0.0
  iv: {}
paths: {raw_data: data/raw}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("VOL_PLATFORM_CONFIG_PATH", str(path))
    config = load_config()
    assert config.pricing.default_model == "black_76"
    assert config.universe["symbols"] == ["QQQ"]


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("configs/does-not-exist.yml")


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_price_cli_black_scholes() -> None:
    result = runner.invoke(
        app,
        [
            "price",
            "--underlying",
            "100",
            "--strike",
            "100",
            "--time",
            "1",
            "--rate",
            "0.05",
            "--vol",
            "0.2",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["price"] == pytest.approx(10.4505835722)
    assert set(payload["greeks"]) == {"delta", "gamma", "vega", "theta"}


def test_price_cli_black76() -> None:
    result = runner.invoke(
        app,
        [
            "price",
            "--underlying",
            "103",
            "--strike",
            "100",
            "--time",
            "0.5",
            "--rate",
            "0.03",
            "--vol",
            "0.25",
            "--model",
            "black_76",
            "--type",
            "put",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["price"] > 0.0


def test_implied_vol_cli_success() -> None:
    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--price",
            "10.450583572185565",
            "--underlying",
            "100",
            "--strike",
            "100",
            "--time",
            "1",
            "--rate",
            "0.05",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["volatility"] == pytest.approx(0.2, abs=1e-9)
    assert payload["status"] == "success"


def test_implied_vol_cli_failure_has_nonzero_exit() -> None:
    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--price",
            "200",
            "--underlying",
            "100",
            "--strike",
            "100",
            "--time",
            "1",
            "--rate",
            "0.05",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.output)["status"] == "price_above_upper_bound"


def test_monte_carlo_barrier_cli() -> None:
    result = runner.invoke(
        app,
        [
            "monte-carlo-barrier",
            "--spot",
            "100",
            "--strike",
            "100",
            "--barrier",
            "125",
            "--time",
            "0.5",
            "--rate",
            "0.04",
            "--vol",
            "0.25",
            "--paths",
            "2000",
            "--steps",
            "16",
            "--seed",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["price"] >= 0.0
    assert payload["path_count"] == 2000
    assert 0.0 <= payload["knock_probability"] <= 1.0
