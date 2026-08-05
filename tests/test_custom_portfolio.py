"""Focused tests for the individual-stock and crypto portfolio simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts import run_part_b
from src import custom_portfolio, features

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_part_b_build_writes_the_asset_artifact_to_the_app_contract_path() -> None:
    assert run_part_b.INVESTABLE_ASSET_RETURNS_PATH == (
        PROJECT_ROOT / custom_portfolio.INVESTABLE_ASSET_RELATIVE_PATH
    )


def _synthetic_asset_returns() -> pd.DataFrame:
    rows = []
    returns = {
        "AAA": [0.10, -0.05, 0.02],
        "BTC-USD": [0.20, 0.10, -0.10],
    }
    for ticker, values in returns.items():
        for date, daily_return in zip(
            pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]),
            values,
            strict=True,
        ):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "asset_group": "Stock" if ticker == "AAA" else "Crypto",
                    "sector": "Tech" if ticker == "AAA" else "Crypto",
                    "daily_return": daily_return,
                }
            )
    return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)


def test_actual_asset_return_artifact_has_common_complete_sample() -> None:
    actual = custom_portfolio.load_investable_asset_returns(PROJECT_ROOT)

    assert tuple(actual.columns) == custom_portfolio.INVESTABLE_ASSET_COLUMNS
    assert actual["ticker"].nunique() == 60
    assert set(actual["asset_group"]) == {"Stock", "Crypto"}
    assert not actual.duplicated(["ticker", "date"]).any()
    summary = actual.groupby("ticker")["date"].agg(["min", "max", "size"])
    assert len(summary.drop_duplicates()) == 1
    assert summary.iloc[0].to_dict() == {
        "min": pd.Timestamp("2020-01-03"),
        "max": pd.Timestamp("2023-12-29"),
        "size": 1005,
    }
    assert np.isfinite(actual["daily_return"]).all()


def test_crypto_return_is_calculated_on_native_calendar_before_alignment() -> None:
    equity_prices = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "date": pd.to_datetime(["2023-01-06", "2023-01-09", "2023-01-10"]),
            "adjClose": [100.0, 110.0, 121.0],
            "sector": ["Tech"] * 3,
        }
    )
    crypto_prices = pd.DataFrame(
        {
            "ticker": ["BTC-USD"] * 5,
            "date": pd.to_datetime(
                ["2023-01-06", "2023-01-07", "2023-01-08", "2023-01-09", "2023-01-10"]
            ),
            "adjClose": [100.0, 110.0, 121.0, 133.1, 119.79],
        }
    )
    equity_returns = features.daily_returns(equity_prices, asset_class="equity")
    crypto_returns = features.daily_returns(crypto_prices, asset_class="crypto")
    aligned = features.combined_returns_panel(equity_returns, crypto_returns)

    monday = aligned.loc[aligned["date"].eq(pd.Timestamp("2023-01-09"))].iloc[0]
    assert monday["AAA"] == pytest.approx(0.10)
    assert monday["BTC-USD"] == pytest.approx(0.10)
    assert monday["BTC-USD"] != pytest.approx(133.1 / 100.0 - 1.0)


def test_custom_portfolio_matches_hand_calculation_and_metrics() -> None:
    returns = _synthetic_asset_returns()
    scenario = custom_portfolio.calculate_custom_portfolio(
        returns,
        {"AAA": 0.60, "BTC-USD": 0.40},
        10_000.0,
    )

    expected_daily = np.array([0.14, 0.01, -0.028])
    expected_growth = np.cumprod(1.0 + expected_daily)
    expected_volatility = expected_daily.std(ddof=1) * np.sqrt(252)
    expected_sharpe = expected_daily.mean() * 252 / expected_volatility
    expected_peak = np.maximum.accumulate(np.concatenate(([1.0], expected_growth)))[1:]
    expected_drawdown = expected_growth / expected_peak - 1.0

    np.testing.assert_allclose(scenario.history["daily_return"], expected_daily)
    np.testing.assert_allclose(scenario.history["growth_of_1"], expected_growth)
    np.testing.assert_allclose(
        scenario.history["portfolio_value"], 10_000.0 * expected_growth
    )
    assert scenario.ending_value == pytest.approx(10_000.0 * expected_growth[-1])
    assert scenario.annualised_return == pytest.approx(expected_growth[-1] ** (252 / 3) - 1)
    assert scenario.annualised_volatility == pytest.approx(expected_volatility)
    assert scenario.sharpe_ratio == pytest.approx(expected_sharpe)
    assert scenario.maximum_drawdown == pytest.approx(expected_drawdown.min())
    assert scenario.asset_mix.set_index("asset_group")["weight"].to_dict() == {
        "Stock": pytest.approx(0.60),
        "Crypto": pytest.approx(0.40),
    }


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"AAA": 0.5}, "between 2 and 15"),
        ({"AAA": 0.7, "BTC-USD": 0.4}, "sum to one"),
        ({"AAA": 1.1, "BTC-USD": -0.1}, "between zero and one"),
        ({"AAA": np.nan, "BTC-USD": 1.0}, "finite"),
    ],
)
def test_custom_portfolio_rejects_invalid_weights(
    weights: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        custom_portfolio.calculate_custom_portfolio(
            _synthetic_asset_returns(), weights, 10_000.0
        )
