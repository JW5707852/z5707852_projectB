"""Independent performance calculations and fund-artifact schema tests."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from scripts.run_part_b import BaseFundBuild, build_base_funds, write_fund_artifacts
from src import metrics, portfolios

METRIC_TOLERANCE = 1e-12


def _synthetic_artifact_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2023-01-03", periods=4)
    returns = pd.DataFrame(
        {
            "fund": "synthetic_equal_weight",
            "asset_family": "synthetic",
            "method": "equal_weight",
            "date": dates,
            "decision_date": pd.Timestamp("2022-12-30"),
            "daily_return": [0.01, -0.02, 0.03, 0.005],
        }
    )
    weights = pd.DataFrame(
        {
            "fund": ["synthetic_equal_weight"] * 4,
            "asset_family": ["synthetic"] * 4,
            "method": ["equal_weight"] * 4,
            "decision_date": pd.to_datetime(
                ["2022-11-30", "2022-11-30", "2022-12-30", "2022-12-30"]
            ),
            "ticker": ["A", "B", "A", "B"],
            "target_weight": [0.5, 0.5, 0.5, 0.5],
        }
    )
    return returns, weights


@pytest.mark.parametrize(
    "daily_returns",
    [
        pytest.param(np.array([0.0, 0.0, 0.0]), id="zero_returns"),
        pytest.param(np.array([0.01, 0.01, 0.01]), id="constant_positive"),
        pytest.param(np.array([0.10, -0.20, 0.05]), id="known_peak_trough"),
        pytest.param(np.array([-0.10, 0.20]), id="initial_loss"),
    ],
)
def test_golden_metric_series(daily_returns: np.ndarray) -> None:
    dates = pd.bdate_range("2023-01-03", periods=len(daily_returns))
    raw = pd.DataFrame(
        {
            "fund": "golden",
            "asset_family": "synthetic",
            "method": "golden",
            "date": dates,
            "decision_date": pd.Timestamp("2022-12-30"),
            "daily_return": daily_returns,
        }
    )
    prepared = metrics.prepare_fund_returns(raw)
    actual = portfolios.performance_metrics(pd.Series(daily_returns))

    expected_wealth = np.cumprod(1 + daily_returns)
    expected_annual_return = expected_wealth[-1] ** (
        252 / len(daily_returns)
    ) - 1
    expected_volatility = np.std(daily_returns, ddof=1) * np.sqrt(252)
    expected_annual_excess = np.mean(daily_returns) * 252
    expected_sharpe = (
        expected_annual_excess / expected_volatility
        if expected_volatility > 0
        else np.nan
    )
    running_peak = np.maximum.accumulate(
        np.concatenate(([1.0], expected_wealth))
    )[1:]
    expected_drawdown = np.min(expected_wealth / running_peak - 1)

    np.testing.assert_allclose(
        prepared["growth_of_1"],
        expected_wealth,
        rtol=METRIC_TOLERANCE,
        atol=METRIC_TOLERANCE,
    )
    assert actual["annualised_return"] == pytest.approx(
        expected_annual_return,
        rel=METRIC_TOLERANCE,
        abs=METRIC_TOLERANCE,
    )
    assert actual["annualised_volatility"] == pytest.approx(
        expected_volatility,
        rel=METRIC_TOLERANCE,
        abs=METRIC_TOLERANCE,
    )
    if expected_volatility == 0:
        assert np.isnan(actual["sharpe_ratio"])
    else:
        assert actual["sharpe_ratio"] == pytest.approx(
            expected_sharpe,
            rel=METRIC_TOLERANCE,
            abs=METRIC_TOLERANCE,
        )
    assert actual["maximum_drawdown"] == pytest.approx(
        expected_drawdown,
        rel=METRIC_TOLERANCE,
        abs=METRIC_TOLERANCE,
    )


def test_growth_and_performance_match_independent_calculations() -> None:
    raw_returns, raw_weights = _synthetic_artifact_inputs()
    artifacts = metrics.build_fund_artifacts(raw_returns, raw_weights)
    returns = raw_returns["daily_return"]
    wealth = (1 + returns).cumprod()
    expected_annual_return = wealth.iloc[-1] ** (252 / len(returns)) - 1
    expected_volatility = returns.std(ddof=1) * np.sqrt(252)
    expected_annual_excess = returns.mean() * 252
    expected_sharpe = expected_annual_excess / expected_volatility
    expected_drawdown = (
        wealth / wealth.cummax().clip(lower=1.0) - 1
    ).min()
    actual = artifacts.performance_metrics.iloc[0]

    assert artifacts.fund_returns["growth_of_1"].to_numpy() == pytest.approx(wealth)
    assert actual["final_growth_of_1"] == pytest.approx(wealth.iloc[-1])
    assert actual["annualised_return"] == pytest.approx(expected_annual_return)
    assert actual["annualised_volatility"] == pytest.approx(expected_volatility)
    assert actual["annualised_mean_excess_return"] == pytest.approx(
        expected_annual_excess
    )
    assert actual["sharpe_ratio"] == pytest.approx(expected_sharpe)
    assert actual["maximum_drawdown"] == pytest.approx(expected_drawdown)
    assert actual["periods_per_year"] == 252
    assert actual["risk_free_rate_annual"] == 0.0
    assert actual["annual_return_method"] == "geometric"


def test_latest_rebalance_is_the_only_current_holding_set() -> None:
    raw_returns, raw_weights = _synthetic_artifact_inputs()
    artifacts = metrics.build_fund_artifacts(raw_returns, raw_weights)
    weights = artifacts.fund_weights
    current = weights.loc[weights["is_current"]]

    assert set(current["ticker"]) == {"A", "B"}
    assert current["decision_date"].nunique() == 1
    assert current["decision_date"].iloc[0] == pd.Timestamp("2022-12-30")
    assert current["target_weight"].sum() == pytest.approx(1.0, abs=1e-12)
    assert (~weights.loc[weights["decision_date"].eq("2022-11-30"), "is_current"]).all()


def test_artifact_validation_rejects_inconsistent_weight_sum() -> None:
    raw_returns, raw_weights = _synthetic_artifact_inputs()
    raw_weights.loc[
        raw_weights["decision_date"].eq("2022-12-30")
        & raw_weights["ticker"].eq("B"),
        "target_weight",
    ] = 0.4

    with pytest.raises(
        metrics.ArtifactValidationError,
        match="do not sum to one",
    ):
        metrics.build_fund_artifacts(raw_returns, raw_weights)


def test_fund_returns_reject_decisions_on_or_after_the_return_date() -> None:
    returns, _ = _synthetic_artifact_inputs()
    returns.loc[0, "decision_date"] = returns.loc[0, "date"]

    with pytest.raises(
        metrics.ArtifactValidationError,
        match="decision on or after",
    ):
        metrics.prepare_fund_returns(returns)


def test_performance_metrics_require_a_positive_annualisation_factor() -> None:
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        portfolios.performance_metrics(pd.Series([0.01, 0.02]), periods_per_year=0)


@pytest.fixture(scope="module")
def official_build() -> BaseFundBuild:
    return build_base_funds()


def test_official_artifact_schemas_names_and_dates(
    official_build: BaseFundBuild,
) -> None:
    artifacts = official_build.artifacts
    returns = artifacts.fund_returns
    weights = artifacts.fund_weights
    performance = artifacts.performance_metrics
    expected_funds = {
        "combined_equal_weight",
        "combined_min_variance",
        "equity_equal_weight",
    }

    assert list(returns.columns) == [
        "fund",
        "asset_family",
        "method",
        "date",
        "decision_date",
        "daily_return",
        "growth_of_1",
    ]
    assert {
        "fund",
        "asset_family",
        "method",
        "date",
        "decision_date",
        "ticker",
        "target_weight",
        "is_current",
    }.issubset(weights.columns)
    assert {
        "fund",
        "asset_family",
        "method",
        "as_of_date",
        "sample_start_date",
        "sample_end_date",
        "current_holdings_date",
        "observations",
        "periods_per_year",
        "risk_free_rate_annual",
        "final_growth_of_1",
        "annualised_return",
        "annualised_volatility",
        "annualised_mean_excess_return",
        "sharpe_ratio",
        "maximum_drawdown",
    }.issubset(performance.columns)
    assert set(returns["fund"]) == expected_funds
    assert set(weights["fund"]) == expected_funds
    assert set(performance["fund"]) == expected_funds
    assert returns.groupby("fund")["date"].min().eq(pd.Timestamp("2021-01-04")).all()
    assert returns.groupby("fund")["date"].max().eq(pd.Timestamp("2023-12-29")).all()
    assert performance["sample_start_date"].eq(pd.Timestamp("2021-01-04")).all()
    assert performance["sample_end_date"].eq(pd.Timestamp("2023-12-29")).all()
    assert performance["observations"].eq(753).all()


def test_official_current_holdings_match_latest_rebalances(
    official_build: BaseFundBuild,
) -> None:
    weights = official_build.artifacts.fund_weights
    performance = official_build.artifacts.performance_metrics.set_index("fund")
    current = weights.loc[weights["is_current"]]
    current_dates = current.groupby("fund")["decision_date"].unique()
    current_sums = current.groupby("fund")["target_weight"].sum()

    assert np.allclose(current_sums, 1.0, atol=1e-10)
    assert current.groupby("fund")["ticker"].nunique().to_dict() == {
        "combined_equal_weight": 60,
        "combined_min_variance": 60,
        "equity_equal_weight": 50,
    }
    for fund, dates in current_dates.items():
        assert len(dates) == 1
        assert dates[0] == performance.loc[fund, "current_holdings_date"]


def test_three_official_returns_reproduce_from_logged_weights(
    official_build: BaseFundBuild,
) -> None:
    returns = official_build.artifacts.fund_returns
    weights = official_build.artifacts.fund_weights
    fund_returns = returns.loc[
        returns["fund"].eq("combined_min_variance")
    ].reset_index(drop=True)
    asset_returns = official_build.combined_asset_returns.set_index("date")

    for position in (0, len(fund_returns) // 2, len(fund_returns) - 1):
        sample = fund_returns.iloc[position]
        active = weights.loc[
            weights["fund"].eq(sample["fund"])
            & weights["decision_date"].eq(sample["decision_date"])
        ].set_index("ticker")["target_weight"]
        reproduced = float(
            asset_returns.loc[sample["date"], active.index].to_numpy()
            @ active.to_numpy()
        )
        assert reproduced == pytest.approx(sample["daily_return"], abs=1e-15)


def test_csv_round_trip_preserves_artifact_schemas(
    official_build: BaseFundBuild,
    tmp_path: pathlib.Path,
) -> None:
    paths = {
        "fund_returns": tmp_path / "data" / "fund_returns.csv",
        "fund_weights": tmp_path / "data" / "fund_weights.csv",
        "performance_metrics": tmp_path / "tables" / "performance_metrics.csv",
    }
    write_fund_artifacts(official_build, output_paths=paths)

    expected_frames = {
        "fund_returns": official_build.artifacts.fund_returns,
        "fund_weights": official_build.artifacts.fund_weights,
        "performance_metrics": official_build.artifacts.performance_metrics,
    }
    for name, path in paths.items():
        loaded = pd.read_csv(path)
        assert list(loaded.columns) == list(expected_frames[name].columns)
        assert len(loaded) == len(expected_frames[name])
