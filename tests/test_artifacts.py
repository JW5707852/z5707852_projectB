"""Validation of the actual generated fund CSV contracts and calculations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts import run_part_b
from src import etl, features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETURN_PATH = PROJECT_ROOT / "results/data/fund_returns.csv"
WEIGHT_PATH = PROJECT_ROOT / "results/data/fund_weights.csv"
METRIC_PATH = PROJECT_ROOT / "results/tables/performance_metrics.csv"
EXPLORATORY_SIGNAL_PATH = PROJECT_ROOT / "results/data/sector_sentiment_21d_coverage.csv"
EXPLORATORY_COMPARISON_PATH = PROJECT_ROOT / "results/tables/fusion_exploratory_comparison.csv"
PREDICTABILITY_PATH = PROJECT_ROOT / "results/tables/sentiment_predictability.csv"
PREDICTABILITY_SUMMARY_PATH = PROJECT_ROOT / "results/tables/sentiment_predictability_summary.csv"
ACTIVE_SECTOR_EXPOSURE_PATH = (
    PROJECT_ROOT / "results/data/active_sector_allocation_exposure.csv"
)
GROWTH_SECTOR_EXPOSURE_PATH = (
    PROJECT_ROOT / "results/data/growth_sector_allocation_exposure.csv"
)
AGGRESSIVE_SECTOR_EXPOSURE_PATH = (
    PROJECT_ROOT / "results/data/aggressive_sector_allocation_exposure.csv"
)
EXPLORATORY_PATHS = (
    EXPLORATORY_SIGNAL_PATH,
    EXPLORATORY_COMPARISON_PATH,
    PREDICTABILITY_PATH,
    PREDICTABILITY_SUMMARY_PATH,
)
EXPLORATORY_GENERATED = all(path.exists() for path in EXPLORATORY_PATHS)
NUMERIC_TOLERANCE = 1e-12
WEIGHT_TOLERANCE = 1e-10
EXPECTED_FUNDS = set(run_part_b.EXPECTED_FUND_IDENTITIES)
IDENTIFIERS = ["fund", "asset_family", "method"]

RETURN_COLUMNS = {
    *IDENTIFIERS,
    "date",
    "decision_date",
    "daily_return",
    "growth_of_1",
}
WEIGHT_COLUMNS = {
    *IDENTIFIERS,
    "date",
    "decision_date",
    "ticker",
    "target_weight",
    "is_current",
    "training_start_date",
    "training_end_date",
    "first_holding_date",
    "window_size",
    "solver_success",
    "solver_status",
    "solver_message",
    "objective_value",
    "covariance_scale",
    "tilt_strength",
    "min_history_observations",
    "zscore_clip_bound",
    "signal_lag_trading_days",
    "transaction_cost_bps",
    "base_target_weight",
    "sector",
    "tradable_sector_zscore",
    "applied_sector_zscore",
    "tradable_signal_source_date",
    "signal_prior_observations",
    "signal_was_missing",
    "tilt_multiplier",
    "active_core_weight",
    "active_satellite_weight",
    "active_crypto_sleeve_weight",
    "active_top_sector_count",
    "active_stock_weight_cap",
    "active_sector_weight_cap",
    "active_crypto_asset_weight_cap",
    "active_volatility_lookback",
    "active_sector_inverse_volatility",
    "active_sector_rank",
    "active_selected_sector",
    "active_sector_target_weight",
}
if EXPLORATORY_GENERATED:
    WEIGHT_COLUMNS.update(
        {
            "signal_window_trading_days",
            "exploratory_method_label",
            "unnormalised_weight",
            "signal_window_start_date",
            "signal_window_end_date",
            "latest_raw_news_date_used",
            "trailing_coverage_weighted_sentiment",
            "effective_coverage",
            "expanding_prior_mean",
            "expanding_prior_std",
            "expanding_prior_observations",
            "raw_trailing_zscore",
            "clipped_trailing_zscore",
            "coverage_adjusted_zscore",
        }
    )
METRIC_COLUMNS = {
    *IDENTIFIERS,
    "as_of_date",
    "sample_start_date",
    "sample_end_date",
    "current_holdings_date",
    "observations",
    "periods_per_year",
    "risk_free_rate_annual",
    "annual_return_method",
    "final_growth_of_1",
    "annualised_return",
    "annualised_volatility",
    "annualised_mean_excess_return",
    "sharpe_ratio",
    "maximum_drawdown",
}


@dataclass(frozen=True)
class LoadedArtifacts:
    returns: pd.DataFrame
    weights: pd.DataFrame
    metrics: pd.DataFrame


@pytest.fixture(scope="module")
def actual_artifacts() -> LoadedArtifacts:
    returns = pd.read_csv(
        RETURN_PATH,
        parse_dates=["date", "decision_date"],
    )
    weight_date_columns = [
            "date",
            "decision_date",
            "training_start_date",
            "training_end_date",
            "first_holding_date",
            "tradable_signal_source_date",
            "signal_window_start_date",
            "signal_window_end_date",
            "latest_raw_news_date_used",
        ]
    weight_header = pd.read_csv(WEIGHT_PATH, nrows=0).columns
    weights = pd.read_csv(
        WEIGHT_PATH,
        parse_dates=[column for column in weight_date_columns if column in weight_header],
    )
    performance = pd.read_csv(
        METRIC_PATH,
        parse_dates=[
            "as_of_date",
            "sample_start_date",
            "sample_end_date",
            "current_holdings_date",
        ],
    )
    return LoadedArtifacts(returns, weights, performance)


@pytest.fixture(scope="module")
def combined_asset_returns() -> pd.DataFrame:
    equities = etl.load_clean_equities()
    crypto = etl.load_clean_crypto()
    equity_returns = features.daily_returns(equities, asset_class="equity")
    crypto_returns = features.daily_returns(crypto, asset_class="crypto")
    combined = features.combined_returns_panel(equity_returns, crypto_returns)
    return features.complete_return_panel(combined).set_index("date")


def test_actual_csv_schemas_types_and_required_values(
    actual_artifacts: LoadedArtifacts,
) -> None:
    returns = actual_artifacts.returns
    weights = actual_artifacts.weights
    performance = actual_artifacts.metrics

    assert set(returns.columns) == RETURN_COLUMNS
    assert set(weights.columns) == WEIGHT_COLUMNS
    assert set(performance.columns) == METRIC_COLUMNS
    for frame in (returns, weights, performance):
        assert not frame[IDENTIFIERS].isna().any().any()
    assert not returns[["date", "decision_date"]].isna().any().any()
    assert not weights[
        [
            "date",
            "decision_date",
            "training_start_date",
            "training_end_date",
            "first_holding_date",
        ]
    ].isna().any().any()
    assert not weights["ticker"].isna().any()
    assert not performance[
        [
            "as_of_date",
            "sample_start_date",
            "sample_end_date",
            "current_holdings_date",
        ]
    ].isna().any().any()
    assert all(
        pd.api.types.is_datetime64_any_dtype(returns[column])
        for column in ("date", "decision_date")
    )
    assert all(
        pd.api.types.is_datetime64_any_dtype(weights[column])
        for column in (
            "date",
            "decision_date",
            "training_start_date",
            "training_end_date",
            "first_holding_date",
        )
    )
    assert all(
        pd.api.types.is_datetime64_any_dtype(performance[column])
        for column in (
            "as_of_date",
            "sample_start_date",
            "sample_end_date",
            "current_holdings_date",
        )
    )
    assert pd.api.types.is_bool_dtype(weights["is_current"])
    if EXPLORATORY_GENERATED:
        assert all(
            pd.api.types.is_datetime64_any_dtype(weights[column])
            for column in (
                "signal_window_start_date",
                "signal_window_end_date",
                "latest_raw_news_date_used",
            )
        )
    for column in ("daily_return", "growth_of_1"):
        assert pd.api.types.is_numeric_dtype(returns[column])
        assert np.isfinite(returns[column]).all()
    assert pd.api.types.is_numeric_dtype(weights["target_weight"])
    assert np.isfinite(weights["target_weight"]).all()
    metric_values = [
        "final_growth_of_1",
        "annualised_return",
        "annualised_volatility",
        "annualised_mean_excess_return",
        "sharpe_ratio",
        "maximum_drawdown",
    ]
    for column in metric_values:
        assert pd.api.types.is_numeric_dtype(performance[column])
        assert np.isfinite(performance[column]).all()
    assert pd.api.types.is_integer_dtype(performance["observations"])
    assert pd.api.types.is_integer_dtype(performance["periods_per_year"])


def test_actual_csv_unique_keys_sorting_and_identifier_reconciliation(
    actual_artifacts: LoadedArtifacts,
) -> None:
    returns = actual_artifacts.returns
    weights = actual_artifacts.weights
    performance = actual_artifacts.metrics

    assert not returns.duplicated(["fund", "date"]).any()
    assert not weights.duplicated(["fund", "decision_date", "ticker"]).any()
    assert not performance.duplicated(["fund"]).any()
    assert len(performance) == performance["fund"].nunique()
    assert set(returns["fund"]) == EXPECTED_FUNDS
    assert set(weights["fund"]) == EXPECTED_FUNDS
    assert set(performance["fund"]) == EXPECTED_FUNDS
    for _, group in returns.groupby("fund", sort=False):
        assert group["date"].is_monotonic_increasing
    for _, group in weights.groupby("fund", sort=False):
        assert group["decision_date"].is_monotonic_increasing

    return_ids = returns[IDENTIFIERS].drop_duplicates().sort_values("fund")
    weight_ids = weights[IDENTIFIERS].drop_duplicates().sort_values("fund")
    metric_ids = performance[IDENTIFIERS].drop_duplicates().sort_values("fund")
    pd.testing.assert_frame_equal(
        return_ids.reset_index(drop=True),
        weight_ids.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        return_ids.reset_index(drop=True),
        metric_ids.reset_index(drop=True),
    )
    expected_periods = performance["asset_family"].map({"crypto": 365}).fillna(252)
    assert performance["periods_per_year"].eq(expected_periods).all()
    assert performance["risk_free_rate_annual"].eq(0.0).all()
    assert performance["annual_return_method"].eq("geometric").all()


def test_actual_growth_sample_dates_and_join_row_conservation(
    actual_artifacts: LoadedArtifacts,
) -> None:
    returns = actual_artifacts.returns
    weights = actual_artifacts.weights
    performance = actual_artifacts.metrics

    for _, group in returns.groupby("fund", sort=False):
        expected_growth = np.cumprod(1 + group["daily_return"].to_numpy())
        np.testing.assert_allclose(
            group["growth_of_1"],
            expected_growth,
            rtol=NUMERIC_TOLERANCE,
            atol=NUMERIC_TOLERANCE,
        )

    sample = returns.groupby("fund").agg(
        sample_start_date=("date", "min"),
        sample_end_date=("date", "max"),
        observations=("date", "size"),
    )
    recorded = performance.set_index("fund")
    pd.testing.assert_series_equal(
        sample["sample_start_date"],
        recorded["sample_start_date"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        sample["sample_end_date"],
        recorded["sample_end_date"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        sample["observations"],
        recorded["observations"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        recorded["as_of_date"],
        recorded["sample_end_date"],
        check_names=False,
    )

    metrics_join = returns.merge(
        performance,
        on=IDENTIFIERS,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    assert len(metrics_join) == len(returns)
    assert metrics_join["_merge"].eq("both").all()

    universe_sizes = (
        weights.groupby([*IDENTIFIERS, "decision_date"])
        .size()
        .rename("universe_size")
        .reset_index()
    )
    return_universes = returns.merge(
        universe_sizes,
        on=[*IDENTIFIERS, "decision_date"],
        how="left",
        validate="many_to_one",
    )
    assert len(return_universes) == len(returns)
    assert return_universes["universe_size"].notna().all()
    expanded = returns.assign(return_row=np.arange(len(returns))).merge(
        weights[[*IDENTIFIERS, "decision_date", "ticker", "target_weight"]],
        on=[*IDENTIFIERS, "decision_date"],
        how="left",
        validate="many_to_many",
    )
    assert expanded["return_row"].nunique() == len(returns)
    assert len(expanded) == int(return_universes["universe_size"].sum())


def test_all_historical_weights_are_valid(
    actual_artifacts: LoadedArtifacts,
) -> None:
    weights = actual_artifacts.weights
    assert np.isfinite(weights["target_weight"]).all()
    assert (weights["target_weight"] >= -WEIGHT_TOLERANCE).all()
    assert (weights["target_weight"] <= 1 + WEIGHT_TOLERANCE).all()
    sums = weights.groupby(["fund", "decision_date"])["target_weight"].sum()
    np.testing.assert_allclose(
        sums,
        1.0,
        rtol=WEIGHT_TOLERANCE,
        atol=WEIGHT_TOLERANCE,
    )


def test_real_fund_metrics_match_independent_formulas(
    actual_artifacts: LoadedArtifacts,
) -> None:
    fund = "combined_min_variance"
    returns = actual_artifacts.returns.loc[
        actual_artifacts.returns["fund"].eq(fund)
    ].sort_values("date")
    actual = actual_artifacts.metrics.set_index("fund").loc[fund]
    daily = returns["daily_return"].to_numpy(dtype=float)
    wealth = np.cumprod(1 + daily)
    annualised_return = wealth[-1] ** (252 / len(daily)) - 1
    annualised_volatility = np.std(daily, ddof=1) * np.sqrt(252)
    annualised_mean_excess = np.mean(daily) * 252
    sharpe = annualised_mean_excess / annualised_volatility
    maximum_drawdown = np.min(wealth / np.maximum.accumulate(wealth) - 1)

    expected = {
        "final_growth_of_1": wealth[-1],
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "annualised_mean_excess_return": annualised_mean_excess,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": maximum_drawdown,
    }
    for metric, value in expected.items():
        assert actual[metric] == pytest.approx(
            value,
            rel=NUMERIC_TOLERANCE,
            abs=NUMERIC_TOLERANCE,
        )


def test_three_actual_csv_returns_reproduce_from_logged_weights(
    actual_artifacts: LoadedArtifacts,
    combined_asset_returns: pd.DataFrame,
) -> None:
    fund = "combined_min_variance"
    returns = actual_artifacts.returns.loc[
        actual_artifacts.returns["fund"].eq(fund)
    ].reset_index(drop=True)
    weights = actual_artifacts.weights

    for position in (0, len(returns) // 2, len(returns) - 1):
        sample = returns.iloc[position]
        active = weights.loc[
            weights["fund"].eq(fund)
            & weights["decision_date"].eq(sample["decision_date"])
        ].set_index("ticker")["target_weight"]
        matching_returns = combined_asset_returns.loc[
            sample["date"],
            active.index,
        ]
        reproduced = float(matching_returns.to_numpy() @ active.to_numpy())
        assert reproduced == pytest.approx(
            sample["daily_return"],
            rel=NUMERIC_TOLERANCE,
            abs=NUMERIC_TOLERANCE,
        )


def test_actual_current_holdings_reconcile_to_fact_sheet(
    actual_artifacts: LoadedArtifacts,
) -> None:
    weights = actual_artifacts.weights
    performance = actual_artifacts.metrics.set_index("fund")
    latest_dates = weights.groupby("fund")["decision_date"].transform("max")
    expected_current = weights["decision_date"].eq(latest_dates)
    assert weights["is_current"].equals(expected_current)

    current = weights.loc[weights["is_current"]]
    sums = current.groupby("fund")["target_weight"].sum()
    np.testing.assert_allclose(
        sums,
        1.0,
        rtol=WEIGHT_TOLERANCE,
        atol=WEIGHT_TOLERANCE,
    )
    selected_dates = current.groupby("fund")["decision_date"].first()
    pd.testing.assert_series_equal(
        selected_dates,
        performance["current_holdings_date"],
        check_names=False,
    )
    assert current[[*IDENTIFIERS, "ticker", "target_weight"]].notna().all().all()


def test_exploratory_artifact_contracts_when_extension_has_been_generated() -> None:
    """The pre-build run accepts the saved baseline; the post-build run is strict."""
    if not EXPLORATORY_GENERATED:
        return
    signal = pd.read_csv(
        EXPLORATORY_SIGNAL_PATH,
        parse_dates=[
            "date",
            "signal_window_start_date",
            "signal_window_end_date",
            "latest_raw_news_date_used",
        ],
    )
    comparison = pd.read_csv(
        EXPLORATORY_COMPARISON_PATH,
        parse_dates=["sample_start_date", "sample_end_date"],
    )
    predictability = pd.read_csv(
        PREDICTABILITY_PATH,
        parse_dates=["decision_date", "first_holding_date", "last_holding_date"],
    )
    summary = pd.read_csv(
        PREDICTABILITY_SUMMARY_PATH,
        parse_dates=["period_start_date", "period_end_date"],
    )
    signal_required = {
        "date",
        "sector",
        "signal_window_trading_days",
        "signal_window_start_date",
        "signal_window_end_date",
        "latest_raw_news_date_used",
        "trailing_coverage_weighted_sentiment",
        "effective_coverage",
        "expanding_prior_mean",
        "expanding_prior_std",
        "expanding_prior_observations",
        "raw_trailing_zscore",
        "clipped_trailing_zscore",
        "coverage_adjusted_zscore",
        "exploratory_method_label",
    }
    assert signal_required.issubset(signal.columns)
    assert not signal.duplicated(["date", "sector"]).any()
    assert signal["signal_window_trading_days"].eq(21).all()
    assert signal["effective_coverage"].between(0.0, 1.0).all()
    dated_windows = signal.dropna(subset=["signal_window_end_date"])
    assert (dated_windows["signal_window_end_date"] < dated_windows["date"]).all()

    comparison_required = {
        "fund",
        "asset_family",
        "method",
        "periods_per_year",
        "tilt_strength",
        "signal_window_trading_days",
        "final_period_used_for_parameter_selection",
    }
    assert comparison_required.issubset(comparison.columns)
    assert set(comparison["fund"]) == {
        "equity_equal_weight",
        "equity_sentiment_21d_coverage_tilt",
    }
    assert comparison["periods_per_year"].eq(252).all()
    assert comparison["final_period_used_for_parameter_selection"].eq(False).all()

    predictability_required = {
        "decision_date",
        "first_holding_date",
        "last_holding_date",
        "sector",
        "signal_method",
        "available_signal",
        "effective_coverage",
        "sector_holding_period_return",
        "market_holding_period_return",
        "sector_excess_return",
        "monthly_spearman_rank_ic",
    }
    assert predictability_required.issubset(predictability.columns)
    assert not predictability.duplicated(
        ["decision_date", "sector", "signal_method"]
    ).any()
    assert (predictability["first_holding_date"] > predictability["decision_date"]).all()
    assert len(predictability) == 36 * 10 * 2
    assert set(predictability["signal_method"]) == {
        "locked_prior_day_expanding_zscore",
        "exploratory_21d_coverage_adjusted_zscore",
    }

    summary_required = {
        "temporal_fold",
        "period_start_date",
        "period_end_date",
        "signal_method",
        "summary_scope",
        "monthly_decisions",
        "usable_sector_observations",
        "mean_monthly_rank_ic",
        "median_monthly_rank_ic",
        "proportion_monthly_rank_ic_above_zero",
        "signal_coverage_share",
        "mean_effective_news_coverage",
        "final_period_used_for_parameter_selection",
    }
    assert summary_required.issubset(summary.columns)
    assert set(summary["temporal_fold"]) == {
        "through_2021",
        "through_2022",
        "through_2023",
    }
    assert summary["final_period_used_for_parameter_selection"].eq(False).all()


def test_exploratory_latest_holdings_are_selected_after_generation(
    actual_artifacts: LoadedArtifacts,
) -> None:
    if not EXPLORATORY_GENERATED:
        return
    fund = "equity_sentiment_21d_coverage_tilt"
    weights = actual_artifacts.weights.loc[actual_artifacts.weights["fund"].eq(fund)]
    metrics = actual_artifacts.metrics.set_index("fund").loc[fund]
    latest = weights["decision_date"].max()
    current = weights.loc[weights["is_current"]]
    assert current["decision_date"].eq(latest).all()
    assert current["ticker"].nunique() == 50
    assert current["target_weight"].sum() == pytest.approx(1.0, abs=WEIGHT_TOLERANCE)
    assert metrics["current_holdings_date"] == latest


def test_active_sector_artifact_preserves_its_fixed_sleeves_and_logged_returns(
    actual_artifacts: LoadedArtifacts,
    combined_asset_returns: pd.DataFrame,
) -> None:
    fund = "combined_active_sector_allocation"
    weights = actual_artifacts.weights.loc[
        actual_artifacts.weights["fund"].eq(fund)
    ].copy()
    returns = actual_artifacts.returns.loc[
        actual_artifacts.returns["fund"].eq(fund)
    ].copy()
    metric = actual_artifacts.metrics.set_index("fund").loc[fund]
    assert ACTIVE_SECTOR_EXPOSURE_PATH.is_file()
    assert weights["active_core_weight"].eq(0.70).all()
    assert weights["active_satellite_weight"].eq(0.20).all()
    assert weights["active_crypto_sleeve_weight"].eq(0.10).all()
    assert weights["active_top_sector_count"].eq(2).all()
    assert weights["active_volatility_lookback"].eq(252).all()
    crypto = weights["ticker"].str.endswith("-USD")
    assert weights.loc[crypto].groupby("decision_date")["target_weight"].sum().eq(
        0.10
    ).all()
    equity_sector = weights.loc[~crypto].groupby(["decision_date", "sector"])[
        "target_weight"
    ].sum()
    assert (equity_sector <= 0.20 + WEIGHT_TOLERANCE).all()
    assert weights.loc[~crypto & weights["active_selected_sector"]].groupby(
        "decision_date"
    ).size().eq(10).all()
    source_dates = weights.loc[~crypto, "tradable_signal_source_date"]
    assert (source_dates < weights.loc[~crypto, "decision_date"]).all()
    for row in returns.iloc[[0, len(returns) // 2, -1]].itertuples(index=False):
        target = weights.loc[weights["decision_date"].eq(row.decision_date)].set_index(
            "ticker"
        )["target_weight"]
        expected = float(combined_asset_returns.loc[row.date, target.index] @ target)
        assert row.daily_return == pytest.approx(expected, abs=NUMERIC_TOLERANCE)
    current = weights.loc[weights["is_current"]]
    assert current["decision_date"].eq(weights["decision_date"].max()).all()
    assert metric["current_holdings_date"] == current["decision_date"].iloc[0]


def test_growth_sector_artifact_preserves_the_80_15_5_contract(
    actual_artifacts: LoadedArtifacts,
) -> None:
    fund = "combined_growth_sector_allocation"
    weights = actual_artifacts.weights.loc[
        actual_artifacts.weights["fund"].eq(fund)
    ].copy()
    assert GROWTH_SECTOR_EXPOSURE_PATH.is_file()
    assert weights["active_core_weight"].eq(0.80).all()
    assert weights["active_satellite_weight"].eq(0.15).all()
    assert weights["active_crypto_sleeve_weight"].eq(0.05).all()
    assert weights["active_top_sector_count"].eq(3).all()
    crypto = weights["ticker"].str.endswith("-USD")
    assert weights.loc[crypto].groupby("decision_date")["target_weight"].sum().eq(
        0.05
    ).all()
    assert weights.loc[~crypto & weights["active_selected_sector"]].groupby(
        "decision_date"
    ).size().eq(15).all()
    sector_weights = weights.loc[~crypto].groupby(["decision_date", "sector"])[
        "target_weight"
    ].sum()
    assert (sector_weights <= 0.15 + WEIGHT_TOLERANCE).all()


def test_aggressive_sector_artifact_preserves_the_50_30_20_contract(
    actual_artifacts: LoadedArtifacts,
) -> None:
    weights = actual_artifacts.weights.loc[
        actual_artifacts.weights["fund"].eq("combined_aggressive_sector_allocation")
    ]
    assert AGGRESSIVE_SECTOR_EXPOSURE_PATH.is_file()
    assert weights["active_core_weight"].eq(0.50).all()
    assert weights["active_satellite_weight"].eq(0.30).all()
    assert weights["active_crypto_sleeve_weight"].eq(0.20).all()
    crypto = weights["ticker"].str.endswith("-USD")
    assert weights.loc[crypto].groupby("decision_date")["target_weight"].sum().eq(
        0.20
    ).all()
    assert weights.loc[~crypto & weights["active_selected_sector"]].groupby(
        "decision_date"
    ).size().eq(15).all()
