"""Walk-forward timing, return-reproduction, and hosted fund integration tests."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scripts.run_part_b import BaseFundBuild, build_base_funds
from src import portfolios


def _regime_return_panel(periods: int = 520) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range("2020-01-02", periods=periods)
    split = periods // 2
    asset_a = np.concatenate(
        [
            rng.normal(0.0002, 0.004, split),
            rng.normal(0.0002, 0.025, periods - split),
        ]
    )
    asset_b = np.concatenate(
        [
            rng.normal(0.0002, 0.022, split),
            rng.normal(0.0002, 0.004, periods - split),
        ]
    )
    asset_c = rng.normal(0.0002, 0.012, periods)
    return pd.DataFrame(
        {"date": dates, "A": asset_a, "B": asset_b, "C": asset_c}
    )


def test_walk_forward_audit_ends_training_before_first_holding_return() -> None:
    result = portfolios.oos_backtest(
        _regime_return_panel(),
        fund="synthetic_min_variance",
        method="min_variance",
        asset_family="synthetic",
        initial_window=252,
    )
    audit = result.rebalance_audit

    assert (audit["training_end_date"] == audit["decision_date"]).all()
    assert (audit["training_end_date"] < audit["first_holding_date"]).all()
    assert audit["window_size"].min() >= 252
    assert audit["solver_success"].all()
    assert np.isfinite(audit["objective_value"]).all()


def test_rebalance_schedule_rejects_duplicate_dates() -> None:
    dates = list(pd.bdate_range("2020-01-02", periods=260))
    dates.insert(100, dates[100])

    with pytest.raises(ValueError, match="dates must be unique"):
        portfolios.monthly_rebalance_schedule(dates, initial_window=252)


def test_future_return_perturbation_does_not_change_past_outputs() -> None:
    panel = _regime_return_panel()
    cutoff = panel.loc[399, "date"]
    perturbed = panel.copy()
    perturbed.loc[perturbed["date"] > cutoff, ["A", "B", "C"]] *= -25

    baseline = portfolios.oos_backtest(
        panel,
        fund="synthetic_min_variance",
        method="min_variance",
        asset_family="synthetic",
        initial_window=252,
    )
    changed = portfolios.oos_backtest(
        perturbed,
        fund="synthetic_min_variance",
        method="min_variance",
        asset_family="synthetic",
        initial_window=252,
    )

    baseline_audit = baseline.rebalance_audit.loc[
        baseline.rebalance_audit["decision_date"] <= cutoff
    ].reset_index(drop=True)
    changed_audit = changed.rebalance_audit.loc[
        changed.rebalance_audit["decision_date"] <= cutoff
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline_audit, changed_audit, check_exact=True)

    baseline_returns = baseline.fund_returns.loc[
        baseline.fund_returns["date"] <= cutoff
    ].reset_index(drop=True)
    changed_returns = changed.fund_returns.loc[
        changed.fund_returns["date"] <= cutoff
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline_returns, changed_returns, check_exact=True)


def test_sampled_daily_return_is_reproduced_from_logged_weights() -> None:
    panel = _regime_return_panel()
    result = portfolios.oos_backtest(
        panel,
        fund="synthetic_min_variance",
        method="min_variance",
        asset_family="synthetic",
        initial_window=252,
    )
    sample = result.fund_returns.iloc[25]
    weights = result.fund_weights.loc[
        result.fund_weights["decision_date"].eq(sample["decision_date"])
    ].set_index("ticker")["target_weight"]
    asset_returns = panel.set_index("date").loc[sample["date"], weights.index]
    reproduced = float(asset_returns.to_numpy() @ weights.to_numpy())

    assert reproduced == pytest.approx(sample["daily_return"], abs=1e-15)


def test_dynamic_weights_vary_across_methods_and_rebalances() -> None:
    panel = _regime_return_panel()
    equal = portfolios.oos_backtest(
        panel,
        fund="synthetic_equal",
        method="equal_weight",
        asset_family="synthetic",
        initial_window=252,
    )
    minimum = portfolios.oos_backtest(
        panel,
        fund="synthetic_minimum",
        method="min_variance",
        asset_family="synthetic",
        initial_window=252,
    )
    combined = portfolios.concatenate_backtests([equal, minimum])

    summary = portfolios.validate_weight_variation(
        combined.fund_weights,
        dynamic_fund="synthetic_minimum",
        benchmark_fund="synthetic_equal",
    )

    assert summary["max_rebalance_weight_change"] > 1e-6
    assert summary["max_method_weight_difference"] > 1e-6


@pytest.fixture(scope="module")
def official_base_funds() -> BaseFundBuild:
    return build_base_funds()


def test_official_base_funds_share_dates_and_have_valid_audits(
    official_base_funds: BaseFundBuild,
) -> None:
    returns = official_base_funds.backtests.fund_returns
    audit = official_base_funds.backtests.rebalance_audit
    expected_funds = {
        "combined_equal_weight",
        "combined_min_variance",
        "equity_equal_weight",
    }

    assert set(returns["fund"]) == expected_funds
    date_sets = returns.groupby("fund")["date"].apply(tuple)
    assert date_sets.map(hash).nunique() == 1
    assert (audit["training_end_date"] < audit["first_holding_date"]).all()
    assert audit["window_size"].min() == 252
    assert audit["solver_success"].all()
    assert np.isfinite(audit["objective_value"]).all()
    assert audit["target_weights"].map(json.loads).map(len).min() == 50


def test_official_weights_are_constrained_and_nontrivial(
    official_base_funds: BaseFundBuild,
) -> None:
    weights = official_base_funds.backtests.fund_weights
    sums = weights.groupby(["fund", "decision_date"])["target_weight"].sum()

    assert np.isfinite(weights["target_weight"]).all()
    assert (weights["target_weight"] >= -1e-12).all()
    assert (weights["target_weight"] <= 1 + 1e-12).all()
    assert np.allclose(sums, 1.0, atol=1e-10)
    assert official_base_funds.variation_summary[
        "max_rebalance_weight_change"
    ] > 1e-6
    assert official_base_funds.variation_summary[
        "max_method_weight_difference"
    ] > 1e-6


def test_official_sampled_return_reproduces_from_logged_weights(
    official_base_funds: BaseFundBuild,
) -> None:
    returns = official_base_funds.backtests.fund_returns
    weights = official_base_funds.backtests.fund_weights
    sample = returns.loc[returns["fund"].eq("combined_min_variance")].iloc[100]
    active_weights = weights.loc[
        weights["fund"].eq("combined_min_variance")
        & weights["decision_date"].eq(sample["decision_date"])
    ].set_index("ticker")["target_weight"]
    asset_returns = official_base_funds.combined_asset_returns.set_index("date").loc[
        sample["date"], active_weights.index
    ]
    reproduced = float(asset_returns.to_numpy() @ active_weights.to_numpy())

    assert reproduced == pytest.approx(sample["daily_return"], abs=1e-15)
