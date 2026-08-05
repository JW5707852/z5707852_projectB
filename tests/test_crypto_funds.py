"""Focused acceptance tests for the native-calendar crypto fund family."""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from scripts.run_part_b import CoreFundBuild, build_core_funds

CRYPTO_FUNDS = ("crypto_equal_weight", "crypto_min_variance")


@pytest.fixture(scope="module")
def crypto_build() -> CoreFundBuild:
    return build_core_funds()


def test_crypto_native_calendar_and_365_day_metrics(crypto_build: CoreFundBuild) -> None:
    returns = crypto_build.artifacts.fund_returns
    crypto = returns.loc[returns["fund"].isin(CRYPTO_FUNDS)]
    native_dates = crypto_build.crypto.crypto_asset_returns["date"]
    assert crypto.groupby("fund")["date"].apply(tuple).nunique() == 1
    assert set(crypto["date"]).issubset(set(native_dates))
    assert crypto["date"].dt.dayofweek.isin([5, 6]).any()
    metrics = crypto_build.artifacts.performance_metrics.set_index("fund").loc[list(CRYPTO_FUNDS)]
    assert metrics["periods_per_year"].eq(365).all()
    assert metrics["risk_free_rate_annual"].eq(0.0).all()
    assert metrics["annual_return_method"].eq("geometric").all()


def test_crypto_timing_weights_and_return_reproduction(crypto_build: CoreFundBuild) -> None:
    weights = crypto_build.artifacts.fund_weights
    returns = crypto_build.artifacts.fund_returns
    crypto_weights = weights.loc[weights["fund"].isin(CRYPTO_FUNDS)]
    assert (crypto_weights["training_end_date"] < crypto_weights["first_holding_date"]).all()
    assert crypto_weights["window_size"].min() >= 252
    assert crypto_weights["transaction_cost_bps"].eq(0.0).all()
    assert np.isfinite(crypto_weights["target_weight"]).all()
    assert crypto_weights["target_weight"].between(0.0, 1.0).all()
    assert np.allclose(
        crypto_weights.groupby(["fund", "decision_date"])["target_weight"].sum(), 1.0
    )
    minimum = crypto_weights.loc[crypto_weights["fund"].eq("crypto_min_variance")]
    assert minimum.pivot(index="decision_date", columns="ticker", values="target_weight").diff().abs().to_numpy()[1:].max() > 1e-6

    panel = crypto_build.crypto.crypto_asset_returns.set_index("date")
    sample = returns.loc[returns["fund"].eq("crypto_min_variance")].iloc[[0, 200, -1]]
    for row in sample.itertuples(index=False):
        target = crypto_weights.loc[
            crypto_weights["fund"].eq(row.fund)
            & crypto_weights["decision_date"].eq(row.decision_date)
        ].set_index("ticker")["target_weight"]
        assert float(panel.loc[row.date, target.index] @ target) == pytest.approx(
            row.daily_return, abs=1e-15
        )


def test_crypto_current_holdings_and_non_crypto_regression(crypto_build: CoreFundBuild) -> None:
    artifacts = crypto_build.artifacts
    metrics = artifacts.performance_metrics.set_index("fund")
    for fund in CRYPTO_FUNDS:
        current = artifacts.fund_weights.loc[
            artifacts.fund_weights["fund"].eq(fund) & artifacts.fund_weights["is_current"]
        ]
        assert current["decision_date"].nunique() == 1
        assert current["decision_date"].iloc[0] == metrics.loc[fund, "current_holdings_date"]

    for before, after in (
        (crypto_build.non_crypto_artifacts.fund_returns, artifacts.fund_returns),
        (crypto_build.non_crypto_artifacts.fund_weights, artifacts.fund_weights),
        (crypto_build.non_crypto_artifacts.performance_metrics, artifacts.performance_metrics),
    ):
        actual = after.loc[~after["asset_family"].eq("crypto")]
        assert before.to_csv(index=False, date_format="%Y-%m-%d") == actual.to_csv(
            index=False, date_format="%Y-%m-%d"
        )
