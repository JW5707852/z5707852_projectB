"""Regression tests for the predeclared active sector allocation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src import active_sector, portfolios


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make a deterministic 50-stock, 10-crypto panel and lagged signals."""
    dates = pd.bdate_range("2020-01-02", periods=310)
    sectors = [f"Sector {number:02d}" for number in range(10)]
    mapping = pd.DataFrame(
        [
            {"ticker": f"EQ{sector_number}{stock_number}", "sector": sector}
            for sector_number, sector in enumerate(sectors)
            for stock_number in range(5)
        ]
    )
    crypto = [f"CRYPTO{number}-USD" for number in range(10)]
    tickers = [*mapping["ticker"], *crypto]
    values = np.random.default_rng(5545).normal(0.0002, 0.01, (len(dates), len(tickers)))
    returns = pd.DataFrame(values, columns=tickers)
    returns.insert(0, "date", dates)
    signals = pd.DataFrame(
        [
            {
                "date": date,
                "sector": sector,
                "tradable_sector_zscore": float(sector_number) + date_position / 10_000,
                "tradable_signal_source_date": dates[max(date_position - 1, 0)],
            }
            for date_position, date in enumerate(dates)
            for sector_number, sector in enumerate(sectors)
        ]
    )
    return returns, signals, mapping


@pytest.fixture(scope="module")
def active_result() -> tuple[portfolios.BacktestResult, pd.DataFrame]:
    returns, signals, mapping = _synthetic_inputs()
    return active_sector.build_active_sector_allocation(returns, signals, mapping), returns


def test_active_sector_weights_respect_predeclared_sleeves_and_caps(
    active_result: tuple[portfolios.BacktestResult, pd.DataFrame],
) -> None:
    result, _ = active_result
    weights = result.fund_weights
    crypto = weights["ticker"].str.endswith("-USD")
    np.testing.assert_allclose(
        weights.groupby("decision_date")["target_weight"].sum(),
        1.0,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        weights.loc[crypto].groupby("decision_date")["target_weight"].sum(),
        active_sector.CRYPTO_SLEEVE_WEIGHT,
        atol=1e-12,
        rtol=0.0,
    )
    equity_sector = weights.loc[~crypto].groupby(["decision_date", "sector"])[
        "target_weight"
    ].sum()
    assert (equity_sector <= active_sector.SECTOR_WEIGHT_CAP + 1e-12).all()
    assert (weights.loc[~crypto, "target_weight"] <= active_sector.STOCK_WEIGHT_CAP + 1e-12).all()
    assert (
        weights.loc[crypto, "target_weight"]
        <= active_sector.CRYPTO_ASSET_WEIGHT_CAP + 1e-12
    ).all()
    selected = weights.loc[~crypto & weights["active_selected_sector"]]
    assert selected.groupby("decision_date").size().eq(5 * active_sector.TOP_SECTOR_COUNT).all()


def test_active_sector_is_lagged_and_reproduces_three_logged_returns(
    active_result: tuple[portfolios.BacktestResult, pd.DataFrame],
) -> None:
    result, panel = active_result
    weights = result.fund_weights
    assert (
        weights.loc[weights["sector"].ne("Crypto"), "tradable_signal_source_date"]
        < weights.loc[weights["sector"].ne("Crypto"), "decision_date"]
    ).all()
    samples = result.fund_returns.iloc[[0, len(result.fund_returns) // 2, -1]]
    for row in samples.itertuples(index=False):
        target = weights.loc[
            weights["decision_date"].eq(row.decision_date)
        ].set_index("ticker")["target_weight"]
        actual = float(panel.loc[panel["date"].eq(row.date), target.index].iloc[0] @ target)
        assert row.daily_return == pytest.approx(actual, abs=1e-12)


def test_future_signal_change_cannot_change_prior_weights_or_returns() -> None:
    returns, signals, mapping = _synthetic_inputs()
    baseline = active_sector.build_active_sector_allocation(returns, signals, mapping)
    cutoff = baseline.fund_weights["decision_date"].sort_values().unique()[2]
    perturbed = signals.copy()
    future = perturbed["date"] > cutoff
    perturbed.loc[future, "tradable_sector_zscore"] *= -1_000.0
    altered = active_sector.build_active_sector_allocation(returns, perturbed, mapping)
    pd.testing.assert_frame_equal(
        baseline.fund_weights.loc[
            baseline.fund_weights["decision_date"] <= cutoff
        ].reset_index(drop=True),
        altered.fund_weights.loc[
            altered.fund_weights["decision_date"] <= cutoff
        ].reset_index(drop=True),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        baseline.fund_returns.loc[baseline.fund_returns["date"] <= cutoff].reset_index(drop=True),
        altered.fund_returns.loc[altered.fund_returns["date"] <= cutoff].reset_index(drop=True),
        check_exact=True,
    )


def test_active_sector_rejects_non_lagged_signal() -> None:
    returns, signals, mapping = _synthetic_inputs()
    decision_date = portfolios.monthly_rebalance_schedule(returns["date"]).iloc[0][
        "decision_date"
    ]
    invalid = signals["date"].eq(decision_date) & signals["sector"].eq("Sector 00")
    signals.loc[invalid, "tradable_signal_source_date"] = decision_date
    with pytest.raises(active_sector.ActiveSectorValidationError, match="strictly lagged"):
        active_sector.build_active_sector_allocation(returns, signals, mapping)


def test_growth_sector_allocation_uses_the_80_15_5_predeclared_design() -> None:
    returns, signals, mapping = _synthetic_inputs()
    result = active_sector.build_growth_sector_allocation(returns, signals, mapping)
    weights = result.fund_weights
    crypto = weights["ticker"].str.endswith("-USD")
    assert set(result.fund_returns["fund"]) == {active_sector.GROWTH_FUND}
    assert set(result.fund_returns["method"]) == {active_sector.GROWTH_METHOD}
    assert weights["active_core_weight"].eq(0.80).all()
    assert weights["active_satellite_weight"].eq(0.15).all()
    assert weights["active_crypto_sleeve_weight"].eq(0.05).all()
    assert weights["active_top_sector_count"].eq(3).all()
    assert weights["active_sector_weight_cap"].eq(0.15).all()
    assert weights["active_stock_weight_cap"].eq(0.03).all()
    assert weights["active_crypto_asset_weight_cap"].eq(0.005).all()
    np.testing.assert_allclose(
        weights.loc[crypto].groupby("decision_date")["target_weight"].sum(),
        0.05,
        atol=1e-12,
        rtol=0.0,
    )
    selected = weights.loc[~crypto & weights["active_selected_sector"]]
    assert selected.groupby("decision_date").size().eq(15).all()


def test_aggressive_sector_allocation_uses_the_50_30_20_design() -> None:
    returns, signals, mapping = _synthetic_inputs()
    result = active_sector.build_aggressive_sector_allocation(returns, signals, mapping)
    weights = result.fund_weights
    crypto = weights["ticker"].str.endswith("-USD")
    assert set(result.fund_returns["fund"]) == {active_sector.AGGRESSIVE_FUND}
    assert weights["active_core_weight"].eq(0.50).all()
    assert weights["active_satellite_weight"].eq(0.30).all()
    assert weights["active_crypto_sleeve_weight"].eq(0.20).all()
    assert weights["active_top_sector_count"].eq(3).all()
    np.testing.assert_allclose(
        weights.loc[crypto].groupby("decision_date")["target_weight"].sum(),
        0.20,
        atol=1e-12,
        rtol=0.0,
    )
    assert (weights.loc[crypto, "target_weight"] <= 0.02 + 1e-12).all()
    selected = weights.loc[~crypto & weights["active_selected_sector"]]
    assert selected.groupby("decision_date").size().eq(15).all()
