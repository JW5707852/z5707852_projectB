"""Temporal tests for the non-causal sector predictability diagnostic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src import fusion, portfolios, sentiment

TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def predictability_fixture() -> dict[str, object]:
    dates = pd.bdate_range("2020-01-02", "2023-12-29")
    position = np.arange(len(dates), dtype=float)
    tickers = ["AAA", "BBB", "CCC"]
    sectors = ["Energy", "Financials", "Technology"]
    returns = pd.DataFrame(
        {
            "date": dates,
            "AAA": 0.0002 + 0.004 * np.sin(position / 13.0),
            "BBB": 0.0003 + 0.003 * np.cos(position / 17.0),
            "CCC": 0.0001 + 0.005 * np.sin(position / 19.0 + 0.4),
        }
    )
    mapping = pd.DataFrame({"ticker": tickers, "sector": sectors})
    ticker_days = pd.concat(
        [
            pd.DataFrame(
                {
                    "trading_date": dates,
                    "ticker": ticker,
                    "sector": sector,
                    "ticker_day_compound": np.sin(position / (11.0 + offset)),
                    "headline_count": 1,
                }
            )
            for offset, (ticker, sector) in enumerate(zip(tickers, sectors, strict=True))
        ],
        ignore_index=True,
    )
    locked_signal = sentiment.sector_sentiment_index(
        ticker_days,
        dates,
        mapping,
        min_history=60,
        zscore_clip=2.0,
        signal_lag=1,
    )
    exploratory_signal = sentiment.build_coverage_adjusted_trailing_signal(
        locked_signal
    )
    base = portfolios.oos_backtest(
        returns,
        fund=fusion.BASE_FUND,
        method="equal_weight",
        asset_family="equity",
        initial_window=252,
    )
    diagnostic = fusion.build_sentiment_predictability(
        base,
        returns,
        mapping,
        locked_signal,
        exploratory_signal,
    )
    summary = fusion.build_predictability_summary(diagnostic)
    return {
        "returns": returns,
        "mapping": mapping,
        "base": base,
        "locked_signal": locked_signal,
        "exploratory_signal": exploratory_signal,
        "diagnostic": diagnostic,
        "summary": summary,
    }


def test_predictability_rows_are_conserved_and_returns_are_strictly_future(
    predictability_fixture: dict[str, object],
) -> None:
    base = predictability_fixture["base"]
    mapping = predictability_fixture["mapping"]
    diagnostic = predictability_fixture["diagnostic"]
    assert isinstance(base, portfolios.BacktestResult)
    assert isinstance(mapping, pd.DataFrame)
    assert isinstance(diagnostic, pd.DataFrame)
    decision_count = base.fund_returns["decision_date"].nunique()
    expected_rows = decision_count * mapping["sector"].nunique() * 2

    assert len(diagnostic) == expected_rows
    assert not diagnostic.duplicated(
        ["decision_date", "sector", "signal_method"]
    ).any()
    assert (diagnostic["first_holding_date"] > diagnostic["decision_date"]).all()
    assert (diagnostic["last_holding_date"] >= diagnostic["first_holding_date"]).all()
    assert set(diagnostic["signal_method"]) == {
        fusion.LOCKED_SIGNAL_METHOD,
        fusion.EXPLORATORY_SIGNAL_METHOD,
    }


def test_sector_market_and_excess_returns_match_direct_compounding(
    predictability_fixture: dict[str, object],
) -> None:
    returns = predictability_fixture["returns"]
    diagnostic = predictability_fixture["diagnostic"]
    assert isinstance(returns, pd.DataFrame)
    assert isinstance(diagnostic, pd.DataFrame)
    sample = diagnostic.loc[
        diagnostic["signal_method"].eq(fusion.EXPLORATORY_SIGNAL_METHOD)
        & diagnostic["sector"].eq("Energy")
    ].iloc[len(diagnostic["decision_date"].unique()) // 2]
    panel = returns.set_index("date")
    holding = panel.loc[sample["first_holding_date"] : sample["last_holding_date"]]
    expected_sector = float((1.0 + holding["AAA"]).prod() - 1.0)
    expected_market = float((1.0 + holding[["AAA", "BBB", "CCC"]].mean(axis=1)).prod() - 1.0)
    expected_excess = (1.0 + expected_sector) / (1.0 + expected_market) - 1.0

    assert sample["sector_holding_period_return"] == pytest.approx(
        expected_sector, abs=TOLERANCE
    )
    assert sample["market_holding_period_return"] == pytest.approx(
        expected_market, abs=TOLERANCE
    )
    assert sample["sector_excess_return"] == pytest.approx(
        expected_excess, abs=TOLERANCE
    )


def test_monthly_spearman_is_cross_sectional_and_not_a_shuffled_fold(
    predictability_fixture: dict[str, object],
) -> None:
    diagnostic = predictability_fixture["diagnostic"]
    assert isinstance(diagnostic, pd.DataFrame)
    usable_groups = diagnostic.dropna(subset=["available_signal"]).groupby(
        ["decision_date", "signal_method"], sort=True
    )
    for _, group in usable_groups:
        expected = group["available_signal"].rank(method="average").corr(
            group["sector_excess_return"].rank(method="average")
        )
        actual = group["monthly_spearman_rank_ic"].iloc[0]
        assert actual == pytest.approx(expected, abs=TOLERANCE)
        assert group["monthly_spearman_rank_ic"].nunique() == 1
        assert group["usable_sector_count_for_ic"].eq(len(group)).all()


def test_summary_uses_only_predeclared_time_ordered_expanding_periods(
    predictability_fixture: dict[str, object],
) -> None:
    summary = predictability_fixture["summary"]
    base = predictability_fixture["base"]
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(base, portfolios.BacktestResult)
    overall = summary.loc[summary["summary_scope"].eq("overall")]
    assert set(overall["temporal_fold"]) == {
        "through_2021",
        "through_2022",
        "through_2023",
    }
    holding_starts = base.fund_returns.groupby("decision_date")["date"].min()
    expected_decisions = {
        fold_name: int(holding_starts.between(fold_start, fold_end).sum())
        for fold_name, fold_start, fold_end in fusion.PREDICTABILITY_FOLDS
    }
    assert (
        overall.groupby("temporal_fold")["monthly_decisions"].first().to_dict()
        == expected_decisions
    )
    assert list(expected_decisions.values()) == sorted(expected_decisions.values())
    assert overall["period_start_date"].eq(pd.Timestamp("2021-01-04")).all()
    assert overall["final_period_used_for_parameter_selection"].eq(False).all()
    assert overall["signal_coverage_share"].between(0.0, 1.0).all()
    assert overall["mean_effective_news_coverage"].between(0.0, 1.0).all()
    sector_rows = summary.loc[summary["summary_scope"].eq("sector")]
    assert set(sector_rows["sector"]) == {"Energy", "Financials", "Technology"}
    assert sector_rows["sector_time_series_spearman"].notna().all()
    diagnostic = predictability_fixture["diagnostic"]
    assert isinstance(diagnostic, pd.DataFrame)
    for row in sector_rows.itertuples(index=False):
        source = diagnostic.loc[
            diagnostic["signal_method"].eq(row.signal_method)
            & diagnostic["sector"].eq(row.sector)
            & diagnostic["first_holding_date"].between(
                row.period_start_date, row.period_end_date
            )
        ]
        assert row.signal_coverage_share == pytest.approx(
            source["available_signal"].notna().mean(), abs=TOLERANCE
        )
        assert row.mean_effective_news_coverage == pytest.approx(
            source["effective_coverage"].mean(), abs=TOLERANCE
        )


def test_future_return_perturbation_cannot_change_an_earlier_holding_diagnostic(
    predictability_fixture: dict[str, object],
) -> None:
    returns = predictability_fixture["returns"]
    mapping = predictability_fixture["mapping"]
    base = predictability_fixture["base"]
    locked = predictability_fixture["locked_signal"]
    exploratory = predictability_fixture["exploratory_signal"]
    baseline = predictability_fixture["diagnostic"]
    assert isinstance(returns, pd.DataFrame)
    assert isinstance(mapping, pd.DataFrame)
    assert isinstance(base, portfolios.BacktestResult)
    assert isinstance(locked, pd.DataFrame)
    assert isinstance(exploratory, pd.DataFrame)
    assert isinstance(baseline, pd.DataFrame)
    first_end = baseline["last_holding_date"].min()
    perturbed_returns = returns.copy()
    perturb = perturbed_returns["date"] > first_end
    perturbed_returns.loc[perturb, ["AAA", "BBB", "CCC"]] *= -25.0
    changed = fusion.build_sentiment_predictability(
        base,
        perturbed_returns,
        mapping,
        locked,
        exploratory,
    )
    columns = [
        "decision_date",
        "sector",
        "signal_method",
        "sector_holding_period_return",
        "market_holding_period_return",
        "sector_excess_return",
        "monthly_spearman_rank_ic",
    ]
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["last_holding_date"].eq(first_end), columns].reset_index(
            drop=True
        ),
        changed.loc[changed["last_holding_date"].eq(first_end), columns].reset_index(
            drop=True
        ),
        check_exact=True,
    )
