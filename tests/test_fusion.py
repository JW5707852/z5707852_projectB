"""Locked, look-ahead-safe equity sentiment-fusion tests."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scripts.run_part_b import EXPECTED_FUND_IDENTITIES, CoreFundBuild, build_core_funds
from src import fusion, portfolios, sentiment

NUMERIC_TOLERANCE = 1e-12
TILT_GRID = (0.00, 0.05, 0.10, 0.15, 0.20)


def _direct_performance(daily_returns: pd.Series) -> dict[str, float]:
    """Calculate locked metrics directly without production metric helpers."""
    values = daily_returns.to_numpy(dtype=float)
    wealth = np.cumprod(1.0 + values)
    volatility = float(np.std(values, ddof=1) * np.sqrt(252))
    return {
        "annualised_return": float(wealth[-1] ** (252 / len(values)) - 1),
        "annualised_volatility": volatility,
        "sharpe_ratio": float(values.mean() * 252 / volatility),
        "maximum_drawdown": float(
            np.min(wealth / np.maximum.accumulate(wealth) - 1)
        ),
    }


def _official_fund_rows(
    official_core_funds: CoreFundBuild,
    *,
    table: str,
    fund: str,
) -> pd.DataFrame:
    frame = getattr(official_core_funds.artifacts, table)
    sort_columns = ["date"] if table == "fund_returns" else ["decision_date", "ticker"]
    return frame.loc[frame["fund"].eq(fund)].sort_values(sort_columns).reset_index(
        drop=True
    )


def _independent_raw_sector_scores(
    official_core_funds: CoreFundBuild,
) -> pd.DataFrame:
    """Rebuild raw sector scores directly from headline-level VADER outputs."""
    headlines = official_core_funds.sentiment.headline_scores
    ticker_days = (
        headlines.dropna(subset=["trading_date"])
        .groupby(["trading_date", "ticker", "sector"], as_index=False)
        .agg(ticker_day_compound=("vader_compound", "mean"))
    )
    return (
        ticker_days.groupby(["trading_date", "sector"], as_index=False)
        .agg(raw_sector_compound=("ticker_day_compound", "mean"))
        .rename(columns={"trading_date": "date"})
        .sort_values(["date", "sector"])
        .reset_index(drop=True)
    )


def _independent_grid_path(
    official_core_funds: CoreFundBuild,
    strength: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a diagnostic strength directly, without production fusion helpers."""
    mapping = official_core_funds.base.equity_sector_map.sort_values("ticker")
    tickers = mapping["ticker"].tolist()
    base_weight = np.full(len(tickers), 1.0 / len(tickers))
    asset_returns = official_core_funds.base.equity_asset_returns.set_index("date")
    base_returns = _official_fund_rows(
        official_core_funds,
        table="fund_returns",
        fund=fusion.BASE_FUND,
    )
    signal = official_core_funds.sentiment.sector_index
    return_records: list[dict[str, object]] = []
    exposure_records: list[dict[str, object]] = []

    for decision_date, holding in base_returns.groupby("decision_date", sort=True):
        sector_signal = (
            signal.loc[
                signal["date"].eq(decision_date),
                ["sector", "tradable_sector_zscore"],
            ]
            .set_index("sector")["tradable_sector_zscore"]
        )
        zscores = mapping["sector"].map(sector_signal).fillna(0.0).to_numpy(dtype=float)
        unnormalised = base_weight * (1.0 + strength * zscores)
        weights = unnormalised / unnormalised.sum()
        assert np.isfinite(weights).all()
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(1.0, abs=NUMERIC_TOLERANCE)

        ordered_holding = holding.sort_values("date")
        asset_slice = asset_returns.loc[ordered_holding["date"], tickers]
        fund_returns = asset_slice.to_numpy(dtype=float) @ weights
        for date, daily_return in zip(
            ordered_holding["date"], fund_returns, strict=True
        ):
            return_records.append(
                {
                    "strength": strength,
                    "date": date,
                    "decision_date": decision_date,
                    "daily_return": float(daily_return),
                }
            )

        decision_exposure = pd.DataFrame(
            {"sector": mapping["sector"], "target_weight": weights}
        ).groupby("sector", as_index=False)["target_weight"].sum()
        for row in decision_exposure.itertuples(index=False):
            exposure_records.append(
                {
                    "strength": strength,
                    "decision_date": decision_date,
                    "sector": row.sector,
                    "sector_weight_difference": float(row.target_weight - 0.1),
                }
            )
    return (
        pd.DataFrame.from_records(return_records),
        pd.DataFrame.from_records(exposure_records),
    )


def _independent_sector_exposure(
    official_core_funds: CoreFundBuild,
    decision_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Rebuild base-versus-tilted sector exposure from stock weights."""
    mapping = official_core_funds.base.equity_sector_map
    selected = official_core_funds.artifacts.fund_weights.loc[
        official_core_funds.artifacts.fund_weights["fund"].isin(
            [fusion.BASE_FUND, fusion.TILTED_FUND]
        )
        & official_core_funds.artifacts.fund_weights["decision_date"].isin(
            decision_dates
        )
    ].drop(columns="sector", errors="ignore")
    long = (
        selected.merge(mapping, on="ticker", validate="many_to_one")
        .groupby(["fund", "decision_date", "sector"], as_index=False)[
            "target_weight"
        ]
        .sum()
    )
    wide = long.pivot(
        index=["decision_date", "sector"],
        columns="fund",
        values="target_weight",
    ).reset_index()
    wide["sector_weight_difference"] = (
        wide[fusion.TILTED_FUND] - wide[fusion.BASE_FUND]
    )
    return wide


@pytest.fixture(scope="module")
def synthetic_fusion_inputs() -> dict[str, object]:
    dates = pd.bdate_range("2022-01-03", periods=150)
    position = np.arange(len(dates), dtype=float)
    returns = pd.DataFrame(
        {
            "date": dates,
            "AAA": 0.0004 + 0.006 * np.sin(position / 8),
            "BBB": 0.0003 + 0.004 * np.cos(position / 11),
            "CCC": 0.0002 + 0.005 * np.sin(position / 13 + 0.5),
        }
    )
    base = portfolios.oos_backtest(
        returns,
        fund=fusion.BASE_FUND,
        method="equal_weight",
        asset_family="equity",
        initial_window=60,
    )
    mapping = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Tech", "Energy"],
        }
    )
    raw = pd.DataFrame(
        {
            "trading_date": np.tile(dates, 2),
            "ticker": np.repeat(["AAA", "CCC"], len(dates)),
            "sector": np.repeat(["Tech", "Energy"], len(dates)),
            "ticker_day_compound": np.concatenate(
                [np.sin(position / 7), np.cos(position / 9)]
            ),
            "headline_count": 1,
        }
    )
    signal = sentiment.sector_sentiment_index(
        raw,
        dates,
        mapping,
        min_history=60,
        zscore_clip=2.0,
        signal_lag=1,
    )
    tilted = fusion.apply_sentiment_tilt(base, returns, signal, mapping)
    return {
        "dates": dates,
        "returns": returns,
        "base": base,
        "mapping": mapping,
        "raw": raw,
        "signal": signal,
        "tilted": tilted,
    }


def test_expanding_zscore_uses_only_prior_values_and_is_lagged(
    synthetic_fusion_inputs: dict[str, object],
) -> None:
    dates = synthetic_fusion_inputs["dates"]
    raw = synthetic_fusion_inputs["raw"]
    signal = synthetic_fusion_inputs["signal"]
    assert isinstance(dates, pd.DatetimeIndex)
    assert isinstance(raw, pd.DataFrame)
    assert isinstance(signal, pd.DataFrame)

    source_position = 80
    source_date = dates[source_position]
    use_date = dates[source_position + 1]
    tech_raw = raw.loc[raw["sector"].eq("Tech")].sort_values("trading_date")
    prior = tech_raw["ticker_day_compound"].iloc[:source_position].to_numpy()
    current = float(tech_raw["ticker_day_compound"].iloc[source_position])
    expected_raw_z = (current - prior.mean()) / prior.std(ddof=1)
    expected_clipped = float(np.clip(expected_raw_z, -2.0, 2.0))
    tech = signal.loc[signal["sector"].eq("Tech")].set_index("date")

    assert tech.loc[source_date, "raw_expanding_zscore"] == pytest.approx(
        expected_raw_z, abs=1e-12
    )
    assert tech.loc[use_date, "tradable_sector_zscore"] == pytest.approx(
        expected_clipped, abs=1e-12
    )
    assert tech.loc[use_date, "tradable_signal_source_date"] == source_date
    assert tech.loc[use_date, "signal_prior_observations"] == source_position
    assert tech.loc[use_date, "tradable_signal_source_date"] < use_date


def test_future_sentiment_perturbation_does_not_change_past_signal_or_weights(
    synthetic_fusion_inputs: dict[str, object],
) -> None:
    dates = synthetic_fusion_inputs["dates"]
    raw = synthetic_fusion_inputs["raw"]
    mapping = synthetic_fusion_inputs["mapping"]
    base = synthetic_fusion_inputs["base"]
    returns = synthetic_fusion_inputs["returns"]
    baseline_signal = synthetic_fusion_inputs["signal"]
    baseline_tilt = synthetic_fusion_inputs["tilted"]
    assert isinstance(dates, pd.DatetimeIndex)
    assert isinstance(raw, pd.DataFrame)
    assert isinstance(mapping, pd.DataFrame)
    assert isinstance(base, portfolios.BacktestResult)
    assert isinstance(returns, pd.DataFrame)
    assert isinstance(baseline_signal, pd.DataFrame)
    assert isinstance(baseline_tilt, portfolios.BacktestResult)

    cutoff = dates[105]
    perturbed_raw = raw.copy()
    perturbed_raw.loc[perturbed_raw["trading_date"] > cutoff, "ticker_day_compound"] *= -50
    changed_signal = sentiment.sector_sentiment_index(
        perturbed_raw,
        dates,
        mapping,
        min_history=60,
        zscore_clip=2.0,
        signal_lag=1,
    )
    changed_tilt = fusion.apply_sentiment_tilt(
        base, returns, changed_signal, mapping
    )

    signal_columns = [
        "date",
        "sector",
        "raw_expanding_zscore",
        "raw_zscore_clipped",
        "tradable_sector_zscore",
        "tradable_signal_source_date",
    ]
    pd.testing.assert_frame_equal(
        baseline_signal.loc[
            baseline_signal["date"] <= cutoff,
            signal_columns,
        ].reset_index(drop=True),
        changed_signal.loc[changed_signal["date"] <= cutoff, signal_columns].reset_index(drop=True),
        check_exact=True,
    )
    weight_columns = ["decision_date", "ticker", "target_weight"]
    pd.testing.assert_frame_equal(
        baseline_tilt.fund_weights.loc[
            baseline_tilt.fund_weights["decision_date"] <= cutoff,
            weight_columns,
        ].reset_index(drop=True),
        changed_tilt.fund_weights.loc[
            changed_tilt.fund_weights["decision_date"] <= cutoff,
            weight_columns,
        ].reset_index(drop=True),
        check_exact=True,
    )


def test_tilt_formula_weight_constraints_and_signal_audit(
    synthetic_fusion_inputs: dict[str, object],
) -> None:
    tilted = synthetic_fusion_inputs["tilted"]
    assert isinstance(tilted, portfolios.BacktestResult)
    weights = tilted.fund_weights
    sums = weights.groupby("decision_date")["target_weight"].sum()

    assert np.isfinite(weights["target_weight"]).all()
    assert (weights["target_weight"] >= 0).all()
    assert np.allclose(sums, 1.0, atol=1e-12)
    assert weights["tilt_multiplier"].between(0.8, 1.2, inclusive="both").all()
    available = weights["tradable_sector_zscore"].notna()
    assert (
        weights.loc[available, "tradable_signal_source_date"]
        < weights.loc[available, "decision_date"]
    ).all()
    assert (weights.loc[available, "signal_prior_observations"] >= 60).all()

    decision = weights.loc[available, "decision_date"].iloc[0]
    sample = weights.loc[weights["decision_date"].eq(decision)].copy()
    unnormalised = sample["base_target_weight"] * (
        1 + 0.10 * sample["tradable_sector_zscore"].fillna(0.0)
    )
    expected = unnormalised / unnormalised.sum()
    np.testing.assert_allclose(sample["target_weight"], expected, atol=1e-12)


def test_missing_sector_signal_uses_multiplier_one(
    synthetic_fusion_inputs: dict[str, object],
) -> None:
    base = synthetic_fusion_inputs["base"]
    returns = synthetic_fusion_inputs["returns"]
    mapping = synthetic_fusion_inputs["mapping"]
    signal = synthetic_fusion_inputs["signal"].copy()
    assert isinstance(base, portfolios.BacktestResult)
    assert isinstance(returns, pd.DataFrame)
    assert isinstance(mapping, pd.DataFrame)
    assert isinstance(signal, pd.DataFrame)

    decision = base.fund_weights["decision_date"].sort_values().iloc[-1]
    missing = signal["date"].eq(decision) & signal["sector"].eq("Tech")
    signal.loc[
        missing,
        ["tradable_sector_zscore", "tradable_signal_source_date", "signal_prior_observations"],
    ] = [np.nan, pd.NaT, np.nan]
    result = fusion.apply_sentiment_tilt(base, returns, signal, mapping)
    tech = result.fund_weights.loc[
        result.fund_weights["decision_date"].eq(decision)
        & result.fund_weights["sector"].eq("Tech")
    ]

    assert tech["signal_was_missing"].all()
    assert tech["tradable_sector_zscore"].isna().all()
    assert tech["tilt_multiplier"].eq(1.0).all()


def test_common_sample_fair_metrics_turnover_and_sector_exposure(
    synthetic_fusion_inputs: dict[str, object],
) -> None:
    base = synthetic_fusion_inputs["base"]
    tilted = synthetic_fusion_inputs["tilted"]
    mapping = synthetic_fusion_inputs["mapping"]
    assert isinstance(base, portfolios.BacktestResult)
    assert isinstance(tilted, portfolios.BacktestResult)
    assert isinstance(mapping, pd.DataFrame)
    combined = portfolios.concatenate_backtests([base, tilted])
    evidence = fusion.build_fusion_evidence(
        combined.fund_returns,
        combined.fund_weights,
        mapping,
    )

    base_returns = combined.fund_returns.loc[
        combined.fund_returns["fund"].eq(fusion.BASE_FUND)
    ].sort_values("date")
    tilted_returns = combined.fund_returns.loc[
        combined.fund_returns["fund"].eq(fusion.TILTED_FUND)
    ].sort_values("date")
    assert tuple(base_returns["date"]) == tuple(tilted_returns["date"])
    assert tuple(base_returns["decision_date"]) == tuple(tilted_returns["decision_date"])
    assert evidence.comparison["observations"].nunique() == 1
    assert evidence.comparison["sample_start_date"].nunique() == 1
    assert evidence.comparison["sample_end_date"].nunique() == 1
    assert evidence.comparison["periods_per_year"].eq(252).all()
    assert evidence.comparison["risk_free_rate_annual"].eq(0.0).all()
    assert evidence.comparison["transaction_cost_bps"].eq(0.0).all()
    base_turnover = evidence.comparison.loc[
        evidence.comparison["role"].eq("base"),
        "average_monthly_target_turnover",
    ].iloc[0]
    assert base_turnover == pytest.approx(0.0, abs=1e-15)
    assert (
        evidence.sector_exposure.groupby("decision_date")[
            "sector_weight_difference"
        ].sum().abs()
        < 1e-12
    ).all()


def test_non_equity_baseline_is_rejected(
    synthetic_fusion_inputs: dict[str, object],
) -> None:
    base = synthetic_fusion_inputs["base"]
    returns = synthetic_fusion_inputs["returns"]
    signal = synthetic_fusion_inputs["signal"]
    mapping = synthetic_fusion_inputs["mapping"]
    assert isinstance(base, portfolios.BacktestResult)
    assert isinstance(returns, pd.DataFrame)
    assert isinstance(signal, pd.DataFrame)
    assert isinstance(mapping, pd.DataFrame)
    invalid = replace(
        base,
        fund_returns=base.fund_returns.assign(asset_family="combined"),
        fund_weights=base.fund_weights.assign(asset_family="combined"),
    )

    with pytest.raises(fusion.FusionValidationError, match="only to equities"):
        fusion.apply_sentiment_tilt(invalid, returns, signal, mapping)


@pytest.fixture(scope="module")
def official_core_funds() -> CoreFundBuild:
    return build_core_funds()


def test_official_fusion_uses_equities_and_matches_the_real_base_sample(
    official_core_funds: CoreFundBuild,
) -> None:
    returns = official_core_funds.artifacts.fund_returns
    weights = official_core_funds.artifacts.fund_weights
    comparison = official_core_funds.fusion_evidence.comparison
    assert set(returns["fund"]) == set(EXPECTED_FUND_IDENTITIES)
    matched = returns.loc[
        returns["fund"].isin([fusion.BASE_FUND, fusion.TILTED_FUND])
    ].groupby("fund")
    assert matched["date"].apply(tuple).map(hash).nunique() == 1
    assert matched["decision_date"].apply(tuple).map(hash).nunique() == 1
    tilted_weights = weights.loc[weights["fund"].eq(fusion.TILTED_FUND)]
    assert tilted_weights["ticker"].nunique() == 50
    assert not tilted_weights["ticker"].str.endswith("-USD").any()
    assert comparison["sample_start_date"].nunique() == 1
    assert comparison["sample_end_date"].nunique() == 1
    assert comparison["observations"].nunique() == 1
    assert comparison["rebalance_count"].nunique() == 1
    assert comparison["transaction_cost_bps"].eq(0.0).all()


def _direct_exploratory_signal(
    sector_index: pd.DataFrame,
    sector_name: str,
    decision_date: pd.Timestamp,
) -> dict[str, object]:
    """Rebuild the fixed exploratory signal without production helpers."""
    group = sector_index.loc[sector_index["sector"].eq(sector_name)].sort_values(
        "date"
    ).reset_index(drop=True)
    position = int(group.index[group["date"].eq(decision_date)][0])

    def trailing_at(target_position: int) -> tuple[float, float]:
        prior = group.iloc[max(0, target_position - 21) : target_position]
        observed = prior.loc[prior["has_observed_news"]]
        denominator = float(observed["ticker_coverage_share"].sum())
        trailing = (
            float(
                (
                    observed["raw_sector_compound"]
                    * observed["ticker_coverage_share"]
                ).sum()
                / denominator
            )
            if denominator > 0
            else np.nan
        )
        possible = float(group["possible_ticker_count"].iloc[0])
        coverage = float(prior["observed_ticker_count"].sum()) / (21 * possible)
        return trailing, coverage

    current, effective_coverage = trailing_at(position)
    prior_values = np.asarray(
        [trailing_at(index)[0] for index in range(position)], dtype=float
    )
    prior_values = prior_values[np.isfinite(prior_values)]
    assert len(prior_values) >= 60
    prior_mean = float(prior_values.mean())
    prior_std = float(prior_values.std(ddof=1))
    raw_z = (current - prior_mean) / prior_std
    clipped = float(np.clip(raw_z, -2.0, 2.0))
    return {
        "trailing": current,
        "coverage": effective_coverage,
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "prior_observations": len(prior_values),
        "raw_z": raw_z,
        "clipped_z": clipped,
        "coverage_adjusted_z": clipped * np.sqrt(effective_coverage),
        "window_start": group.iloc[max(0, position - 21)]["date"],
        "window_end": group.iloc[position - 1]["date"],
    }


def test_original_four_fund_subsets_are_exactly_unchanged_in_extended_build(
    official_core_funds: CoreFundBuild,
) -> None:
    locked = official_core_funds.locked_artifacts
    existing = set(locked.fund_returns["fund"])
    for table in ("fund_returns", "fund_weights", "performance_metrics"):
        expected = getattr(locked, table).reset_index(drop=True)
        actual_all = getattr(official_core_funds.artifacts, table)
        actual = actual_all.loc[
            actual_all["fund"].isin(existing), expected.columns
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_exploratory_zero_signal_is_exact_base_placebo_and_samples_match(
    official_core_funds: CoreFundBuild,
) -> None:
    signal = official_core_funds.exploratory_signal.copy()
    signal.loc[signal["coverage_adjusted_zscore"].notna(), "coverage_adjusted_zscore"] = 0.0
    placebo = fusion.apply_exploratory_coverage_tilt(
        official_core_funds.base.backtests,
        official_core_funds.base.equity_asset_returns,
        signal,
        official_core_funds.base.equity_sector_map,
    )
    base_returns = _official_fund_rows(
        official_core_funds, table="fund_returns", fund=fusion.BASE_FUND
    )
    base_weights = _official_fund_rows(
        official_core_funds, table="fund_weights", fund=fusion.BASE_FUND
    )
    placebo_returns = placebo.fund_returns.sort_values("date").reset_index(drop=True)
    placebo_weights = placebo.fund_weights.sort_values(
        ["decision_date", "ticker"]
    ).reset_index(drop=True)
    np.testing.assert_allclose(
        placebo_returns["daily_return"],
        base_returns["daily_return"],
        rtol=0.0,
        atol=NUMERIC_TOLERANCE,
    )
    np.testing.assert_allclose(
        placebo_weights["target_weight"],
        base_weights["target_weight"],
        rtol=0.0,
        atol=NUMERIC_TOLERANCE,
    )
    assert tuple(placebo_returns["date"]) == tuple(base_returns["date"])
    assert tuple(placebo_returns["decision_date"]) == tuple(
        base_returns["decision_date"]
    )


@pytest.mark.parametrize(
    "decision_date",
    ["2020-12-31", "2022-06-30", "2023-11-30"],
)
def test_three_exploratory_rebalances_match_direct_reconstruction(
    official_core_funds: CoreFundBuild,
    decision_date: str,
) -> None:
    decision = pd.Timestamp(decision_date)
    sector_index = official_core_funds.sentiment.sector_index
    mapping = official_core_funds.base.equity_sector_map.sort_values("ticker")
    expected_signal = {
        sector_name: _direct_exploratory_signal(
            sector_index, sector_name, decision
        )
        for sector_name in sorted(mapping["sector"].unique())
    }
    actual_signal = official_core_funds.exploratory_signal.loc[
        official_core_funds.exploratory_signal["date"].eq(decision)
    ].set_index("sector")
    for sector_name, expected in expected_signal.items():
        actual = actual_signal.loc[sector_name]
        assert actual["signal_window_start_date"] == expected["window_start"]
        assert actual["signal_window_end_date"] == expected["window_end"]
        assert actual["signal_window_end_date"] < decision
        for actual_name, expected_name in (
            ("trailing_coverage_weighted_sentiment", "trailing"),
            ("effective_coverage", "coverage"),
            ("expanding_prior_mean", "prior_mean"),
            ("expanding_prior_std", "prior_std"),
            ("raw_trailing_zscore", "raw_z"),
            ("clipped_trailing_zscore", "clipped_z"),
            ("coverage_adjusted_zscore", "coverage_adjusted_z"),
        ):
            assert actual[actual_name] == pytest.approx(
                expected[expected_name], abs=NUMERIC_TOLERANCE
            )
        assert actual["expanding_prior_observations"] == expected[
            "prior_observations"
        ]

    base = _official_fund_rows(
        official_core_funds, table="fund_weights", fund=fusion.BASE_FUND
    )
    base = base.loc[base["decision_date"].eq(decision)].sort_values("ticker")
    expected_z = mapping["sector"].map(
        {key: value["coverage_adjusted_z"] for key, value in expected_signal.items()}
    ).to_numpy(dtype=float)
    unnormalised = base["target_weight"].to_numpy(dtype=float) * (
        1.0 + 0.10 * expected_z
    )
    expected_weights = unnormalised / unnormalised.sum()
    actual_weights = _official_fund_rows(
        official_core_funds,
        table="fund_weights",
        fund=fusion.EXPLORATORY_FUND,
    )
    actual_weights = actual_weights.loc[
        actual_weights["decision_date"].eq(decision)
    ].sort_values("ticker")
    np.testing.assert_allclose(
        actual_weights["target_weight"],
        expected_weights,
        rtol=0.0,
        atol=NUMERIC_TOLERANCE,
    )
    np.testing.assert_allclose(
        actual_weights["unnormalised_weight"],
        unnormalised,
        rtol=0.0,
        atol=NUMERIC_TOLERANCE,
    )
    holding_returns = _official_fund_rows(
        official_core_funds,
        table="fund_returns",
        fund=fusion.EXPLORATORY_FUND,
    )
    first_holding = holding_returns.loc[
        holding_returns["decision_date"].eq(decision)
    ].iloc[0]
    asset_returns = official_core_funds.base.equity_asset_returns.set_index("date")
    expected_daily_return = float(
        asset_returns.loc[first_holding["date"], mapping["ticker"]].to_numpy(
            dtype=float
        )
        @ expected_weights
    )
    assert first_holding["daily_return"] == pytest.approx(
        expected_daily_return, abs=NUMERIC_TOLERANCE
    )


def test_exploratory_future_perturbation_cannot_change_past_path(
    official_core_funds: CoreFundBuild,
) -> None:
    cutoff = pd.Timestamp("2022-06-30")
    source = official_core_funds.sentiment.sector_index.copy()
    future_observed = (source["date"] > cutoff) & source["has_observed_news"]
    source.loc[future_observed, "raw_sector_compound"] *= -100.0
    changed_signal = sentiment.build_coverage_adjusted_trailing_signal(source)
    changed_backtest = fusion.apply_exploratory_coverage_tilt(
        official_core_funds.base.backtests,
        official_core_funds.base.equity_asset_returns,
        changed_signal,
        official_core_funds.base.equity_sector_map,
    )
    signal_columns = [
        "date",
        "sector",
        "trailing_coverage_weighted_sentiment",
        "coverage_adjusted_zscore",
    ]
    pd.testing.assert_frame_equal(
        official_core_funds.exploratory_signal.loc[
            official_core_funds.exploratory_signal["date"] <= cutoff,
            signal_columns,
        ].reset_index(drop=True),
        changed_signal.loc[changed_signal["date"] <= cutoff, signal_columns].reset_index(
            drop=True
        ),
        check_exact=True,
    )
    for table, date_column in (
        ("fund_weights", "decision_date"),
        ("fund_returns", "decision_date"),
    ):
        baseline = getattr(official_core_funds.exploratory_backtest, table)
        changed = getattr(changed_backtest, table)
        compare_columns = (
            ["decision_date", "ticker", "target_weight"]
            if table == "fund_weights"
            else ["date", "decision_date", "daily_return"]
        )
        pd.testing.assert_frame_equal(
            baseline.loc[baseline[date_column] <= cutoff, compare_columns].reset_index(
                drop=True
            ),
            changed.loc[changed[date_column] <= cutoff, compare_columns].reset_index(
                drop=True
            ),
            check_exact=True,
        )


def test_exploratory_constraints_identity_and_locked_design(
    official_core_funds: CoreFundBuild,
) -> None:
    weights = _official_fund_rows(
        official_core_funds,
        table="fund_weights",
        fund=fusion.EXPLORATORY_FUND,
    )
    returns = _official_fund_rows(
        official_core_funds,
        table="fund_returns",
        fund=fusion.EXPLORATORY_FUND,
    )
    base_returns = _official_fund_rows(
        official_core_funds, table="fund_returns", fund=fusion.BASE_FUND
    )
    assert weights["asset_family"].eq("equity").all()
    assert weights["method"].eq(fusion.EXPLORATORY_METHOD).all()
    assert not weights["ticker"].str.endswith("-USD").any()
    assert np.isfinite(weights["target_weight"]).all()
    assert (weights["target_weight"] >= 0.0).all()
    np.testing.assert_allclose(
        weights.groupby("decision_date")["target_weight"].sum(),
        1.0,
        rtol=0.0,
        atol=NUMERIC_TOLERANCE,
    )
    assert tuple(returns["date"]) == tuple(base_returns["date"])
    assert tuple(returns["decision_date"]) == tuple(base_returns["decision_date"])
    comparison = official_core_funds.exploratory_comparison
    assert comparison["periods_per_year"].eq(252).all()
    assert comparison["risk_free_rate_annual"].eq(0.0).all()
    assert comparison["transaction_cost_bps"].eq(0.0).all()
    assert comparison["signal_window_trading_days"].eq(21).all()
    assert comparison["tilt_strength"].eq(0.10).all()
    assert comparison["minimum_expanding_history"].eq(60).all()
    assert comparison["final_period_used_for_parameter_selection"].eq(False).all()


@pytest.fixture(scope="module")
def official_raw_sector_scores(
    official_core_funds: CoreFundBuild,
) -> pd.DataFrame:
    return _independent_raw_sector_scores(official_core_funds)


def test_official_zero_sentiment_placebo_matches_base_every_date(
    official_core_funds: CoreFundBuild,
) -> None:
    placebo_signal = official_core_funds.sentiment.sector_index.copy()
    available = placebo_signal["tradable_sector_zscore"].notna()
    placebo_signal.loc[available, "tradable_sector_zscore"] = 0.0
    placebo = fusion.apply_sentiment_tilt(
        official_core_funds.base.backtests,
        official_core_funds.base.equity_asset_returns,
        placebo_signal,
        official_core_funds.base.equity_sector_map,
    )
    base_returns = _official_fund_rows(
        official_core_funds,
        table="fund_returns",
        fund=fusion.BASE_FUND,
    )
    base_weights = _official_fund_rows(
        official_core_funds,
        table="fund_weights",
        fund=fusion.BASE_FUND,
    )
    placebo_returns = placebo.fund_returns.sort_values("date").reset_index(drop=True)
    placebo_weights = placebo.fund_weights.sort_values(
        ["decision_date", "ticker"]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        placebo_returns[["date", "decision_date"]],
        base_returns[["date", "decision_date"]],
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        placebo_weights[["decision_date", "ticker"]],
        base_weights[["decision_date", "ticker"]],
        check_exact=True,
    )
    weight_difference = (
        placebo_weights["target_weight"] - base_weights["target_weight"]
    ).abs()
    return_difference = (
        placebo_returns["daily_return"] - base_returns["daily_return"]
    ).abs()
    assert weight_difference.max() <= NUMERIC_TOLERANCE
    assert return_difference.max() <= NUMERIC_TOLERANCE
    assert placebo_weights["asset_family"].eq("equity").all()
    assert not placebo_weights["ticker"].str.endswith("-USD").any()
    assert np.isfinite(placebo_weights["target_weight"]).all()
    assert (placebo_weights["target_weight"] >= 0).all()
    np.testing.assert_allclose(
        placebo_weights.groupby("decision_date")["target_weight"].sum(),
        1.0,
        rtol=NUMERIC_TOLERANCE,
        atol=NUMERIC_TOLERANCE,
    )


@pytest.mark.parametrize(
    ("decision_date", "expected_first_holding_date"),
    [
        pytest.param("2020-12-31", "2021-01-04", id="first_rebalance"),
        pytest.param("2022-06-30", "2022-07-01", id="middle_rebalance"),
        pytest.param("2023-11-30", "2023-12-01", id="latest_rebalance"),
    ],
)
def test_three_official_rebalances_match_independent_manual_reconstruction(
    official_core_funds: CoreFundBuild,
    official_raw_sector_scores: pd.DataFrame,
    decision_date: str,
    expected_first_holding_date: str,
) -> None:
    decision = pd.Timestamp(decision_date)
    first_holding_date = pd.Timestamp(expected_first_holding_date)
    calendar = pd.DatetimeIndex(
        official_core_funds.sentiment.sector_index["date"].drop_duplicates()
    ).sort_values()
    decision_position = int(calendar.get_loc(decision))
    source_date = calendar[decision_position - 1]
    assert source_date < decision

    expected_signals: list[dict[str, object]] = []
    for sector_name in sorted(official_core_funds.base.equity_sector_map["sector"].unique()):
        sector_raw = official_raw_sector_scores.loc[
            official_raw_sector_scores["sector"].eq(sector_name)
        ].set_index("date")["raw_sector_compound"]
        prior = sector_raw.loc[sector_raw.index < source_date].dropna().to_numpy(
            dtype=float
        )
        source_value = float(sector_raw.loc[source_date])
        assert len(prior) >= 60
        prior_std = float(np.std(prior, ddof=1))
        assert prior_std > 0
        raw_zscore = (source_value - float(np.mean(prior))) / prior_std
        clipped_zscore = float(np.clip(raw_zscore, -2.0, 2.0))
        expected_signals.append(
            {
                "sector": sector_name,
                "expected_source_date": source_date,
                "expected_zscore": clipped_zscore,
            }
        )
    expected_signal = pd.DataFrame.from_records(expected_signals)
    actual_signal = official_core_funds.sentiment.sector_index.loc[
        official_core_funds.sentiment.sector_index["date"].eq(decision),
        ["sector", "tradable_signal_source_date", "tradable_sector_zscore"],
    ]
    signal_check = expected_signal.merge(
        actual_signal,
        on="sector",
        validate="one_to_one",
    )
    assert (
        signal_check["tradable_signal_source_date"]
        == signal_check["expected_source_date"]
    ).all()
    assert (signal_check["tradable_signal_source_date"] < decision).all()
    signal_error = (
        signal_check["expected_zscore"]
        - signal_check["tradable_sector_zscore"]
    ).abs()
    assert signal_error.max() <= NUMERIC_TOLERANCE

    mapping = official_core_funds.base.equity_sector_map.sort_values(
        "ticker"
    ).reset_index(drop=True)
    assert len(mapping) == 50
    assert not mapping["ticker"].str.endswith("-USD").any()
    expected_by_sector = expected_signal.set_index("sector")["expected_zscore"]
    base_weights = np.full(len(mapping), 1.0 / len(mapping))
    multipliers = 1.0 + fusion.TILT_STRENGTH * mapping["sector"].map(
        expected_by_sector
    ).to_numpy(dtype=float)
    unnormalised = base_weights * multipliers
    expected_weights = unnormalised / unnormalised.sum()
    assert expected_weights.sum() == pytest.approx(1.0, abs=NUMERIC_TOLERANCE)

    actual_weights = _official_fund_rows(
        official_core_funds,
        table="fund_weights",
        fund=fusion.TILTED_FUND,
    )
    actual_weights = actual_weights.loc[
        actual_weights["decision_date"].eq(decision)
    ].sort_values("ticker")
    assert actual_weights["ticker"].tolist() == mapping["ticker"].tolist()
    weight_error = np.abs(
        expected_weights - actual_weights["target_weight"].to_numpy(dtype=float)
    )
    assert weight_error.max() <= NUMERIC_TOLERANCE

    actual_returns = _official_fund_rows(
        official_core_funds,
        table="fund_returns",
        fund=fusion.TILTED_FUND,
    )
    first_holding = actual_returns.loc[
        actual_returns["decision_date"].eq(decision)
    ].iloc[0]
    assert first_holding["date"] == first_holding_date
    asset_returns = official_core_funds.base.equity_asset_returns.set_index("date")
    expected_return = float(
        asset_returns.loc[first_holding_date, mapping["ticker"]].to_numpy(dtype=float)
        @ expected_weights
    )
    assert abs(expected_return - first_holding["daily_return"]) <= NUMERIC_TOLERANCE


def test_official_predeclared_tilt_grid_is_a_dataset_specific_snapshot(
    official_core_funds: CoreFundBuild,
) -> None:
    assert fusion.TILT_STRENGTH == 0.10
    assert fusion.TRANSACTION_COST_BPS == 0.0
    base_returns = _official_fund_rows(
        official_core_funds,
        table="fund_returns",
        fund=fusion.BASE_FUND,
    )
    records: list[dict[str, object]] = []
    for strength in TILT_GRID:
        grid_returns, grid_exposure = _independent_grid_path(
            official_core_funds,
            strength,
        )
        statistics = _direct_performance(grid_returns["daily_return"])
        records.append(
            {
                "strength": strength,
                "sample_start_date": grid_returns["date"].min(),
                "sample_end_date": grid_returns["date"].max(),
                "observations": len(grid_returns),
                "periods_per_year": 252,
                "risk_free_rate_annual": 0.0,
                "annual_return_method": "geometric",
                "transaction_cost_bps": 0.0,
                **statistics,
                "mean_absolute_sector_exposure_change": float(
                    grid_exposure["sector_weight_difference"].abs().mean()
                ),
            }
        )
        if strength == 0.0:
            pd.testing.assert_series_equal(
                grid_returns["date"].reset_index(drop=True),
                base_returns["date"].reset_index(drop=True),
                check_names=False,
            )
            np.testing.assert_allclose(
                grid_returns["daily_return"],
                base_returns["daily_return"],
                rtol=NUMERIC_TOLERANCE,
                atol=NUMERIC_TOLERANCE,
            )

    results = pd.DataFrame.from_records(records).set_index("strength")
    assert results["sample_start_date"].nunique() == 1
    assert results["sample_end_date"].nunique() == 1
    assert results["observations"].eq(753).all()
    assert results["periods_per_year"].eq(252).all()
    assert results["risk_free_rate_annual"].eq(0.0).all()
    assert results["annual_return_method"].eq("geometric").all()
    assert results["transaction_cost_bps"].eq(0.0).all()

    # Dataset-specific regression evidence only: no grid value is selected here.
    expected_snapshot = pd.DataFrame(
        {
            "annualised_return": [
                0.126435292220,
                0.125914353800,
                0.125452560947,
                0.125049507502,
                0.124704977603,
            ],
            "annualised_volatility": [
                0.161661974777,
                0.161611889907,
                0.161587552215,
                0.161588675230,
                0.161615155921,
            ],
            "sharpe_ratio": [
                0.817387362651,
                0.814728768144,
                0.812288997787,
                0.810068617835,
                0.808068242803,
            ],
            "maximum_drawdown": [
                -0.203218626269,
                -0.202915252010,
                -0.202605752020,
                -0.202289303926,
                -0.201965085364,
            ],
            "mean_absolute_sector_exposure_change": [
                0.0,
                0.002943239380,
                0.005870446079,
                0.008785141602,
                0.011690765079,
            ],
        },
        index=pd.Index(TILT_GRID, name="strength"),
    )
    np.testing.assert_allclose(
        results[expected_snapshot.columns],
        expected_snapshot,
        rtol=NUMERIC_TOLERANCE,
        atol=NUMERIC_TOLERANCE,
    )


@pytest.mark.parametrize(
    ("fold_end", "expected_observations", "expected_rebalances", "snapshot"),
    [
        pytest.param(
            "2021-12-31",
            252,
            12,
            {
                "annualised_return_difference": -0.009367259325,
                "annualised_volatility_difference": 0.000261481736,
                "sharpe_ratio_difference": -0.062643365879,
                "maximum_drawdown_difference": 0.000311471773,
                "mean_absolute_exposure_difference": 0.006809502992,
                "maximum_absolute_exposure_difference": 0.020157876834,
            },
            id="fold_1_2021",
        ),
        pytest.param(
            "2022-12-30",
            503,
            24,
            {
                "annualised_return_difference": -0.003005985102,
                "annualised_volatility_difference": -0.000284012303,
                "sharpe_ratio_difference": -0.014921282116,
                "maximum_drawdown_difference": 0.000612874249,
                "mean_absolute_exposure_difference": 0.006451941735,
                "maximum_absolute_exposure_difference": 0.020157876834,
            },
            id="fold_2_2022",
        ),
        pytest.param(
            "2023-12-29",
            753,
            36,
            {
                "annualised_return_difference": -0.000982731273,
                "annualised_volatility_difference": -0.000074422562,
                "sharpe_ratio_difference": -0.005098364864,
                "maximum_drawdown_difference": 0.000612874249,
                "mean_absolute_exposure_difference": 0.005870446079,
                "maximum_absolute_exposure_difference": 0.020157876834,
            },
            id="fold_3_2023",
        ),
    ],
)
def test_official_expanding_temporal_fold_evidence_is_reproducible(
    official_core_funds: CoreFundBuild,
    fold_end: str,
    expected_observations: int,
    expected_rebalances: int,
    snapshot: dict[str, float],
) -> None:
    end_date = pd.Timestamp(fold_end)
    base = _official_fund_rows(
        official_core_funds,
        table="fund_returns",
        fund=fusion.BASE_FUND,
    )
    tilted = _official_fund_rows(
        official_core_funds,
        table="fund_returns",
        fund=fusion.TILTED_FUND,
    )
    base_fold = base.loc[base["date"] <= end_date].reset_index(drop=True)
    tilted_fold = tilted.loc[tilted["date"] <= end_date].reset_index(drop=True)
    pd.testing.assert_series_equal(
        base_fold["date"],
        tilted_fold["date"],
        check_names=False,
    )
    assert base_fold["date"].min() == pd.Timestamp("2021-01-04")
    assert base_fold["date"].max() == end_date
    assert len(base_fold) == expected_observations
    assert len(tilted_fold) == expected_observations
    decision_dates = pd.DatetimeIndex(tilted_fold["decision_date"].unique())
    assert len(decision_dates) == expected_rebalances

    base_metrics = _direct_performance(base_fold["daily_return"])
    tilted_metrics = _direct_performance(tilted_fold["daily_return"])
    differences = {
        f"{metric}_difference": tilted_metrics[metric] - base_metrics[metric]
        for metric in base_metrics
    }
    exposure = _independent_sector_exposure(
        official_core_funds,
        decision_dates,
    )["sector_weight_difference"].abs()
    differences["mean_absolute_exposure_difference"] = float(exposure.mean())
    differences["maximum_absolute_exposure_difference"] = float(exposure.max())
    assert np.isfinite(list(base_metrics.values())).all()
    assert np.isfinite(list(tilted_metrics.values())).all()
    assert np.isfinite(list(differences.values())).all()
    for name, expected_value in snapshot.items():
        assert differences[name] == pytest.approx(
            expected_value,
            rel=NUMERIC_TOLERANCE,
            abs=NUMERIC_TOLERANCE,
        )

    performance = official_core_funds.artifacts.performance_metrics.set_index("fund")
    matched_performance = performance.loc[[fusion.BASE_FUND, fusion.TILTED_FUND]]
    assert matched_performance["periods_per_year"].eq(252).all()
    assert matched_performance["risk_free_rate_annual"].eq(0.0).all()
    assert matched_performance["annual_return_method"].eq("geometric").all()
    assert official_core_funds.fusion_evidence.comparison[
        "transaction_cost_bps"
    ].eq(0.0).all()
