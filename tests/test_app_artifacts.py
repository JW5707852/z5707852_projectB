"""Artifact contracts, allocation arithmetic, and PortFoYou interaction tests."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.app_artifacts import (
    ARTIFACT_RELATIVE_PATHS,
    EXPECTED_FUND_IDENTITIES,
    EXPECTED_SECTORS,
    FUND_LABELS,
    FUND_RETURN_COLUMNS,
    FUND_WEIGHT_COLUMNS,
    PERFORMANCE_COLUMNS,
    SENTIMENT_COLUMNS,
    AppArtifactError,
    apply_annual_management_fee,
    calculate_allocation_scenario,
    drawdown_history,
    load_app_artifacts,
    validate_app_artifacts,
)
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "streamlit_app.py"


@pytest.fixture(scope="module")
def actual_artifacts():
    return load_app_artifacts(PROJECT_ROOT)


def test_required_app_artifacts_exist_and_load_from_project_results() -> None:
    assert set(ARTIFACT_RELATIVE_PATHS) == {
        "fund_returns",
        "fund_weights",
        "sector_sentiment",
        "performance_metrics",
    }
    for relative_path in ARTIFACT_RELATIVE_PATHS.values():
        path = PROJECT_ROOT / relative_path
        assert path.is_file(), relative_path
        assert path.is_relative_to(PROJECT_ROOT / "results")


def test_actual_artifact_schemas_names_dates_and_unique_keys(actual_artifacts) -> None:
    returns = actual_artifacts.fund_returns
    weights = actual_artifacts.fund_weights
    sentiment = actual_artifacts.sector_sentiment
    performance = actual_artifacts.performance_metrics

    assert set(returns) >= FUND_RETURN_COLUMNS
    assert set(weights) >= FUND_WEIGHT_COLUMNS
    assert set(sentiment) >= SENTIMENT_COLUMNS
    assert set(performance) >= PERFORMANCE_COLUMNS
    assert not returns.duplicated(["fund", "date"]).any()
    assert not weights.duplicated(["fund", "decision_date", "ticker"]).any()
    assert not sentiment.duplicated(["date", "sector"]).any()
    assert not performance.duplicated("fund").any()

    expected_funds = set(EXPECTED_FUND_IDENTITIES)
    assert set(returns["fund"]) == expected_funds
    assert set(weights["fund"]) == expected_funds
    assert set(performance["fund"]) == expected_funds
    assert set(sentiment["sector"]) == set(EXPECTED_SECTORS)

    return_ranges = returns.groupby("fund")["date"].agg(["min", "max", "size"])
    assert return_ranges.loc[return_ranges.index.str.startswith("crypto_")].shape[0] == 2
    assert return_ranges.loc[return_ranges.index.str.startswith("crypto_"), "size"].eq(1187).all()
    assert return_ranges.loc[~return_ranges.index.str.startswith("crypto_"), "size"].eq(753).all()
    assert pd.api.types.is_datetime64_any_dtype(returns["date"])
    assert pd.api.types.is_datetime64_any_dtype(weights["decision_date"])
    assert pd.api.types.is_datetime64_any_dtype(sentiment["date"])


def test_actual_artifact_joins_metrics_and_latest_holdings_reconcile(
    actual_artifacts,
) -> None:
    returns = actual_artifacts.fund_returns
    weights = actual_artifacts.fund_weights
    performance = actual_artifacts.performance_metrics.set_index("fund")

    identities = returns[["fund", "asset_family", "method"]].drop_duplicates()
    actual_identity = {
        row.fund: (row.asset_family, row.method)
        for row in identities.itertuples(index=False)
    }
    assert actual_identity == EXPECTED_FUND_IDENTITIES

    return_decisions = returns[["fund", "decision_date"]].drop_duplicates()
    weight_decisions = weights[["fund", "decision_date"]].drop_duplicates()
    matched = return_decisions.merge(
        weight_decisions,
        on=["fund", "decision_date"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    assert matched["_merge"].eq("both").all()
    assert len(matched) == len(return_decisions)

    sums = weights.groupby(["fund", "decision_date"])["target_weight"].sum()
    np.testing.assert_allclose(sums, 1.0, atol=1e-8, rtol=0.0)
    assert np.isfinite(weights["target_weight"]).all()
    assert weights["target_weight"].between(0, 1).all()

    current = weights.loc[weights["is_current"]]
    latest = weights.groupby("fund")["decision_date"].max()
    assert current.groupby("fund")["decision_date"].first().equals(latest)
    np.testing.assert_allclose(
        current.groupby("fund")["target_weight"].sum(),
        1.0,
        atol=1e-8,
        rtol=0.0,
    )
    assert performance["current_holdings_date"].equals(latest)

    final_growth = returns.groupby("fund")["growth_of_1"].last()
    np.testing.assert_allclose(
        performance["final_growth_of_1"],
        final_growth,
        atol=1e-10,
        rtol=1e-10,
    )


def test_allocation_scenario_matches_direct_numpy_calculation(actual_artifacts) -> None:
    returns = actual_artifacts.fund_returns
    funds = returns["fund"].drop_duplicates().tolist()
    raw = np.arange(1, len(funds) + 1, dtype=float)
    fractions = raw / raw.sum()
    allocations = dict(zip(funds, fractions, strict=True))
    initial_value = 125_000.0

    actual = calculate_allocation_scenario(returns, allocations, initial_value)
    wide = returns.pivot(index="date", columns="fund", values="daily_return")
    wide = wide.reindex(columns=funds).sort_index().dropna(how="any")
    expected_daily = wide.to_numpy() @ fractions
    expected_growth = np.cumprod(1.0 + expected_daily)

    np.testing.assert_allclose(
        actual.history["daily_return"], expected_daily, atol=1e-14, rtol=1e-14
    )
    np.testing.assert_allclose(
        actual.history["growth_of_1"], expected_growth, atol=1e-12, rtol=1e-12
    )
    assert actual.ending_value == pytest.approx(initial_value * expected_growth[-1])
    assert len(actual.history) == wide.shape[0]
    assert len(actual.summary) == len(funds)

    common_growth = (1.0 + wide).cumprod().iloc[-1]
    expected_end_by_fund = initial_value * fractions * common_growth.to_numpy()
    np.testing.assert_allclose(
        actual.summary["historical_ending_value"],
        expected_end_by_fund,
        atol=1e-8,
        rtol=1e-12,
    )


def test_app_drawdown_path_includes_the_starting_dollar_peak() -> None:
    returns = pd.DataFrame(
        {
            "fund": ["sample", "sample"],
            "date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "growth_of_1": [0.90, 1.08],
        }
    )

    actual = drawdown_history(returns, "sample")

    np.testing.assert_allclose(actual["drawdown"], [-0.10, 0.0], atol=1e-15)


def test_hypothetical_management_fee_matches_daily_compounding_by_hand() -> None:
    returns = pd.DataFrame(
        {
            "fund": ["one", "one"],
            "date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "daily_return": [0.10, 0.00],
        }
    )
    gross = calculate_allocation_scenario(returns, {"one": 1.0}, 100.0)
    zero_fee = apply_annual_management_fee(gross, 0.0)
    positive_fee = apply_annual_management_fee(gross, 0.252)
    daily_factor = (1.0 - 0.252) ** (1.0 / 252.0)
    expected_daily = np.array([(1.10 * daily_factor) - 1.0, daily_factor - 1.0])
    expected_ending = 100.0 * np.prod(1.0 + expected_daily)

    np.testing.assert_array_equal(
        zero_fee.history["fee_adjusted_daily_return"], gross.history["daily_return"]
    )
    assert zero_fee.fee_adjusted_ending_value == gross.ending_value
    np.testing.assert_allclose(
        positive_fee.history["fee_adjusted_daily_return"], expected_daily, atol=1e-15
    )
    assert positive_fee.fee_adjusted_ending_value == pytest.approx(expected_ending)
    assert positive_fee.fee_adjusted_ending_value < gross.ending_value


def test_management_fee_overlay_does_not_mutate_fund_returns(actual_artifacts) -> None:
    source = actual_artifacts.fund_returns.copy(deep=True)
    allocations = {fund: 1.0 / source["fund"].nunique() for fund in source["fund"].unique()}
    scenario = calculate_allocation_scenario(source, allocations, 100_000.0)
    apply_annual_management_fee(scenario, 0.01)
    pd.testing.assert_frame_equal(source, actual_artifacts.fund_returns, check_exact=True)


@pytest.mark.parametrize(
    ("allocations", "message"),
    [
        (
            {fund: 0.09 for fund in EXPECTED_FUND_IDENTITIES},
            "sum to one",
        ),
        (
            {
                fund: (-0.1 if position == 0 else 1.1 / 4)
                for position, fund in enumerate(EXPECTED_FUND_IDENTITIES)
            },
            "non-negative",
        ),
    ],
)
def test_allocation_rejects_invalid_weights(
    actual_artifacts,
    allocations: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_allocation_scenario(
            actual_artifacts.fund_returns,
            allocations,
            100_000,
        )


def test_missing_file_and_schema_errors_are_explicit(
    tmp_path: Path,
    actual_artifacts,
) -> None:
    with pytest.raises(
        AppArtifactError,
        match=r"results/data/fund_returns\.csv",
    ):
        load_app_artifacts(tmp_path)

    invalid_returns = actual_artifacts.fund_returns.drop(columns="daily_return")
    with pytest.raises(AppArtifactError, match="daily_return"):
        validate_app_artifacts(
            invalid_returns,
            actual_artifacts.fund_weights,
            actual_artifacts.sector_sentiment,
            actual_artifacts.performance_metrics,
        )


def test_app_source_and_dependency_split_keep_runtime_light() -> None:
    source = APP_PATH.read_text(encoding="utf-8").lower()
    for forbidden in (
        "data_access",
        "load_equity_prices",
        "load_crypto_prices",
        "load_news_headlines",
        "run_part_b",
        "st.secrets",
        "nltk",
    ):
        assert forbidden not in source

    deployment = (
        PROJECT_ROOT / "requirements.txt"
    ).read_text(encoding="utf-8").lower()
    development = (
        PROJECT_ROOT / "requirements-dev.txt"
    ).read_text(encoding="utf-8").lower()
    assert "nltk" not in deployment
    assert "nltk" in development
    assert "plotly" in deployment
    for build_only in ("scipy", "pyarrow", "requests", "matplotlib"):
        assert build_only not in deployment
        assert build_only in development

    config = (
        PROJECT_ROOT / ".streamlit/config.toml"
    ).read_text(encoding="utf-8")
    assert 'base = "light"' in config
    assert 'textColor = "#14283D"' in config
    assert '[theme.sidebar]' in config


def test_streamlit_investor_journey_and_error_interactions() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not app.exception
    assert not app.title
    assert not app.sidebar.radio
    assert app.segmented_control[0].value == "Fund Screener"
    assert any(header.value == "Fund Screener" for header in app.header)
    assert sum("PortFoYou" in item.value for item in app.markdown) == 1
    assert any(item.value == "**Data provenance**" for item in app.markdown)
    assert all(metric.help for metric in app.metric)
    for label in FUND_LABELS.values():
        assert any(label in html.unescape(item.value) for item in app.markdown)

    app.segmented_control[0].set_value("Fund Profile").run()
    assert not app.exception
    assert any(header.value == "Fund Profile" for header in app.header)
    app.selectbox[0].set_value("equity_sentiment_21d_coverage_tilt").run()
    assert not app.exception
    assert any("Exploratory strategy" in item.value for item in app.caption)
    assert len(app.metric) >= 5
    assert all(metric.help for metric in app.metric)
    app.selectbox[0].set_value("combined_active_sector_allocation").run()
    assert not app.exception
    assert any("Exploratory fixed design" in item.value for item in app.caption)
    app.selectbox[0].set_value("combined_growth_sector_allocation").run()
    assert not app.exception
    assert any("15% maximum sector weight" in item.value for item in app.caption)
    app.selectbox[0].set_value("combined_aggressive_sector_allocation").run()
    assert not app.exception
    assert any("Exploratory high-growth design" in item.value for item in app.caption)

    app.segmented_control[0].set_value("Fund Allocation").run()
    assert not app.exception
    assert any(header.value == "Fund Allocation" for header in app.header)
    assert not app.warning
    assert not app.error
    assert len(app.number_input) == 2 + len(FUND_LABELS)
    fee_input = app.number_input[1]
    assert fee_input.label == "Hypothetical annual management fee (%)"
    assert fee_input.value == 0.0
    assert {item.label for item in app.number_input[2:]} == {
        f"{label} (%)" for label in FUND_LABELS.values()
    }
    assert any(metric.label == "Gross ending value" for metric in app.metric)
    assert any(metric.label == "Fee-adjusted ending value" for metric in app.metric)
    assert any(metric.label == "Estimated fee drag" for metric in app.metric)
    assert any(metric.label == "Fee-adjusted total return" for metric in app.metric)
    assert any("0.00% annual management fee" in item.value for item in app.caption)
    fee_input.set_value(1.0).run()
    assert not app.exception
    assert any("1.00% annual management fee" in item.value for item in app.caption)
    original_weight = app.number_input[2].value
    app.number_input[2].set_value(0.0).run()
    assert any("Target weights must total 100%" in error.value for error in app.error)
    app.number_input[2].set_value(original_weight).run()
    assert not app.error

    app.segmented_control[0].set_value("Portfolio Simulator").run()
    assert not app.exception
    assert any(header.value == "Portfolio Simulator" for header in app.header)
    assert any("individual US stocks and cryptoassets" in item.value for item in app.markdown)
    assert any(item.label == "+ Add asset" for item in app.button)
    assert any(item.label == "Use equal weights" for item in app.button)
    assert len([item for item in app.selectbox if item.label.startswith("Type ")]) == 2
    assert len([item for item in app.selectbox if item.label.startswith("Asset ")]) == 2
    assert any(metric.label == "Annual volatility" for metric in app.metric)
    assert any(metric.label == "Maximum drawdown" for metric in app.metric)
    weight_inputs = [item for item in app.number_input if item.label.startswith("Weight ")]
    weight_inputs[0].set_value(40.0).run()
    assert any("Target weights must total 100%" in error.value for error in app.error)
    weight_inputs = [item for item in app.number_input if item.label.startswith("Weight ")]
    weight_inputs[1].set_value(60.0).run()
    assert not app.error
    add_button = next(item for item in app.button if item.label == "+ Add asset")
    add_button.click().run()
    assert len([item for item in app.selectbox if item.label.startswith("Asset ")]) == 3
    assert not app.error
    weight_inputs = [item for item in app.number_input if item.label.startswith("Weight ")]
    weight_inputs[2].set_value(10.0).run()
    assert any("Target weights must total 100%" in error.value for error in app.error)
    equal_button = next(item for item in app.button if item.label == "Use equal weights")
    equal_button.click().run()
    assert not app.error
    assert any(subheader.value == "Current asset-type mix" for subheader in app.subheader)

    app.segmented_control[0].set_value("Sector Sentiment").run()
    assert not app.exception
    assert any(header.value == "Sector Sentiment" for header in app.header)
    assert set(app.multiselect[0].value) == {
        "Tech",
        "Financials",
        "Energy",
        "Healthcare",
    }
    series_radio = next(item for item in app.radio if item.key == "sentiment_series")
    series_radio.set_value("Daily").run()
    assert not app.exception
    assert any("appear as gaps" in item.value for item in app.caption)
    app.multiselect[0].set_value([]).run()
    assert any("Select at least one sector" in error.value for error in app.error)

    source = APP_PATH.read_text(encoding="utf-8")
    for unclear_label in (
        "Best growth",
        "Performance snapshot",
        "Out-of-sample growth",
        "Risk and return map",
        "Scenario breakdown",
        "Investor journey",
        "Neural Sentiment Robustness",
    ):
        assert unclear_label not in source
    assert all("1/N" not in label and "Tilt" not in label for label in FUND_LABELS.values())
    assert not app.code


def test_optional_simulator_artifact_failure_is_isolated_from_core_pages() -> None:
    app = AppTest.from_string(
        """
import streamlit_app
from src.custom_portfolio import CustomPortfolioError

def fail_asset_load():
    raise CustomPortfolioError(
        "Required custom-portfolio artifact is missing: "
        "results/data/investable_asset_returns.csv. Rebuild project artifacts."
    )

streamlit_app._load_asset_returns = fail_asset_load
streamlit_app.main()
""",
        default_timeout=20,
    ).run()

    assert not app.exception
    assert not app.error
    assert any(header.value == "Fund Screener" for header in app.header)

    app.segmented_control[0].set_value("Portfolio Simulator").run()
    assert not app.exception
    assert any(header.value == "Portfolio Simulator" for header in app.header)
    assert any(
        "results/data/investable_asset_returns.csv" in error.value
        for error in app.error
    )

    app.segmented_control[0].set_value("Fund Profile").run()
    assert not app.exception
    assert not app.error
    assert any(header.value == "Fund Profile" for header in app.header)


def test_core_artifact_failure_reports_the_specific_missing_path() -> None:
    app = AppTest.from_string(
        """
import streamlit_app
from src.app_artifacts import AppArtifactError

def fail_core_load():
    raise AppArtifactError(
        "Required PortFoYou artifact is missing: results/data/fund_returns.csv. "
        "Rebuild the project artifacts locally before opening the app."
    )

streamlit_app._load_artifacts = fail_core_load
streamlit_app.main()
""",
        default_timeout=20,
    ).run()

    assert not app.exception
    assert any("Core market research data is unavailable" in item.value for item in app.error)
    assert any("results/data/fund_returns.csv" in item.value for item in app.error)
