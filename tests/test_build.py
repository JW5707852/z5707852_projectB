"""Focused tests for deterministic core-build schemas and consistency gates."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest
from scripts import run_part_b
from src import metrics


@dataclass(frozen=True)
class CoreFrames:
    artifacts: metrics.FundArtifacts
    sector_index: pd.DataFrame


@pytest.fixture(scope="module")
def core_frames() -> CoreFrames:
    returns = pd.read_csv(
        run_part_b.FUND_ARTIFACT_PATHS["fund_returns"],
        parse_dates=["date", "decision_date"],
    )
    weights = pd.read_csv(
        run_part_b.FUND_ARTIFACT_PATHS["fund_weights"],
        parse_dates=[
            "date",
            "decision_date",
            "training_start_date",
            "training_end_date",
            "first_holding_date",
            "tradable_signal_source_date",
            "signal_window_start_date",
            "signal_window_end_date",
            "latest_raw_news_date_used",
        ],
    )
    performance = pd.read_csv(
        run_part_b.FUND_ARTIFACT_PATHS["performance_metrics"],
        parse_dates=[
            "as_of_date",
            "sample_start_date",
            "sample_end_date",
            "current_holdings_date",
        ],
    )
    sector_index = pd.read_csv(
        run_part_b.SENTIMENT_ARTIFACT_PATH,
        parse_dates=["date", "tradable_signal_source_date"],
    )
    return CoreFrames(
        metrics.FundArtifacts(returns, weights, performance),
        sector_index,
    )


def test_required_output_paths_are_exactly_project_root_relative() -> None:
    run_part_b.validate_core_paths()
    relative_paths = {
        "fund_returns": run_part_b.FUND_ARTIFACT_PATHS[
            "fund_returns"
        ].relative_to(run_part_b.PROJECT_ROOT).as_posix(),
        "fund_weights": run_part_b.FUND_ARTIFACT_PATHS[
            "fund_weights"
        ].relative_to(run_part_b.PROJECT_ROOT).as_posix(),
        "sector_sentiment_index": run_part_b.SENTIMENT_ARTIFACT_PATH.relative_to(
            run_part_b.PROJECT_ROOT
        ).as_posix(),
        "performance_metrics": run_part_b.FUND_ARTIFACT_PATHS[
            "performance_metrics"
        ].relative_to(run_part_b.PROJECT_ROOT).as_posix(),
    }
    assert relative_paths == {
        "fund_returns": "results/data/fund_returns.csv",
        "fund_weights": "results/data/fund_weights.csv",
        "sector_sentiment_index": "results/data/sector_sentiment_index.csv",
        "performance_metrics": "results/tables/performance_metrics.csv",
    }


def test_actual_required_artifacts_pass_the_consolidated_gate(
    core_frames: CoreFrames,
) -> None:
    summary = run_part_b.validate_core_artifacts(
        core_frames.artifacts,
        core_frames.sector_index,
    )
    assert summary["funds"] == sorted(run_part_b.EXPECTED_FUND_IDENTITIES)
    assert summary["fund_return_rows"] == int(
        core_frames.artifacts.performance_metrics["observations"].sum()
    )
    assert summary["fund_weight_rows"] == len(core_frames.artifacts.fund_weights)
    assert summary["metric_rows"] == len(run_part_b.EXPECTED_FUND_IDENTITIES)
    assert summary["sector_sentiment_rows"] == 10_060


def test_schema_order_change_is_rejected(core_frames: CoreFrames) -> None:
    invalid_returns = core_frames.artifacts.fund_returns[
        list(reversed(run_part_b.CORE_RETURN_SCHEMA))
    ]
    invalid = metrics.FundArtifacts(
        invalid_returns,
        core_frames.artifacts.fund_weights,
        core_frames.artifacts.performance_metrics,
    )
    with pytest.raises(run_part_b.CoreBuildValidationError, match="schema differs"):
        run_part_b.validate_core_artifacts(invalid, core_frames.sector_index)


def test_duplicate_return_key_is_rejected(core_frames: CoreFrames) -> None:
    duplicated = pd.concat(
        [
            core_frames.artifacts.fund_returns,
            core_frames.artifacts.fund_returns.iloc[[0]],
        ],
        ignore_index=True,
    ).sort_values(["fund", "date"], kind="mergesort")
    invalid = metrics.FundArtifacts(
        duplicated,
        core_frames.artifacts.fund_weights,
        core_frames.artifacts.performance_metrics,
    )
    with pytest.raises(run_part_b.CoreBuildValidationError, match="duplicate fund-date"):
        run_part_b.validate_core_artifacts(invalid, core_frames.sector_index)


def test_weight_sum_failure_is_rejected(core_frames: CoreFrames) -> None:
    invalid_weights = core_frames.artifacts.fund_weights.copy()
    invalid_weights.loc[invalid_weights.index[0], "target_weight"] += 0.01
    invalid = metrics.FundArtifacts(
        core_frames.artifacts.fund_returns,
        invalid_weights,
        core_frames.artifacts.performance_metrics,
    )
    with pytest.raises(run_part_b.CoreBuildValidationError, match="do not sum to one"):
        run_part_b.validate_core_artifacts(invalid, core_frames.sector_index)


def test_latest_holdings_metric_mismatch_is_rejected(core_frames: CoreFrames) -> None:
    invalid_performance = core_frames.artifacts.performance_metrics.copy()
    invalid_performance.loc[0, "current_holdings_date"] -= pd.Timedelta(days=1)
    invalid = metrics.FundArtifacts(
        core_frames.artifacts.fund_returns,
        core_frames.artifacts.fund_weights,
        invalid_performance,
    )
    with pytest.raises(
        run_part_b.CoreBuildValidationError,
        match="holdings dates are not latest",
    ):
        run_part_b.validate_core_artifacts(invalid, core_frames.sector_index)


def test_sentiment_missingness_policy_is_enforced(core_frames: CoreFrames) -> None:
    invalid_sector_index = core_frames.sector_index.copy()
    no_news_row = invalid_sector_index.index[
        ~invalid_sector_index["has_observed_news"]
    ][0]
    invalid_sector_index.loc[no_news_row, "raw_sector_compound"] = 0.0
    with pytest.raises(
        run_part_b.CoreBuildValidationError,
        match="raw sentiment missingness is inconsistent",
    ):
        run_part_b.validate_core_artifacts(
            core_frames.artifacts,
            invalid_sector_index,
        )
