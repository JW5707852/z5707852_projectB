"""Orchestrate reproducible Part B fund and sentiment builds."""
from __future__ import annotations

import pathlib
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import (
    active_sector,
    custom_portfolio,
    etl,
    features,
    fusion,
    metrics,
    portfolios,
    sentiment,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
FUND_ARTIFACT_PATHS = {
    "fund_returns": PROJECT_ROOT / "results/data/fund_returns.csv",
    "fund_weights": PROJECT_ROOT / "results/data/fund_weights.csv",
    "performance_metrics": PROJECT_ROOT / "results/tables/performance_metrics.csv",
}
SENTIMENT_ARTIFACT_PATH = PROJECT_ROOT / "results/data/sector_sentiment_index.csv"
INVESTABLE_ASSET_RETURNS_PATH = (
    PROJECT_ROOT / custom_portfolio.INVESTABLE_ASSET_RELATIVE_PATH
)
FUSION_COMPARISON_PATH = PROJECT_ROOT / "results/tables/fusion_comparison.csv"
FUSION_SECTOR_EXPOSURE_PATH = (
    PROJECT_ROOT / "results/data/fusion_sector_exposure.csv"
)
EXPLORATORY_SIGNAL_PATH = (
    PROJECT_ROOT / "results/data/sector_sentiment_21d_coverage.csv"
)
EXPLORATORY_COMPARISON_PATH = (
    PROJECT_ROOT / "results/tables/fusion_exploratory_comparison.csv"
)
PREDICTABILITY_PATH = PROJECT_ROOT / "results/tables/sentiment_predictability.csv"
PREDICTABILITY_SUMMARY_PATH = (
    PROJECT_ROOT / "results/tables/sentiment_predictability_summary.csv"
)
ACTIVE_SECTOR_EXPOSURE_PATH = (
    PROJECT_ROOT / "results/data/active_sector_allocation_exposure.csv"
)
GROWTH_SECTOR_EXPOSURE_PATH = (
    PROJECT_ROOT / "results/data/growth_sector_allocation_exposure.csv"
)
AGGRESSIVE_SECTOR_EXPOSURE_PATH = (
    PROJECT_ROOT / "results/data/aggressive_sector_allocation_exposure.csv"
)
BUILD_RANDOM_SEED = 5545
WEIGHT_SUM_TOLERANCE = 1e-10
EXPECTED_FUND_IDENTITIES = {
    "combined_equal_weight": ("combined", "equal_weight"),
    "combined_min_variance": ("combined", "min_variance"),
    active_sector.FUND: (active_sector.ASSET_FAMILY, active_sector.METHOD),
    active_sector.GROWTH_FUND: (
        active_sector.ASSET_FAMILY,
        active_sector.GROWTH_METHOD,
    ),
    active_sector.AGGRESSIVE_FUND: (
        active_sector.ASSET_FAMILY,
        active_sector.AGGRESSIVE_METHOD,
    ),
    "equity_equal_weight": ("equity", "equal_weight"),
    "equity_sentiment_21d_coverage_tilt": (
        "equity",
        "exploratory_sentiment_21d_coverage_tilt",
    ),
    "equity_sentiment_tilt": ("equity", "sentiment_tilt"),
    "crypto_equal_weight": ("crypto", "equal_weight"),
    "crypto_min_variance": ("crypto", "min_variance"),
}
CORE_RETURN_SCHEMA = (
    "fund",
    "asset_family",
    "method",
    "date",
    "decision_date",
    "daily_return",
    "growth_of_1",
)
CORE_WEIGHT_SCHEMA = (
    "fund",
    "asset_family",
    "method",
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
)
CORE_METRIC_SCHEMA = (
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
    "annual_return_method",
    "final_growth_of_1",
    "annualised_return",
    "annualised_volatility",
    "annualised_mean_excess_return",
    "sharpe_ratio",
    "maximum_drawdown",
)
CORE_SENTIMENT_SCHEMA = (
    "date",
    "sector",
    "raw_sector_compound",
    "observed_ticker_count",
    "headline_count",
    "possible_ticker_count",
    "ticker_coverage_share",
    "has_observed_news",
    "prior_observations_for_raw_z",
    "raw_expanding_zscore",
    "raw_zscore_clipped",
    "tradable_sector_zscore",
    "tradable_signal_source_date",
    "signal_prior_observations",
    "sentiment_model",
    "nltk_version",
    "vader_lexicon",
    "lexicon_extension",
    "score_name",
    "compound_score_definition",
    "text_input_column",
    "text_preprocessing",
    "raw_index_purpose",
    "tradable_signal_purpose",
    "min_history_observations",
    "zscore_clip_bound",
    "signal_lag_trading_days",
    "zscore_std_ddof",
)
WEIGHT_ALLOWED_MISSING_COLUMNS = {
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
SENTIMENT_ALLOWED_MISSING_COLUMNS = {
    "raw_sector_compound",
    "raw_expanding_zscore",
    "raw_zscore_clipped",
    "tradable_sector_zscore",
    "tradable_signal_source_date",
    "signal_prior_observations",
}


class CoreBuildValidationError(RuntimeError):
    """Raised when generated core artifacts are not mutually reproducible."""


@dataclass(frozen=True)
class BaseFundBuild:
    """Base-fund outputs plus source returns needed for independent audits."""

    backtests: portfolios.BacktestResult
    combined_asset_returns: pd.DataFrame
    equity_asset_returns: pd.DataFrame
    equity_sector_map: pd.DataFrame
    return_missingness_audit: pd.DataFrame
    variation_summary: dict[str, float]
    artifacts: metrics.FundArtifacts


@dataclass(frozen=True)
class CryptoFundBuild:
    """Crypto-native fund outputs kept separate from equity-calendar funds."""

    backtests: portfolios.BacktestResult
    crypto_asset_returns: pd.DataFrame
    variation_summary: dict[str, float]


@dataclass(frozen=True)
class CoreFundBuild:
    """Locked core funds plus one separately labelled exploratory extension."""

    base: BaseFundBuild
    crypto: CryptoFundBuild
    sentiment: sentiment.SentimentBuild
    fusion_backtest: portfolios.BacktestResult
    exploratory_signal: pd.DataFrame
    exploratory_backtest: portfolios.BacktestResult
    active_sector_backtest: portfolios.BacktestResult
    growth_sector_backtest: portfolios.BacktestResult
    aggressive_sector_backtest: portfolios.BacktestResult
    backtests: portfolios.BacktestResult
    locked_artifacts: metrics.FundArtifacts
    non_crypto_artifacts: metrics.FundArtifacts
    artifacts: metrics.FundArtifacts
    fusion_evidence: fusion.FusionEvidence
    exploratory_evidence: fusion.FusionEvidence
    exploratory_comparison: pd.DataFrame
    predictability: pd.DataFrame
    predictability_summary: pd.DataFrame
    active_sector_exposure: pd.DataFrame
    growth_sector_exposure: pd.DataFrame
    aggressive_sector_exposure: pd.DataFrame


def _require_exact_schema(
    frame: pd.DataFrame,
    expected: tuple[str, ...],
    name: str,
) -> None:
    actual = tuple(frame.columns)
    if actual != expected:
        raise CoreBuildValidationError(
            f"{name} schema differs: expected {expected}, found {actual}"
        )


def _normalise_required_dates(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    name: str,
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        parsed = pd.to_datetime(result[column], errors="coerce", utc=True)
        if parsed.isna().any():
            raise CoreBuildValidationError(f"{name}.{column} contains missing dates")
        result[column] = parsed.dt.tz_convert(None).dt.normalize()
    return result


def _require_sorted(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    actual = frame[keys].reset_index(drop=True)
    expected = frame.sort_values(keys, kind="mergesort")[keys].reset_index(drop=True)
    if not actual.equals(expected):
        raise CoreBuildValidationError(f"{name} is not stably sorted by {keys}")


def _require_expected_fund_identities(frame: pd.DataFrame, name: str) -> None:
    identifiers = frame[["fund", "asset_family", "method"]]
    if identifiers.isna().any().any():
        raise CoreBuildValidationError(f"{name} contains missing fund identifiers")
    unique = identifiers.drop_duplicates()
    if unique["fund"].duplicated().any():
        raise CoreBuildValidationError(f"{name} maps a fund to multiple identities")
    actual = {
        row.fund: (row.asset_family, row.method)
        for row in unique.itertuples(index=False)
    }
    if actual != EXPECTED_FUND_IDENTITIES:
        raise CoreBuildValidationError(
            f"{name} fund identities differ: {actual}"
        )


def validate_core_artifacts(
    artifacts: metrics.FundArtifacts,
    sector_index: pd.DataFrame,
) -> dict[str, object]:
    """Validate the exact schemas and joins of the four required CSV frames."""
    _require_exact_schema(artifacts.fund_returns, CORE_RETURN_SCHEMA, "fund_returns")
    _require_exact_schema(artifacts.fund_weights, CORE_WEIGHT_SCHEMA, "fund_weights")
    _require_exact_schema(
        artifacts.performance_metrics,
        CORE_METRIC_SCHEMA,
        "performance_metrics",
    )
    _require_exact_schema(
        sector_index,
        CORE_SENTIMENT_SCHEMA,
        "sector_sentiment_index",
    )

    returns = _normalise_required_dates(
        artifacts.fund_returns,
        ("date", "decision_date"),
        "fund_returns",
    )
    weights = _normalise_required_dates(
        artifacts.fund_weights,
        (
            "date",
            "decision_date",
            "training_start_date",
            "training_end_date",
            "first_holding_date",
        ),
        "fund_weights",
    )
    performance = _normalise_required_dates(
        artifacts.performance_metrics,
        (
            "as_of_date",
            "sample_start_date",
            "sample_end_date",
            "current_holdings_date",
        ),
        "performance_metrics",
    )
    sectors = _normalise_required_dates(
        sector_index,
        ("date",),
        "sector_sentiment_index",
    )

    for frame, name in (
        (returns, "fund_returns"),
        (weights, "fund_weights"),
        (performance, "performance_metrics"),
    ):
        _require_expected_fund_identities(frame, name)

    if returns.isna().any().any():
        raise CoreBuildValidationError("fund_returns contains missing values")
    unexpected_weight_missing = {
        column
        for column in weights.columns[weights.isna().any()]
        if column not in WEIGHT_ALLOWED_MISSING_COLUMNS
    }
    if unexpected_weight_missing:
        raise CoreBuildValidationError(
            "fund_weights contains unexpected missing values in "
            f"{sorted(unexpected_weight_missing)}"
        )
    if performance.isna().any().any():
        raise CoreBuildValidationError("performance_metrics contains missing values")
    unexpected_sentiment_missing = {
        column
        for column in sectors.columns[sectors.isna().any()]
        if column not in SENTIMENT_ALLOWED_MISSING_COLUMNS
    }
    if unexpected_sentiment_missing:
        raise CoreBuildValidationError(
            "sector_sentiment_index contains unexpected missing values in "
            f"{sorted(unexpected_sentiment_missing)}"
        )

    if returns.duplicated(["fund", "date"]).any():
        raise CoreBuildValidationError("fund_returns has duplicate fund-date keys")
    if weights.duplicated(["fund", "decision_date", "ticker"]).any():
        raise CoreBuildValidationError("fund_weights has duplicate target-weight keys")
    if performance.duplicated(["fund"]).any():
        raise CoreBuildValidationError("performance_metrics has duplicate funds")
    if sectors.duplicated(["date", "sector"]).any():
        raise CoreBuildValidationError(
            "sector_sentiment_index has duplicate date-sector keys"
        )
    _require_sorted(returns, ["fund", "date"], "fund_returns")
    _require_sorted(
        weights,
        ["fund", "decision_date", "ticker"],
        "fund_weights",
    )
    _require_sorted(performance, ["fund"], "performance_metrics")
    _require_sorted(sectors, ["date", "sector"], "sector_sentiment_index")

    return_values = returns[["daily_return", "growth_of_1"]].to_numpy(dtype=float)
    if not np.isfinite(return_values).all():
        raise CoreBuildValidationError("fund returns or wealth values are non-finite")
    target_weights = pd.to_numeric(weights["target_weight"], errors="coerce")
    if not np.isfinite(target_weights).all():
        raise CoreBuildValidationError("target weights are non-finite")
    if (target_weights < -WEIGHT_SUM_TOLERANCE).any() or (
        target_weights > 1.0 + WEIGHT_SUM_TOLERANCE
    ).any():
        raise CoreBuildValidationError("target weights violate long-only bounds")
    weight_sums = weights.assign(target_weight=target_weights).groupby(
        ["fund", "decision_date"]
    )["target_weight"].sum()
    if not np.allclose(
        weight_sums,
        1.0,
        atol=WEIGHT_SUM_TOLERANCE,
        rtol=0.0,
    ):
        raise CoreBuildValidationError("target weights do not sum to one")
    if not weights["date"].equals(weights["decision_date"]):
        raise CoreBuildValidationError("generic weight date differs from decision_date")

    expected_current = weights["decision_date"].eq(
        weights.groupby("fund")["decision_date"].transform("max")
    )
    if not weights["is_current"].astype(bool).equals(expected_current):
        raise CoreBuildValidationError("is_current does not select latest holdings")
    current = weights.loc[expected_current]
    current_sums = current.groupby("fund")["target_weight"].sum()
    if not np.allclose(
        current_sums,
        1.0,
        atol=WEIGHT_SUM_TOLERANCE,
        rtol=0.0,
    ):
        raise CoreBuildValidationError("latest holdings do not sum to one")

    for fund, group in returns.groupby("fund", sort=True):
        dates = group["date"]
        if not dates.is_monotonic_increasing or dates.duplicated().any():
            raise CoreBuildValidationError(
                f"fund_returns must be sorted and unique within fund {fund}"
            )
    return_decisions = (
        returns[["fund", "decision_date"]]
        .drop_duplicates()
        .sort_values(["fund", "decision_date"])
        .reset_index(drop=True)
    )
    weight_decisions = (
        weights[["fund", "decision_date"]]
        .drop_duplicates()
        .sort_values(["fund", "decision_date"])
        .reset_index(drop=True)
    )
    if not return_decisions.equals(weight_decisions):
        raise CoreBuildValidationError("return and weight rebalance keys differ")

    return_summary = returns.groupby("fund", sort=True).agg(
        sample_start_date=("date", "min"),
        sample_end_date=("date", "max"),
        observations=("date", "size"),
        final_growth_of_1=("growth_of_1", "last"),
    )
    metric_summary = performance.set_index("fund").sort_index()
    for column in ("sample_start_date", "sample_end_date", "observations"):
        if not return_summary[column].equals(metric_summary[column]):
            raise CoreBuildValidationError(f"metric join differs for {column}")
    if not np.allclose(
        return_summary["final_growth_of_1"],
        metric_summary["final_growth_of_1"],
        atol=1e-12,
        rtol=1e-12,
    ):
        raise CoreBuildValidationError("metric growth does not match fund returns")
    if not metric_summary["as_of_date"].equals(metric_summary["sample_end_date"]):
        raise CoreBuildValidationError("metric as-of dates differ from sample ends")
    latest_weight_dates = weights.groupby("fund")["decision_date"].max().sort_index()
    if not latest_weight_dates.equals(metric_summary["current_holdings_date"]):
        raise CoreBuildValidationError("fact-sheet holdings dates are not latest")
    expected_periods = metric_summary["asset_family"].map({"crypto": 365}).fillna(252)
    if not metric_summary["periods_per_year"].eq(expected_periods).all():
        raise CoreBuildValidationError(
            "core metrics must use 365 periods for crypto and 252 otherwise"
        )
    if not metric_summary["risk_free_rate_annual"].eq(0.0).all():
        raise CoreBuildValidationError("core metrics must use a 0% risk-free rate")
    if not metric_summary["annual_return_method"].eq("geometric").all():
        raise CoreBuildValidationError("core annual returns must be geometric")

    if sectors[["date", "sector"]].isna().any().any():
        raise CoreBuildValidationError("sector index has missing identifiers")
    sector_names = sorted(sectors["sector"].unique())
    sector_dates = sectors["date"].drop_duplicates().sort_values()
    if len(sector_names) != 10 or len(sectors) != len(sector_dates) * 10:
        raise CoreBuildValidationError("sector index is not a complete ten-sector grid")
    if not sectors.groupby("date")["sector"].nunique().eq(10).all():
        raise CoreBuildValidationError("a sector-index date does not contain ten sectors")
    if not sectors["possible_ticker_count"].eq(5).all():
        raise CoreBuildValidationError("each supplied sector must contain five tickers")
    observed = pd.to_numeric(sectors["observed_ticker_count"], errors="coerce")
    possible = pd.to_numeric(sectors["possible_ticker_count"], errors="coerce")
    coverage = pd.to_numeric(sectors["ticker_coverage_share"], errors="coerce")
    if not np.isfinite(observed).all() or not np.isfinite(coverage).all():
        raise CoreBuildValidationError("sector coverage fields are non-finite")
    if (observed < 0).any() or (observed > possible).any():
        raise CoreBuildValidationError("sector observed counts are outside bounds")
    if not np.allclose(coverage, observed / possible, atol=1e-12, rtol=0.0):
        raise CoreBuildValidationError("sector ticker coverage is inconsistent")
    has_news = sectors["has_observed_news"].astype(bool)
    if not has_news.equals(observed.gt(0)):
        raise CoreBuildValidationError("sector news flag disagrees with ticker count")
    if not sectors["raw_sector_compound"].isna().equals(~has_news):
        raise CoreBuildValidationError("raw sentiment missingness is inconsistent")
    available_signal = sectors["tradable_sector_zscore"].notna()
    source_dates = pd.to_datetime(
        sectors["tradable_signal_source_date"], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()
    if source_dates.loc[available_signal].isna().any() or not (
        source_dates.loc[available_signal] < sectors.loc[available_signal, "date"]
    ).all():
        raise CoreBuildValidationError("tradable sentiment source dates are invalid")
    non_crypto_returns = returns.loc[~returns["asset_family"].eq("crypto")]
    if not non_crypto_returns["date"].isin(sector_dates).all():
        raise CoreBuildValidationError("fund returns fall outside the sentiment calendar")

    return {
        "funds": sorted(returns["fund"].unique()),
        "fund_return_rows": len(returns),
        "fund_weight_rows": len(weights),
        "metric_rows": len(performance),
        "sector_sentiment_rows": len(sectors),
        "sample_start_date": returns["date"].min(),
        "sample_end_date": returns["date"].max(),
        "rebalance_count": weights["decision_date"].nunique(),
        "latest_holdings_date": weights["decision_date"].max(),
    }


def validate_core_paths() -> None:
    """Keep the required artifact names anchored to the project root."""
    expected = {
        "fund_returns": PROJECT_ROOT / "results/data/fund_returns.csv",
        "fund_weights": PROJECT_ROOT / "results/data/fund_weights.csv",
        "performance_metrics": PROJECT_ROOT / "results/tables/performance_metrics.csv",
        "sector_sentiment_index": PROJECT_ROOT
        / "results/data/sector_sentiment_index.csv",
    }
    actual = {**FUND_ARTIFACT_PATHS, "sector_sentiment_index": SENTIMENT_ARTIFACT_PATH}
    for name, expected_path in expected.items():
        if actual[name].resolve() != expected_path.resolve():
            raise CoreBuildValidationError(f"incorrect project-root path for {name}")


def validate_core_build(build: CoreFundBuild) -> dict[str, object]:
    """Validate required artifacts plus the source panels and fusion evidence."""
    summary = validate_core_artifacts(build.artifacts, build.sentiment.sector_index)
    for name, panel in (
        ("combined_asset_returns", build.base.combined_asset_returns),
        ("equity_asset_returns", build.base.equity_asset_returns),
        ("crypto_asset_returns", build.crypto.crypto_asset_returns),
    ):
        if panel["date"].duplicated().any():
            raise CoreBuildValidationError(f"{name} has duplicate dates")
        values = panel.drop(columns="date").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise CoreBuildValidationError(f"{name} contains missing returns")
    locked_comparison_funds = set(build.fusion_evidence.comparison["fund"])
    if locked_comparison_funds != {fusion.BASE_FUND, fusion.TILTED_FUND}:
        raise CoreBuildValidationError("locked fusion comparison has wrong funds")
    exploratory_comparison_funds = set(build.exploratory_comparison["fund"])
    if exploratory_comparison_funds != {
        fusion.BASE_FUND,
        fusion.EXPLORATORY_FUND,
    }:
        raise CoreBuildValidationError("exploratory fusion comparison has wrong funds")
    if build.fusion_evidence.sector_exposure.duplicated(
        ["decision_date", "sector"]
    ).any():
        raise CoreBuildValidationError("locked sector exposure has duplicate keys")
    if build.predictability.duplicated(
        ["decision_date", "sector", "signal_method"]
    ).any():
        raise CoreBuildValidationError("predictability evidence has duplicate keys")
    if not (
        pd.to_datetime(build.predictability["first_holding_date"])
        > pd.to_datetime(build.predictability["decision_date"])
    ).all():
        raise CoreBuildValidationError("predictability returns do not follow decisions")
    active_weights = build.artifacts.fund_weights.loc[
        build.artifacts.fund_weights["fund"].eq(active_sector.FUND)
    ].copy()
    if active_weights.empty:
        raise CoreBuildValidationError("active-sector fund weights are absent")
    crypto = active_weights["ticker"].str.endswith("-USD")
    crypto_sleeves = active_weights.loc[crypto].groupby("decision_date")[
        "target_weight"
    ].sum()
    if not np.allclose(
        crypto_sleeves.to_numpy(),
        active_sector.CRYPTO_SLEEVE_WEIGHT,
        atol=WEIGHT_SUM_TOLERANCE,
        rtol=0.0,
    ):
        raise CoreBuildValidationError("active-sector crypto sleeve is not fixed at 10%")
    equity_sector_weights = active_weights.loc[~crypto].groupby(
        ["decision_date", "sector"]
    )["target_weight"].sum()
    if (equity_sector_weights > active_sector.SECTOR_WEIGHT_CAP + WEIGHT_SUM_TOLERANCE).any():
        raise CoreBuildValidationError("active-sector fund exceeds its sector cap")
    selected_stock_counts = active_weights.loc[
        ~crypto & active_weights["active_selected_sector"].fillna(False)
    ].groupby("decision_date").size()
    if not selected_stock_counts.eq(5 * active_sector.TOP_SECTOR_COUNT).all():
        raise CoreBuildValidationError("active-sector fund does not select two sectors")
    expected_exposure = active_sector.active_sector_exposure(active_weights)
    if not expected_exposure.equals(build.active_sector_exposure):
        raise CoreBuildValidationError("active-sector exposure does not match logged weights")
    growth_weights = build.artifacts.fund_weights.loc[
        build.artifacts.fund_weights["fund"].eq(active_sector.GROWTH_FUND)
    ].copy()
    growth_crypto = growth_weights["ticker"].str.endswith("-USD")
    growth_crypto_sleeve = growth_weights.loc[growth_crypto].groupby(
        "decision_date"
    )["target_weight"].sum()
    if not np.allclose(
        growth_crypto_sleeve.to_numpy(),
        active_sector.GROWTH_SECTOR_SPEC.crypto_sleeve_weight,
        atol=WEIGHT_SUM_TOLERANCE,
        rtol=0.0,
    ):
        raise CoreBuildValidationError("growth-sector crypto sleeve is not fixed at 5%")
    selected_growth_stocks = growth_weights.loc[
        ~growth_crypto & growth_weights["active_selected_sector"].fillna(False)
    ].groupby("decision_date").size()
    if not selected_growth_stocks.eq(
        5 * active_sector.GROWTH_SECTOR_SPEC.top_sector_count
    ).all():
        raise CoreBuildValidationError("growth-sector fund does not select three sectors")
    expected_growth_exposure = active_sector.sector_allocation_exposure(
        growth_weights,
        fund=active_sector.GROWTH_FUND,
    )
    if not expected_growth_exposure.equals(build.growth_sector_exposure):
        raise CoreBuildValidationError("growth-sector exposure does not match logged weights")
    return summary


def validate_written_core_artifacts() -> dict[str, object]:
    """Reload and validate the four required CSVs after serialization."""
    for path in [*FUND_ARTIFACT_PATHS.values(), SENTIMENT_ARTIFACT_PATH]:
        if not path.is_file():
            raise CoreBuildValidationError(f"required artifact was not written: {path}")
    returns = pd.read_csv(
        FUND_ARTIFACT_PATHS["fund_returns"],
        parse_dates=["date", "decision_date"],
    )
    weight_header = pd.read_csv(
        FUND_ARTIFACT_PATHS["fund_weights"], nrows=0
    ).columns
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
    weights = pd.read_csv(
        FUND_ARTIFACT_PATHS["fund_weights"],
        parse_dates=[
            column for column in weight_date_columns if column in weight_header
        ],
    )
    performance = pd.read_csv(
        FUND_ARTIFACT_PATHS["performance_metrics"],
        parse_dates=[
            "as_of_date",
            "sample_start_date",
            "sample_end_date",
            "current_holdings_date",
        ],
    )
    sector_index = pd.read_csv(
        SENTIMENT_ARTIFACT_PATH,
        parse_dates=["date", "tradable_signal_source_date"],
    )
    return validate_core_artifacts(
        metrics.FundArtifacts(returns, weights, performance),
        sector_index,
    )


def build_base_funds(
    *,
    initial_window: int = 252,
    solver: Callable[..., Any] = minimize,
) -> BaseFundBuild:
    """Build the three locked non-sentiment funds on one OOS calendar."""
    equities = etl.load_clean_equities()
    crypto = etl.load_clean_crypto()
    equity_returns = features.daily_returns(equities, asset_class="equity")
    crypto_returns = features.daily_returns(crypto, asset_class="crypto")
    combined_raw = features.combined_returns_panel(equity_returns, crypto_returns)
    missingness = features.return_missingness_audit(combined_raw)
    combined = features.complete_return_panel(combined_raw)

    equity_tickers = sorted(equities["ticker"].unique())
    equity_panel = combined[["date", *equity_tickers]].copy()
    equity_sector_map = (
        equities[["ticker", "sector"]]
        .drop_duplicates()
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    results = [
        portfolios.oos_backtest(
            combined,
            fund="combined_equal_weight",
            method="equal_weight",
            asset_family="combined",
            initial_window=initial_window,
        ),
        portfolios.oos_backtest(
            combined,
            fund="combined_min_variance",
            method="min_variance",
            asset_family="combined",
            initial_window=initial_window,
            solver=solver,
        ),
        portfolios.oos_backtest(
            equity_panel,
            fund="equity_equal_weight",
            method="equal_weight",
            asset_family="equity",
            initial_window=initial_window,
        ),
    ]
    combined_results = portfolios.concatenate_backtests(results)
    variation = portfolios.validate_weight_variation(
        combined_results.fund_weights,
        dynamic_fund="combined_min_variance",
        benchmark_fund="combined_equal_weight",
    )
    date_sets = {
        fund: tuple(group["date"])
        for fund, group in combined_results.fund_returns.groupby("fund", sort=True)
    }
    if len(set(date_sets.values())) != 1:
        raise portfolios.PortfolioValidationError(
            "base funds do not share the same OOS dates"
        )
    artifacts = metrics.build_fund_artifacts(
        combined_results.fund_returns,
        combined_results.fund_weights,
    )
    return BaseFundBuild(
        backtests=combined_results,
        combined_asset_returns=combined,
        equity_asset_returns=equity_panel,
        equity_sector_map=equity_sector_map,
        return_missingness_audit=missingness,
        variation_summary=variation,
        artifacts=artifacts,
    )


def build_crypto_funds(
    *,
    initial_window: int = 252,
    solver: Callable[..., Any] = minimize,
) -> CryptoFundBuild:
    """Build crypto-only funds from native seven-day return observations."""
    crypto = etl.load_clean_crypto()
    crypto_returns = features.daily_returns(crypto, asset_class="crypto")
    crypto_panel = crypto_returns.pivot(
        index="date", columns="ticker", values="daily_return"
    ).sort_index().reset_index()
    crypto_panel.columns.name = None
    crypto_panel = features.complete_return_panel(crypto_panel)
    results = portfolios.concatenate_backtests(
        [
            portfolios.oos_backtest(
                crypto_panel,
                fund="crypto_equal_weight",
                method="equal_weight",
                asset_family="crypto",
                initial_window=initial_window,
            ),
            portfolios.oos_backtest(
                crypto_panel,
                fund="crypto_min_variance",
                method="min_variance",
                asset_family="crypto",
                initial_window=initial_window,
                solver=solver,
            ),
        ]
    )
    variation = portfolios.validate_weight_variation(
        results.fund_weights,
        dynamic_fund="crypto_min_variance",
        benchmark_fund="crypto_equal_weight",
    )
    # This is a published audit field (rather than an implicit build default).
    results.fund_weights["transaction_cost_bps"] = 0.0
    return CryptoFundBuild(
        backtests=results,
        crypto_asset_returns=crypto_panel,
        variation_summary=variation,
    )


def build_core_sentiment() -> sentiment.SentimentBuild:
    """Build the locked ten-sector VADER index without writing artifacts."""
    equities = etl.load_clean_equities()
    headlines = etl.load_clean_news()
    analyzer = sentiment.get_vader_analyzer()
    return sentiment.build_sentiment_index(
        headlines,
        equities,
        analyzer=analyzer,
        min_history=60,
        zscore_clip=2.0,
        signal_lag=1,
        expected_sector_count=10,
    )


def build_core_funds(
    *,
    initial_window: int = 252,
    solver: Callable[..., Any] = minimize,
) -> CoreFundBuild:
    """Build the locked funds plus separately labelled exploratory extensions."""
    base = build_base_funds(initial_window=initial_window, solver=solver)
    crypto_build = build_crypto_funds(initial_window=initial_window, solver=solver)
    sentiment_build = build_core_sentiment()
    fusion_backtest = fusion.apply_sentiment_tilt(
        base.backtests,
        base.equity_asset_returns,
        sentiment_build.sector_index,
        base.equity_sector_map,
    )
    locked_backtests = portfolios.concatenate_backtests(
        [base.backtests, fusion_backtest]
    )
    locked_artifacts = metrics.build_fund_artifacts(
        locked_backtests.fund_returns,
        locked_backtests.fund_weights,
    )
    evidence = fusion.build_fusion_evidence(
        locked_artifacts.fund_returns,
        locked_artifacts.fund_weights,
        base.equity_sector_map,
    )

    exploratory_signal = sentiment.build_coverage_adjusted_trailing_signal(
        sentiment_build.sector_index
    )
    exploratory_backtest = fusion.apply_exploratory_coverage_tilt(
        base.backtests,
        base.equity_asset_returns,
        exploratory_signal,
        base.equity_sector_map,
    )
    active_sector_backtest = active_sector.build_active_sector_allocation(
        base.combined_asset_returns,
        sentiment_build.sector_index,
        base.equity_sector_map,
        initial_window=initial_window,
    )
    growth_sector_backtest = active_sector.build_growth_sector_allocation(
        base.combined_asset_returns,
        sentiment_build.sector_index,
        base.equity_sector_map,
        initial_window=initial_window,
    )
    aggressive_sector_backtest = active_sector.build_aggressive_sector_allocation(
        base.combined_asset_returns,
        sentiment_build.sector_index,
        base.equity_sector_map,
        initial_window=initial_window,
    )
    non_crypto_backtests = portfolios.concatenate_backtests(
        [
            locked_backtests,
            exploratory_backtest,
            active_sector_backtest,
            growth_sector_backtest,
            aggressive_sector_backtest,
        ]
    )
    non_crypto_artifacts = metrics.build_fund_artifacts(
        non_crypto_backtests.fund_returns,
        non_crypto_backtests.fund_weights,
    )
    backtests = portfolios.concatenate_backtests(
        [non_crypto_backtests, crypto_build.backtests]
    )
    artifacts = metrics.build_fund_artifacts(
        backtests.fund_returns,
        backtests.fund_weights,
        periods_per_year_by_family={"crypto": metrics.CRYPTO_PERIODS_PER_YEAR},
    )
    for before, after, name in (
        (non_crypto_artifacts.fund_returns, artifacts.fund_returns, "fund_returns"),
        (non_crypto_artifacts.fund_weights, artifacts.fund_weights, "fund_weights"),
        (
            non_crypto_artifacts.performance_metrics,
            artifacts.performance_metrics,
            "performance_metrics",
        ),
    ):
        actual = after.loc[~after["asset_family"].eq("crypto")]
        if before.to_csv(index=False, date_format="%Y-%m-%d") != actual.to_csv(
            index=False, date_format="%Y-%m-%d"
        ):
            raise CoreBuildValidationError(
                f"crypto build changed existing {name} bytes"
            )
    existing_funds = set(locked_artifacts.fund_returns["fund"].unique())
    locked_return_subset = artifacts.fund_returns.loc[
        artifacts.fund_returns["fund"].isin(existing_funds),
        locked_artifacts.fund_returns.columns,
    ].reset_index(drop=True)
    locked_weight_subset = artifacts.fund_weights.loc[
        artifacts.fund_weights["fund"].isin(existing_funds),
        locked_artifacts.fund_weights.columns,
    ].reset_index(drop=True)
    locked_metric_subset = artifacts.performance_metrics.loc[
        artifacts.performance_metrics["fund"].isin(existing_funds),
        locked_artifacts.performance_metrics.columns,
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        locked_return_subset,
        locked_artifacts.fund_returns.reset_index(drop=True),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        locked_weight_subset,
        locked_artifacts.fund_weights.reset_index(drop=True),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        locked_metric_subset,
        locked_artifacts.performance_metrics.reset_index(drop=True),
        check_exact=True,
    )

    exploratory_evidence = fusion.build_fusion_evidence(
        artifacts.fund_returns,
        artifacts.fund_weights,
        base.equity_sector_map,
        base_fund=fusion.BASE_FUND,
        tilted_fund=fusion.EXPLORATORY_FUND,
    )
    exploratory_comparison = exploratory_evidence.comparison.copy()
    exposure_absolute = exploratory_evidence.sector_exposure[
        "sector_weight_difference"
    ].abs()
    enhanced = exploratory_comparison["role"].eq("enhanced")
    exploratory_comparison["mean_absolute_sector_exposure_difference"] = 0.0
    exploratory_comparison["maximum_absolute_sector_exposure_difference"] = 0.0
    exploratory_comparison.loc[
        enhanced, "mean_absolute_sector_exposure_difference"
    ] = float(exposure_absolute.mean())
    exploratory_comparison.loc[
        enhanced, "maximum_absolute_sector_exposure_difference"
    ] = float(exposure_absolute.max())
    exploratory_comparison["exploratory_method_label"] = (
        fusion.EXPLORATORY_METHOD_LABEL
    )
    exploratory_comparison["signal_window_trading_days"] = (
        fusion.EXPLORATORY_SIGNAL_WINDOW
    )
    exploratory_comparison["tilt_strength"] = fusion.TILT_STRENGTH
    exploratory_comparison["zscore_clip_lower"] = -fusion.ZSCORE_CLIP
    exploratory_comparison["zscore_clip_upper"] = fusion.ZSCORE_CLIP
    exploratory_comparison["minimum_expanding_history"] = fusion.MIN_HISTORY
    exploratory_comparison["final_period_used_for_parameter_selection"] = False

    predictability = fusion.build_sentiment_predictability(
        base.backtests,
        base.equity_asset_returns,
        base.equity_sector_map,
        sentiment_build.sector_index,
        exploratory_signal,
    )
    predictability_summary = fusion.build_predictability_summary(predictability)
    active_exposure = active_sector.active_sector_exposure(
        active_sector_backtest.fund_weights
    )
    growth_exposure = active_sector.sector_allocation_exposure(
        growth_sector_backtest.fund_weights,
        fund=active_sector.GROWTH_FUND,
    )
    aggressive_exposure = active_sector.sector_allocation_exposure(
        aggressive_sector_backtest.fund_weights,
        fund=active_sector.AGGRESSIVE_FUND,
    )
    result = CoreFundBuild(
        base=base,
        crypto=crypto_build,
        sentiment=sentiment_build,
        fusion_backtest=fusion_backtest,
        exploratory_signal=exploratory_signal,
        exploratory_backtest=exploratory_backtest,
        active_sector_backtest=active_sector_backtest,
        growth_sector_backtest=growth_sector_backtest,
        aggressive_sector_backtest=aggressive_sector_backtest,
        backtests=backtests,
        locked_artifacts=locked_artifacts,
        non_crypto_artifacts=non_crypto_artifacts,
        artifacts=artifacts,
        fusion_evidence=evidence,
        exploratory_evidence=exploratory_evidence,
        exploratory_comparison=exploratory_comparison,
        predictability=predictability,
        predictability_summary=predictability_summary,
        active_sector_exposure=active_exposure,
        growth_sector_exposure=growth_exposure,
        aggressive_sector_exposure=aggressive_exposure,
    )
    validate_core_build(result)
    return result


def write_fund_artifacts(
    build: BaseFundBuild | CoreFundBuild,
    *,
    output_paths: dict[str, pathlib.Path] | None = None,
) -> dict[str, pathlib.Path]:
    """Write already-validated fund artifacts to their exact output paths."""
    paths = FUND_ARTIFACT_PATHS if output_paths is None else output_paths
    required = set(FUND_ARTIFACT_PATHS)
    if set(paths) != required:
        raise ValueError(f"output_paths must have exactly these keys: {sorted(required)}")
    frames = {
        "fund_returns": build.artifacts.fund_returns,
        "fund_weights": build.artifacts.fund_weights,
        "performance_metrics": build.artifacts.performance_metrics,
    }
    for name, destination in paths.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        frames[name].to_csv(destination, index=False, date_format="%Y-%m-%d")
    return paths


def write_sentiment_artifact(
    build: sentiment.SentimentBuild,
    *,
    output_path: pathlib.Path = SENTIMENT_ARTIFACT_PATH,
) -> pathlib.Path:
    """Write the precomputed index consumed by the deployed app."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build.sector_index.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return output_path


def write_investable_asset_returns(
    build: BaseFundBuild | CoreFundBuild,
    *,
    output_path: pathlib.Path = INVESTABLE_ASSET_RETURNS_PATH,
) -> pathlib.Path:
    """Write the validated individual-security return panel used by the app."""
    base = build.base if isinstance(build, CoreFundBuild) else build
    artifact = custom_portfolio.build_investable_asset_returns(
        base.combined_asset_returns,
        base.equity_sector_map,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    custom_portfolio.validate_investable_asset_returns(pd.read_csv(output_path))
    return output_path


def write_fusion_artifacts(
    build: CoreFundBuild,
    *,
    comparison_path: pathlib.Path = FUSION_COMPARISON_PATH,
    exposure_path: pathlib.Path = FUSION_SECTOR_EXPOSURE_PATH,
) -> dict[str, pathlib.Path]:
    """Write the matched performance/turnover and sector-exposure evidence."""
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    exposure_path.parent.mkdir(parents=True, exist_ok=True)
    build.fusion_evidence.comparison.to_csv(
        comparison_path,
        index=False,
        date_format="%Y-%m-%d",
    )
    build.fusion_evidence.sector_exposure.to_csv(
        exposure_path,
        index=False,
        date_format="%Y-%m-%d",
    )
    return {
        "fusion_comparison": comparison_path,
        "fusion_sector_exposure": exposure_path,
    }


def write_exploratory_artifacts(
    build: CoreFundBuild,
    *,
    signal_path: pathlib.Path = EXPLORATORY_SIGNAL_PATH,
    comparison_path: pathlib.Path = EXPLORATORY_COMPARISON_PATH,
    predictability_path: pathlib.Path = PREDICTABILITY_PATH,
    summary_path: pathlib.Path = PREDICTABILITY_SUMMARY_PATH,
) -> dict[str, pathlib.Path]:
    """Write clearly labelled robustness and non-causal diagnostic evidence."""
    paths = {
        "sector_sentiment_21d_coverage": signal_path,
        "fusion_exploratory_comparison": comparison_path,
        "sentiment_predictability": predictability_path,
        "sentiment_predictability_summary": summary_path,
    }
    frames = {
        "sector_sentiment_21d_coverage": build.exploratory_signal,
        "fusion_exploratory_comparison": build.exploratory_comparison,
        "sentiment_predictability": build.predictability,
        "sentiment_predictability_summary": build.predictability_summary,
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[name].to_csv(path, index=False, date_format="%Y-%m-%d")
    return paths


def write_active_sector_artifacts(
    build: CoreFundBuild,
    *,
    output_path: pathlib.Path = ACTIVE_SECTOR_EXPOSURE_PATH,
) -> pathlib.Path:
    """Write the active fund's sector-and-crypto exposure audit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build.active_sector_exposure.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return output_path


def write_growth_sector_artifacts(
    build: CoreFundBuild,
    *,
    output_path: pathlib.Path = GROWTH_SECTOR_EXPOSURE_PATH,
) -> pathlib.Path:
    """Write the balanced-growth fund's sector-and-crypto exposure audit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build.growth_sector_exposure.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return output_path


def write_aggressive_sector_artifacts(
    build: CoreFundBuild,
    *,
    output_path: pathlib.Path = AGGRESSIVE_SECTOR_EXPOSURE_PATH,
) -> pathlib.Path:
    """Write the aggressive fund's sector-and-crypto exposure audit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build.aggressive_sector_exposure.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )
    return output_path


def main() -> None:
    started = time.perf_counter()
    np.random.seed(BUILD_RANDOM_SEED)
    validate_core_paths()
    build = build_core_funds()
    sentiment_build = build.sentiment
    returns = build.backtests.fund_returns
    weights = build.backtests.fund_weights
    paths = write_fund_artifacts(build)
    sentiment_path = write_sentiment_artifact(sentiment_build)
    asset_returns_path = write_investable_asset_returns(build)
    fusion_paths = write_fusion_artifacts(build)
    exploratory_paths = write_exploratory_artifacts(build)
    active_exposure_path = write_active_sector_artifacts(build)
    growth_exposure_path = write_growth_sector_artifacts(build)
    aggressive_exposure_path = write_aggressive_sector_artifacts(build)
    written_summary = validate_written_core_artifacts()
    print(
        "core funds:",
        sorted(returns["fund"].unique()),
        f"dates={returns['date'].nunique():,}",
        f"rebalances={weights['decision_date'].nunique():,}",
    )
    print("weight variation:", build.base.variation_summary)
    print("crypto weight variation:", build.crypto.variation_summary)
    for name, path in paths.items():
        print(f"wrote {name}: {path.relative_to(PROJECT_ROOT)}")
    print(
        "sentiment:",
        f"headlines={len(sentiment_build.headline_scores):,}",
        f"distinct_titles={len(sentiment_build.title_score_cache):,}",
        f"ticker_days={len(sentiment_build.ticker_day_scores):,}",
        f"sector_rows={len(sentiment_build.sector_index):,}",
    )
    print(f"wrote sector_sentiment_index: {sentiment_path.relative_to(PROJECT_ROOT)}")
    print(
        "wrote investable_asset_returns: "
        f"{asset_returns_path.relative_to(PROJECT_ROOT)}"
    )
    for name, path in fusion_paths.items():
        print(f"wrote {name}: {path.relative_to(PROJECT_ROOT)}")
    for name, path in exploratory_paths.items():
        print(f"wrote {name}: {path.relative_to(PROJECT_ROOT)}")
    print(
        "wrote active_sector_allocation_exposure: "
        f"{active_exposure_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "wrote growth_sector_allocation_exposure: "
        f"{growth_exposure_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "wrote aggressive_sector_allocation_exposure: "
        f"{aggressive_exposure_path.relative_to(PROJECT_ROOT)}"
    )
    print("fusion comparison:")
    print(build.fusion_evidence.comparison.to_string(index=False))
    print("exploratory fusion comparison:")
    print(build.exploratory_comparison.to_string(index=False))
    print("predictability summary (overall folds):")
    print(
        build.predictability_summary.loc[
            build.predictability_summary["summary_scope"].eq("overall")
        ].to_string(index=False)
    )
    print(
        "core artifact validation:",
        f"returns={written_summary['fund_return_rows']:,}",
        f"weights={written_summary['fund_weight_rows']:,}",
        f"metrics={written_summary['metric_rows']:,}",
        f"sentiment={written_summary['sector_sentiment_rows']:,}",
        f"sample={written_summary['sample_start_date'].date()}"
        f"..{written_summary['sample_end_date'].date()}",
        f"latest_holdings={written_summary['latest_holdings_date'].date()}",
    )
    print(
        "build completed:",
        f"seed={BUILD_RANDOM_SEED}",
        f"elapsed_seconds={time.perf_counter() - started:.3f}",
    )


if __name__ == "__main__":
    main()
