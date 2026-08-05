"""Validated, precomputed artifact access for the PortFoYou app.

This module deliberately contains no build, hosted-data, or sentiment-scoring
imports.  It gives the Streamlit layer small deterministic functions for loading
the four published CSV contracts and for calculating allocation scenarios from
the already backtested daily fund returns.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT_RELATIVE_PATHS = {
    "fund_returns": Path("results/data/fund_returns.csv"),
    "fund_weights": Path("results/data/fund_weights.csv"),
    "sector_sentiment": Path("results/data/sector_sentiment_index.csv"),
    "performance_metrics": Path("results/tables/performance_metrics.csv"),
}

FINBERT_ARTIFACT_RELATIVE_PATHS = {
    "sector_sentiment": Path("results/data/sector_sentiment_finbert.csv"),
    "model_comparison": Path("results/tables/sentiment_model_comparison.csv"),
    "disagreements": Path("results/tables/sentiment_model_disagreements.csv"),
    "metadata": Path("results/tables/finbert_run_metadata.json"),
}

MANUAL_VALIDATION_RELATIVE_PATHS = {
    "metrics": Path("results/tables/sentiment_manual_validation_metrics.csv"),
    "metadata": Path("results/tables/sentiment_manual_validation_metadata.json"),
}

EXPECTED_FUND_IDENTITIES = {
    "combined_equal_weight": ("combined", "equal_weight"),
    "combined_min_variance": ("combined", "min_variance"),
    "combined_active_sector_allocation": (
        "combined",
        "active_sector_sentiment_risk_budget",
    ),
    "combined_growth_sector_allocation": (
        "combined",
        "growth_sector_sentiment_risk_budget",
    ),
    "combined_aggressive_sector_allocation": (
        "combined",
        "aggressive_sector_sentiment_risk_budget",
    ),
    "equity_equal_weight": ("equity", "equal_weight"),
    "equity_sentiment_tilt": ("equity", "sentiment_tilt"),
    "equity_sentiment_21d_coverage_tilt": (
        "equity",
        "exploratory_sentiment_21d_coverage_tilt",
    ),
    "crypto_equal_weight": ("crypto", "equal_weight"),
    "crypto_min_variance": ("crypto", "min_variance"),
}

FUND_LABELS = {
    "combined_equal_weight": "Multi-Asset Equal Weight",
    "combined_min_variance": "Multi-Asset Minimum Variance",
    "combined_active_sector_allocation": "Multi-Asset Active Sector Allocation",
    "combined_growth_sector_allocation": "Balanced Growth Sector Allocation",
    "combined_aggressive_sector_allocation": "Aggressive Sector & Crypto Allocation",
    "equity_equal_weight": "US Equity Equal Weight",
    "equity_sentiment_tilt": "US Equity Sentiment Strategy",
    "equity_sentiment_21d_coverage_tilt": (
        "US Equity 21-Day Sentiment (Exploratory)"
    ),
    "crypto_equal_weight": "Crypto Equal Weight",
    "crypto_min_variance": "Crypto Minimum Variance",
}

EXPECTED_SECTORS = (
    "Comm",
    "Consumer",
    "Energy",
    "Financials",
    "Healthcare",
    "Industrials",
    "Materials",
    "RealEstate",
    "Tech",
    "Utilities",
)

FUND_RETURN_COLUMNS = {
    "fund",
    "asset_family",
    "method",
    "date",
    "decision_date",
    "daily_return",
    "growth_of_1",
}
FUND_WEIGHT_COLUMNS = {
    "fund",
    "asset_family",
    "method",
    "decision_date",
    "ticker",
    "target_weight",
    "is_current",
}
PERFORMANCE_COLUMNS = {
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
    "sharpe_ratio",
    "maximum_drawdown",
}
SENTIMENT_COLUMNS = {
    "date",
    "sector",
    "raw_sector_compound",
    "observed_ticker_count",
    "headline_count",
    "possible_ticker_count",
    "ticker_coverage_share",
    "has_observed_news",
    "tradable_sector_zscore",
    "tradable_signal_source_date",
    "signal_lag_trading_days",
}
FINBERT_SENTIMENT_COLUMNS = {
    "date",
    "sector",
    "raw_sector_finbert",
    "sector_probability_positive",
    "sector_probability_negative",
    "sector_probability_neutral",
    "observed_ticker_count",
    "headline_count",
    "possible_ticker_count",
    "ticker_coverage_share",
    "has_observed_news",
    "sentiment_model",
    "score_name",
}
FINBERT_COMPARISON_COLUMNS = {
    "observation_unit",
    "sector",
    "paired_observation_count",
    "vader_mean",
    "finbert_mean",
    "pearson_correlation",
    "spearman_correlation",
    "descriptive_label_agreement_rate",
    "opposite_sign_rate",
    "vader_neutral_rate",
    "finbert_neutral_rate",
    "vader_label_rule",
    "finbert_label_rule",
    "correlation_basis",
    "interpretation_status",
}
FINBERT_DISAGREEMENT_COLUMNS = {
    "date",
    "ticker",
    "sector",
    "text_raw",
    "vader_compound",
    "vader_label",
    "finbert_score",
    "finbert_label",
    "sampling_stratum",
    "year",
    "sample_purpose",
    "validation_status",
}
MANUAL_VALIDATION_METRIC_COLUMNS = {
    "model",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "ci_method",
    "sample_purpose",
    "sample_count",
    "sampling_weight_sum",
    "effective_sample_size",
}

WEIGHT_TOLERANCE = 1e-8
GROWTH_TOLERANCE = 1e-10


class AppArtifactError(RuntimeError):
    """Raised when a published app artifact is missing or inconsistent."""


@dataclass(frozen=True)
class AppArtifacts:
    """The four validated app-facing dataframes."""

    fund_returns: pd.DataFrame
    fund_weights: pd.DataFrame
    sector_sentiment: pd.DataFrame
    performance_metrics: pd.DataFrame


@dataclass(frozen=True)
class FinBERTAppArtifacts:
    """Optional validated neural-sentiment artifacts used only for display."""

    sector_sentiment: pd.DataFrame
    model_comparison: pd.DataFrame
    disagreements: pd.DataFrame
    metadata: dict[str, object]
    manual_validation: ManualValidationAppArtifacts | None = None
    manual_validation_error: str | None = None


@dataclass(frozen=True)
class ManualValidationAppArtifacts:
    """Optional student-labelled evidence shown after Phase 4 passes."""

    metrics: pd.DataFrame
    metadata: dict[str, object]


@dataclass(frozen=True)
class AllocationScenario:
    """A historical allocation path and fund-level contribution table."""

    summary: pd.DataFrame
    history: pd.DataFrame
    initial_value: float
    ending_value: float


@dataclass(frozen=True)
class FeeAdjustedAllocationScenario:
    """A hypothetical fee overlay on a user-selected allocation scenario only."""

    history: pd.DataFrame
    annual_management_fee: float
    gross_ending_value: float
    fee_adjusted_ending_value: float

    @property
    def estimated_fee_drag(self) -> float:
        """Return the scenario-only difference between gross and fee-adjusted wealth."""
        return self.gross_ending_value - self.fee_adjusted_ending_value


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AppArtifactError(f"{name} is missing required columns: {missing}")


def _normalise_dates(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    for column in columns:
        if column not in frame:
            continue
        try:
            parsed = pd.to_datetime(frame[column], errors="raise", utc=True)
        except (TypeError, ValueError) as exc:
            raise AppArtifactError(f"{name}.{column} contains an invalid date: {exc}") from exc
        if parsed.isna().any():
            raise AppArtifactError(f"{name}.{column} contains missing dates")
        frame[column] = parsed.dt.tz_convert(None).dt.normalize()


def _parse_boolean(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalised = series.astype("string").str.strip().str.lower()
    mapped = normalised.map({"true": True, "false": False})
    if mapped.isna().any():
        bad = sorted(normalised.loc[mapped.isna()].dropna().unique().tolist())
        raise AppArtifactError(f"{label} contains invalid Boolean values: {bad}")
    return mapped.astype(bool)


def _validate_identifiers(frame: pd.DataFrame, name: str) -> None:
    identifiers = ["fund", "asset_family", "method"]
    if frame[identifiers].isna().any().any():
        raise AppArtifactError(f"{name} contains missing fund identifiers")
    if (frame[identifiers].astype(str).apply(lambda column: column.str.strip()) == "").any().any():
        raise AppArtifactError(f"{name} contains blank fund identifiers")
    counts = frame.groupby("fund", sort=False)[["asset_family", "method"]].nunique()
    if (counts != 1).any().any():
        raise AppArtifactError(f"{name} contains inconsistent fund mappings")


def _identity_map(frame: pd.DataFrame) -> dict[str, tuple[str, str]]:
    identity = frame[["fund", "asset_family", "method"]].drop_duplicates("fund")
    return {
        str(row.fund): (str(row.asset_family), str(row.method))
        for row in identity.itertuples(index=False)
    }


def _validate_sorted(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    expected = frame.sort_values(columns, kind="mergesort").reset_index(drop=True)
    if not frame[columns].reset_index(drop=True).equals(expected[columns]):
        raise AppArtifactError(f"{name} is not stably sorted by {columns}")


def validate_app_artifacts(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    sector_sentiment: pd.DataFrame,
    performance_metrics: pd.DataFrame,
) -> AppArtifacts:
    """Validate and type the four CSV contracts consumed by the app."""
    returns = fund_returns.copy()
    weights = fund_weights.copy()
    sentiment = sector_sentiment.copy()
    performance = performance_metrics.copy()

    _require_columns(returns, FUND_RETURN_COLUMNS, "fund_returns.csv")
    _require_columns(weights, FUND_WEIGHT_COLUMNS, "fund_weights.csv")
    _require_columns(sentiment, SENTIMENT_COLUMNS, "sector_sentiment_index.csv")
    _require_columns(performance, PERFORMANCE_COLUMNS, "performance_metrics.csv")

    _normalise_dates(returns, ("date", "decision_date"), "fund_returns.csv")
    _normalise_dates(
        weights,
        ("date", "decision_date", "first_holding_date"),
        "fund_weights.csv",
    )
    _normalise_dates(sentiment, ("date",), "sector_sentiment_index.csv")
    if "tradable_signal_source_date" in sentiment:
        source = sentiment["tradable_signal_source_date"]
        non_missing = source.notna()
        sentiment["tradable_signal_source_date"] = pd.NaT
        if non_missing.any():
            try:
                sentiment.loc[non_missing, "tradable_signal_source_date"] = (
                    pd.to_datetime(source.loc[non_missing], errors="raise", utc=True)
                    .dt.tz_convert(None)
                    .dt.normalize()
                )
            except (TypeError, ValueError) as exc:
                raise AppArtifactError(
                    "sector_sentiment_index.csv.tradable_signal_source_date "
                    f"contains an invalid date: {exc}"
                ) from exc
    _normalise_dates(
        performance,
        (
            "as_of_date",
            "sample_start_date",
            "sample_end_date",
            "current_holdings_date",
        ),
        "performance_metrics.csv",
    )

    for frame, name in (
        (returns, "fund_returns.csv"),
        (weights, "fund_weights.csv"),
        (performance, "performance_metrics.csv"),
    ):
        _validate_identifiers(frame, name)

    returns["daily_return"] = pd.to_numeric(returns["daily_return"], errors="coerce")
    returns["growth_of_1"] = pd.to_numeric(returns["growth_of_1"], errors="coerce")
    if returns.duplicated(["fund", "date"]).any():
        raise AppArtifactError("fund_returns.csv contains duplicate fund + date keys")
    if not np.isfinite(returns[["daily_return", "growth_of_1"]].to_numpy()).all():
        raise AppArtifactError("fund_returns.csv contains non-finite return values")
    if (returns["daily_return"] <= -1).any() or (returns["growth_of_1"] <= 0).any():
        raise AppArtifactError("fund_returns.csv contains invalid wealth values")
    if not (returns["decision_date"] < returns["date"]).all():
        raise AppArtifactError("fund_returns.csv has a decision on or after its return date")
    _validate_sorted(returns, ["fund", "date"], "fund_returns.csv")
    recalculated_growth = returns.groupby("fund", sort=False)["daily_return"].transform(
        lambda values: (1.0 + values).cumprod()
    )
    if not np.allclose(
        returns["growth_of_1"],
        recalculated_growth,
        atol=GROWTH_TOLERANCE,
        rtol=GROWTH_TOLERANCE,
    ):
        raise AppArtifactError("fund_returns.csv growth_of_1 does not reconcile")

    weights["target_weight"] = pd.to_numeric(weights["target_weight"], errors="coerce")
    weights["is_current"] = _parse_boolean(weights["is_current"], "fund_weights.csv.is_current")
    weight_key = ["fund", "decision_date", "ticker"]
    if weights["ticker"].isna().any() or (weights["ticker"].astype(str).str.strip() == "").any():
        raise AppArtifactError("fund_weights.csv contains missing ticker identifiers")
    if weights.duplicated(weight_key).any():
        raise AppArtifactError(
            "fund_weights.csv contains duplicate fund + decision_date + ticker keys"
        )
    if not np.isfinite(weights["target_weight"]).all():
        raise AppArtifactError("fund_weights.csv contains non-finite weights")
    if (weights["target_weight"] < -WEIGHT_TOLERANCE).any() or (
        weights["target_weight"] > 1 + WEIGHT_TOLERANCE
    ).any():
        raise AppArtifactError("fund_weights.csv violates long-only weight bounds")
    weight_sums = weights.groupby(["fund", "decision_date"])["target_weight"].sum()
    if not np.allclose(weight_sums, 1.0, atol=WEIGHT_TOLERANCE, rtol=0.0):
        raise AppArtifactError("fund_weights.csv target weights do not sum to one")
    _validate_sorted(weights, weight_key, "fund_weights.csv")
    latest_dates = weights.groupby("fund")["decision_date"].transform("max")
    if not weights["is_current"].equals(weights["decision_date"].eq(latest_dates)):
        raise AppArtifactError(
            "fund_weights.csv is_current rows do not select each latest rebalance"
        )

    performance_numeric = [
        "observations",
        "periods_per_year",
        "risk_free_rate_annual",
        "final_growth_of_1",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
    ]
    for column in performance_numeric:
        performance[column] = pd.to_numeric(performance[column], errors="coerce")
    if performance.duplicated("fund").any():
        raise AppArtifactError("performance_metrics.csv must contain one row per fund")
    if not np.isfinite(performance[performance_numeric].to_numpy()).all():
        raise AppArtifactError("performance_metrics.csv contains non-finite metrics")
    expected_periods = performance["asset_family"].map(
        {"crypto": 365}
    ).fillna(252)
    if not performance["periods_per_year"].eq(expected_periods).all():
        raise AppArtifactError(
            "performance_metrics.csv annualisation must be 365 for crypto and 252 otherwise"
        )
    if not np.allclose(performance["risk_free_rate_annual"], 0.0, atol=0.0, rtol=0.0):
        raise AppArtifactError("performance_metrics.csv must use a 0% risk-free rate")
    if not performance["annual_return_method"].eq("geometric").all():
        raise AppArtifactError("performance_metrics.csv annual returns must be geometric")
    _validate_sorted(performance, ["fund"], "performance_metrics.csv")

    sentiment["has_observed_news"] = _parse_boolean(
        sentiment["has_observed_news"],
        "sector_sentiment_index.csv.has_observed_news",
    )
    sentiment_numeric = [
        "raw_sector_compound",
        "observed_ticker_count",
        "headline_count",
        "possible_ticker_count",
        "ticker_coverage_share",
        "tradable_sector_zscore",
        "signal_lag_trading_days",
    ]
    for column in sentiment_numeric:
        sentiment[column] = pd.to_numeric(sentiment[column], errors="coerce")
    if sentiment[["date", "sector"]].isna().any().any():
        raise AppArtifactError("sector_sentiment_index.csv contains missing keys")
    if sentiment.duplicated(["date", "sector"]).any():
        raise AppArtifactError("sector_sentiment_index.csv contains duplicate date + sector keys")
    if set(sentiment["sector"].unique()) != set(EXPECTED_SECTORS):
        raise AppArtifactError(
            "sector_sentiment_index.csv does not contain the required ten sectors"
        )
    count_columns = [
        "observed_ticker_count",
        "headline_count",
        "possible_ticker_count",
        "ticker_coverage_share",
        "signal_lag_trading_days",
    ]
    if not np.isfinite(sentiment[count_columns].to_numpy()).all():
        raise AppArtifactError("sector_sentiment_index.csv has invalid coverage values")
    if (
        (sentiment["observed_ticker_count"] < 0).any()
        or (sentiment["headline_count"] < 0).any()
        or (sentiment["possible_ticker_count"] <= 0).any()
        or (sentiment["observed_ticker_count"] > sentiment["possible_ticker_count"]).any()
        or (sentiment["ticker_coverage_share"] < 0).any()
        or (sentiment["ticker_coverage_share"] > 1).any()
    ):
        raise AppArtifactError("sector_sentiment_index.csv coverage is out of bounds")
    expected_coverage = sentiment["observed_ticker_count"] / sentiment["possible_ticker_count"]
    if not np.allclose(
        sentiment["ticker_coverage_share"],
        expected_coverage,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AppArtifactError("sector sentiment coverage share does not reconcile")
    observed = sentiment["has_observed_news"]
    if sentiment.loc[observed, "raw_sector_compound"].isna().any():
        raise AppArtifactError("observed sector-news rows are missing raw sentiment")
    if sentiment.loc[~observed, "raw_sector_compound"].notna().any():
        raise AppArtifactError("no-news sector-days must remain missing, not neutral")
    _validate_sorted(sentiment, ["date", "sector"], "sector_sentiment_index.csv")

    identities = (
        _identity_map(returns),
        _identity_map(weights),
        _identity_map(performance),
    )
    if not identities[0] == identities[1] == identities[2]:
        raise AppArtifactError("fund identities differ across app artifacts")
    if identities[0] != EXPECTED_FUND_IDENTITIES:
        raise AppArtifactError(
            "app artifact fund identities differ from the published PortFoYou set"
        )

    return_decisions = returns[["fund", "decision_date"]].drop_duplicates()
    weight_decisions = weights[["fund", "decision_date"]].drop_duplicates()
    decision_join = return_decisions.merge(
        weight_decisions,
        on=["fund", "decision_date"],
        how="left",
        indicator=True,
    )
    if not decision_join["_merge"].eq("both").all():
        raise AppArtifactError("fund returns reference unavailable target weights")

    return_summary = returns.groupby("fund").agg(
        sample_start_date=("date", "min"),
        sample_end_date=("date", "max"),
        observations=("date", "size"),
        final_growth_of_1=("growth_of_1", "last"),
    )
    latest_weights = weights.groupby("fund")["decision_date"].max()
    metric_check = performance.set_index("fund").join(
        return_summary,
        rsuffix="_returns",
    )
    for column in ("sample_start_date", "sample_end_date", "observations"):
        if not metric_check[column].equals(metric_check[f"{column}_returns"]):
            raise AppArtifactError(
                f"performance_metrics.csv {column} does not reconcile to returns"
            )
    if not np.allclose(
        metric_check["final_growth_of_1"],
        metric_check["final_growth_of_1_returns"],
        atol=GROWTH_TOLERANCE,
        rtol=GROWTH_TOLERANCE,
    ):
        raise AppArtifactError("performance_metrics.csv final growth does not reconcile to returns")
    if not performance.set_index("fund")["current_holdings_date"].equals(latest_weights):
        raise AppArtifactError("performance current_holdings_date does not match latest weights")

    return AppArtifacts(returns, weights, sentiment, performance)


def load_app_artifacts(project_root: Path) -> AppArtifacts:
    """Read the four project-relative CSVs and validate them as one contract."""
    root = Path(project_root)
    frames: dict[str, pd.DataFrame] = {}
    for name, relative_path in ARTIFACT_RELATIVE_PATHS.items():
        path = root / relative_path
        if not path.is_file():
            raise AppArtifactError(
                f"Required PortFoYou artifact is missing: {relative_path.as_posix()}. "
                "Rebuild the project artifacts locally before opening the app."
            )
        try:
            frames[name] = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise AppArtifactError(f"Could not read {relative_path.as_posix()}: {exc}") from exc
    return validate_app_artifacts(
        frames["fund_returns"],
        frames["fund_weights"],
        frames["sector_sentiment"],
        frames["performance_metrics"],
    )


def _finbert_unavailable(message: str) -> AppArtifactError:
    return AppArtifactError(f"FinBERT robustness artifacts are unavailable: {message}")


def validate_finbert_app_artifacts(
    sector_sentiment: pd.DataFrame,
    model_comparison: pd.DataFrame,
    disagreements: pd.DataFrame,
    metadata: Mapping[str, object],
    vader_sector_sentiment: pd.DataFrame,
) -> FinBERTAppArtifacts:
    """Validate the optional precomputed FinBERT display contract."""
    finbert = sector_sentiment.copy()
    comparison = model_comparison.copy()
    examples = disagreements.copy()
    run_metadata = dict(metadata)

    try:
        _require_columns(
            finbert,
            FINBERT_SENTIMENT_COLUMNS,
            "sector_sentiment_finbert.csv",
        )
        _require_columns(
            comparison,
            FINBERT_COMPARISON_COLUMNS,
            "sentiment_model_comparison.csv",
        )
        _require_columns(
            examples,
            FINBERT_DISAGREEMENT_COLUMNS,
            "sentiment_model_disagreements.csv",
        )
        _normalise_dates(finbert, ("date",), "sector_sentiment_finbert.csv")
        _normalise_dates(examples, ("date",), "sentiment_model_disagreements.csv")
    except AppArtifactError as exc:
        raise _finbert_unavailable(str(exc)) from exc

    try:
        finbert["has_observed_news"] = _parse_boolean(
            finbert["has_observed_news"],
            "sector_sentiment_finbert.csv.has_observed_news",
        )
    except AppArtifactError as exc:
        raise _finbert_unavailable(str(exc)) from exc
    finbert_numeric = [
        "raw_sector_finbert",
        "sector_probability_positive",
        "sector_probability_negative",
        "sector_probability_neutral",
        "observed_ticker_count",
        "headline_count",
        "possible_ticker_count",
        "ticker_coverage_share",
    ]
    finbert[finbert_numeric] = finbert[finbert_numeric].apply(pd.to_numeric, errors="coerce")
    if finbert[["date", "sector"]].isna().any().any():
        raise _finbert_unavailable("the FinBERT sector index has missing keys")
    if finbert.duplicated(["date", "sector"]).any():
        raise _finbert_unavailable("the FinBERT sector index has duplicate keys")
    if set(finbert["sector"]) != set(EXPECTED_SECTORS):
        raise _finbert_unavailable("the FinBERT sector universe is inconsistent")
    observed = finbert["has_observed_news"]
    probabilities = finbert[
        [
            "sector_probability_positive",
            "sector_probability_negative",
            "sector_probability_neutral",
        ]
    ]
    if not np.isfinite(finbert.loc[observed, finbert_numeric].to_numpy()).all():
        raise _finbert_unavailable("observed FinBERT sector rows are non-finite")
    if finbert.loc[~observed, "raw_sector_finbert"].notna().any():
        raise _finbert_unavailable("no-news FinBERT sector-days are not missing")
    if not np.allclose(
        probabilities.loc[observed].sum(axis=1),
        1.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise _finbert_unavailable("FinBERT sector probabilities do not sum to one")
    expected_score = finbert["sector_probability_positive"] - finbert["sector_probability_negative"]
    if not np.allclose(
        finbert.loc[observed, "raw_sector_finbert"],
        expected_score.loc[observed],
        atol=1e-12,
        rtol=0.0,
    ):
        raise _finbert_unavailable("the FinBERT score definition is inconsistent")
    if not finbert["sentiment_model"].eq("ProsusAI/finbert").all():
        raise _finbert_unavailable("the FinBERT model name is inconsistent")
    if not finbert["score_name"].eq("P(positive) - P(negative)").all():
        raise _finbert_unavailable("the FinBERT score name is inconsistent")
    try:
        _validate_sorted(finbert, ["date", "sector"], "sector_sentiment_finbert.csv")
    except AppArtifactError as exc:
        raise _finbert_unavailable(str(exc)) from exc

    coverage_columns = [
        "date",
        "sector",
        "observed_ticker_count",
        "headline_count",
        "possible_ticker_count",
        "ticker_coverage_share",
        "has_observed_news",
    ]
    vader_coverage = vader_sector_sentiment[coverage_columns].reset_index(drop=True)
    finbert_coverage = finbert[coverage_columns].reset_index(drop=True)
    exact_columns = [
        "date",
        "sector",
        "observed_ticker_count",
        "headline_count",
        "possible_ticker_count",
        "has_observed_news",
    ]
    if len(vader_coverage) != len(finbert_coverage) or any(
        not vader_coverage[column].equals(finbert_coverage[column]) for column in exact_columns
    ):
        raise _finbert_unavailable("FinBERT and VADER sector coverage differs")
    if not np.allclose(
        vader_coverage["ticker_coverage_share"],
        finbert_coverage["ticker_coverage_share"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise _finbert_unavailable("FinBERT and VADER ticker coverage differs")

    expected_units = {"clean_headline_row", "matched_date_sector"}
    if set(comparison["observation_unit"]) != expected_units:
        raise _finbert_unavailable("comparison observation units are invalid")
    if comparison.duplicated(["observation_unit", "sector"]).any():
        raise _finbert_unavailable("comparison rows are not uniquely identified")
    expected_comparison_sectors = {"All", *EXPECTED_SECTORS}
    for unit in expected_units:
        sectors = set(comparison.loc[comparison["observation_unit"].eq(unit), "sector"])
        if sectors != expected_comparison_sectors:
            raise _finbert_unavailable(f"comparison sectors are incomplete for {unit}")
    comparison_numeric = [
        "paired_observation_count",
        "vader_mean",
        "finbert_mean",
        "pearson_correlation",
        "spearman_correlation",
        "descriptive_label_agreement_rate",
        "opposite_sign_rate",
        "vader_neutral_rate",
        "finbert_neutral_rate",
    ]
    comparison[comparison_numeric] = comparison[comparison_numeric].apply(
        pd.to_numeric, errors="coerce"
    )
    matched = comparison["observation_unit"].eq("matched_date_sector")
    headline = comparison["observation_unit"].eq("clean_headline_row")
    if comparison.loc[matched, comparison_numeric].isna().any().any():
        raise _finbert_unavailable("matched sector comparison values are missing")
    if (
        comparison.loc[headline, "pearson_correlation"].notna().any()
        or comparison.loc[headline, "spearman_correlation"].notna().any()
    ):
        raise _finbert_unavailable("headline-level correlations must not be reported")
    rate_columns = [
        "descriptive_label_agreement_rate",
        "opposite_sign_rate",
        "vader_neutral_rate",
        "finbert_neutral_rate",
    ]
    if not comparison[rate_columns].stack().between(0.0, 1.0).all():
        raise _finbert_unavailable("comparison rates are outside zero to one")

    examples_numeric = ["vader_compound", "finbert_score", "year"]
    examples[examples_numeric] = examples[examples_numeric].apply(pd.to_numeric, errors="coerce")
    if examples.empty or examples[sorted(FINBERT_DISAGREEMENT_COLUMNS)].isna().any().any():
        raise _finbert_unavailable("disagreement examples contain missing values")
    if examples.duplicated("text_raw").any():
        raise _finbert_unavailable("disagreement examples repeat a headline")
    if not set(examples["sector"]).issubset(EXPECTED_SECTORS):
        raise _finbert_unavailable("disagreement sectors are invalid")
    if not examples["sample_purpose"].eq("descriptive_disagreement_audit").all():
        raise _finbert_unavailable("disagreement examples have an invalid purpose")
    if not examples["validation_status"].eq("pending student review").all():
        raise _finbert_unavailable("disagreement validation status is inconsistent")
    allowed_labels = {"positive", "negative", "neutral"}
    if not set(examples["vader_label"]).issubset(allowed_labels) or not set(
        examples["finbert_label"]
    ).issubset(allowed_labels):
        raise _finbert_unavailable("disagreement labels are invalid")

    required_metadata = {
        "phase_status",
        "model_name",
        "pinned_revision",
        "score_definition",
        "matched_sector_day_count",
        "manual_review_status",
        "claims_status",
    }
    missing_metadata = sorted(required_metadata.difference(run_metadata))
    if missing_metadata:
        raise _finbert_unavailable(f"metadata keys are missing: {missing_metadata}")
    if run_metadata["phase_status"] != "PASS":
        raise _finbert_unavailable("Phase 2 metadata is not PASS")
    if run_metadata["model_name"] != "ProsusAI/finbert":
        raise _finbert_unavailable("metadata model name is inconsistent")
    if run_metadata["pinned_revision"] != "4556d13015211d73dccd3fdd39d39232506f3e43":
        raise _finbert_unavailable("metadata model revision is inconsistent")
    if run_metadata["score_definition"] != "probability_positive - probability_negative":
        raise _finbert_unavailable("metadata score definition is inconsistent")
    overall_matched = comparison.loc[matched & comparison["sector"].eq("All")].iloc[0]
    if int(run_metadata["matched_sector_day_count"]) != int(
        overall_matched["paired_observation_count"]
    ):
        raise _finbert_unavailable("metadata paired count differs from comparison")

    return FinBERTAppArtifacts(finbert, comparison, examples, run_metadata)


def validate_manual_validation_app_artifacts(
    metrics: pd.DataFrame,
    metadata: Mapping[str, object],
) -> ManualValidationAppArtifacts:
    """Validate representative manual metrics without loading labelled headlines."""
    frame = metrics.copy()
    run_metadata = dict(metadata)
    _require_columns(
        frame,
        MANUAL_VALIDATION_METRIC_COLUMNS,
        "sentiment_manual_validation_metrics.csv",
    )
    if frame.duplicated(["model", "metric"]).any():
        raise AppArtifactError("manual-validation metric rows are duplicated")
    numeric = [
        "value",
        "ci_low",
        "ci_high",
        "sample_count",
        "sampling_weight_sum",
        "effective_sample_size",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    model_rows = frame["model"].isin(["VADER", "FinBERT"])
    if frame.loc[model_rows, ["value", "sample_count"]].isna().any().any():
        raise AppArtifactError("manual-validation model metrics contain missing values")
    required_metrics = {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    }
    for model in ("VADER", "FinBERT"):
        available = set(frame.loc[frame["model"].eq(model), "metric"])
        if not required_metrics.issubset(available):
            raise AppArtifactError(f"manual-validation metrics are incomplete for {model}")
    bounded = frame.loc[
        model_rows
        & frame["metric"].isin(
            [
                *required_metrics,
                "precision_negative",
                "precision_neutral",
                "precision_positive",
                "recall_negative",
                "recall_neutral",
                "recall_positive",
                "predicted_negative_rate",
                "predicted_neutral_rate",
                "predicted_positive_rate",
            ]
        ),
        "value",
    ]
    if not bounded.between(0.0, 1.0).all():
        raise AppArtifactError("manual-validation classification metrics are outside [0, 1]")
    accuracy = frame.loc[model_rows & frame["metric"].eq("accuracy")]
    if len(accuracy) != 2:
        raise AppArtifactError("manual-validation accuracy rows are incomplete")
    if accuracy[["ci_low", "ci_high"]].isna().any().any():
        raise AppArtifactError("manual-validation accuracy intervals are missing")
    if not (
        accuracy["ci_low"].le(accuracy["value"])
        & accuracy["value"].le(accuracy["ci_high"])
        & accuracy["ci_low"].ge(0.0)
        & accuracy["ci_high"].le(1.0)
    ).all():
        raise AppArtifactError("manual-validation accuracy intervals are invalid")
    paired = frame.loc[frame["model"].eq("Paired comparison")]
    required_paired = {
        "mcnemar_exact_p_value",
        "weighted_accuracy_difference_finbert_minus_vader",
        "vader_only_correct",
        "finbert_only_correct",
    }
    if not required_paired.issubset(set(paired["metric"])):
        raise AppArtifactError("manual-validation paired comparison is incomplete")
    p_value = float(
        paired.loc[paired["metric"].eq("mcnemar_exact_p_value"), "value"].iloc[0]
    )
    if not 0.0 <= p_value <= 1.0:
        raise AppArtifactError("manual-validation McNemar p-value is invalid")

    required_metadata = {
        "phase_status",
        "representative_rows",
        "diagnostic_rows",
        "reviewer_attestation",
        "claims_status",
    }
    missing = sorted(required_metadata.difference(run_metadata))
    if missing:
        raise AppArtifactError(f"manual-validation metadata keys are missing: {missing}")
    if run_metadata["phase_status"] != "PASS":
        raise AppArtifactError("manual-validation metadata is not PASS")
    if int(run_metadata["representative_rows"]) != 100:
        raise AppArtifactError("manual-validation representative count is not 100")
    if int(run_metadata["diagnostic_rows"]) != 50:
        raise AppArtifactError("manual-validation diagnostic count is not 50")
    if not frame["sample_purpose"].eq("representative_evaluation").all():
        raise AppArtifactError("manual-validation metrics mix sampling purposes")
    return ManualValidationAppArtifacts(frame, run_metadata)


def load_manual_validation_app_artifacts(
    project_root: Path,
) -> ManualValidationAppArtifacts | None:
    """Load optional Phase 4 summaries while keeping labelled headlines offline."""
    root = Path(project_root)
    paths = {name: root / relative for name, relative in MANUAL_VALIDATION_RELATIVE_PATHS.items()}
    existing = {name: path.is_file() for name, path in paths.items()}
    if not any(existing.values()):
        return None
    if not all(existing.values()):
        missing = sorted(name for name, present in existing.items() if not present)
        raise AppArtifactError(f"manual-validation artifacts are incomplete: {missing}")
    try:
        metrics = pd.read_csv(paths["metrics"])
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise AppArtifactError(f"could not read manual-validation artifacts: {exc}") from exc
    if not isinstance(metadata, dict):
        raise AppArtifactError("manual-validation metadata must be a JSON object")
    return validate_manual_validation_app_artifacts(metrics, metadata)


def load_finbert_app_artifacts(
    project_root: Path,
    vader_sector_sentiment: pd.DataFrame,
) -> FinBERTAppArtifacts:
    """Read only approved precomputed FinBERT display artifacts."""
    root = Path(project_root)
    frames: dict[str, pd.DataFrame] = {}
    for name in ("sector_sentiment", "model_comparison", "disagreements"):
        relative_path = FINBERT_ARTIFACT_RELATIVE_PATHS[name]
        path = root / relative_path
        if not path.is_file():
            raise _finbert_unavailable(f"missing {relative_path.as_posix()}")
        try:
            frames[name] = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise _finbert_unavailable(f"could not read {relative_path.as_posix()}") from exc
    metadata_relative = FINBERT_ARTIFACT_RELATIVE_PATHS["metadata"]
    metadata_path = root / metadata_relative
    if not metadata_path.is_file():
        raise _finbert_unavailable(f"missing {metadata_relative.as_posix()}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _finbert_unavailable(f"could not read {metadata_relative.as_posix()}") from exc
    if not isinstance(metadata, dict):
        raise _finbert_unavailable("run metadata must be a JSON object")
    validated = validate_finbert_app_artifacts(
        frames["sector_sentiment"],
        frames["model_comparison"],
        frames["disagreements"],
        metadata,
        vader_sector_sentiment,
    )
    try:
        manual = load_manual_validation_app_artifacts(root)
    except AppArtifactError as exc:
        return replace(validated, manual_validation_error=str(exc))
    return replace(validated, manual_validation=manual)


def sentiment_model_history(
    vader_sector_sentiment: pd.DataFrame,
    finbert_sector_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Join validated precomputed sector indices without filling missing scores."""
    vader = vader_sector_sentiment[["date", "sector", "raw_sector_compound"]].rename(
        columns={"raw_sector_compound": "vader_score"}
    )
    finbert = finbert_sector_sentiment[["date", "sector", "raw_sector_finbert"]].rename(
        columns={"raw_sector_finbert": "finbert_score"}
    )
    history = vader.merge(
        finbert,
        on=["date", "sector"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not history["_merge"].eq("both").all():
        raise AppArtifactError("FinBERT and VADER date-sector grids do not align")
    return (
        history.drop(columns="_merge")
        .sort_values(["sector", "date"], kind="mergesort")
        .reset_index(drop=True)
    )


def latest_holdings(fund_weights: pd.DataFrame, fund: str) -> pd.DataFrame:
    """Return the latest target holdings for one validated fund."""
    selected = fund_weights.loc[fund_weights["fund"].eq(fund) & fund_weights["is_current"]].copy()
    if selected.empty:
        raise AppArtifactError(f"No current target holdings found for {fund}")
    columns = ["ticker", "target_weight"]
    if "sector" in selected:
        columns.append("sector")
    return (
        selected[columns]
        .sort_values(["target_weight", "ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )


def drawdown_history(fund_returns: pd.DataFrame, fund: str) -> pd.DataFrame:
    """Calculate drawdown using a running peak that includes the starting $1."""
    selected = fund_returns.loc[fund_returns["fund"].eq(fund), ["date", "growth_of_1"]]
    selected = selected.sort_values("date").reset_index(drop=True)
    if selected.empty:
        raise AppArtifactError(f"No fund returns found for {fund}")
    running_peak = selected["growth_of_1"].cummax().clip(lower=1.0)
    selected["drawdown"] = selected["growth_of_1"] / running_peak - 1
    return selected[["date", "drawdown"]]


def calculate_allocation_scenario(
    fund_returns: pd.DataFrame,
    allocations: Mapping[str, float],
    initial_value: float,
    *,
    tolerance: float = 1e-8,
) -> AllocationScenario:
    """Apply fixed fund allocations to the common historical return sample.

    Allocations are decimal fractions and are held fixed for this transparent
    scenario.  No forecast, optimiser, or personalised recommendation is made.
    """
    value = float(initial_value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("initial_value must be finite and positive")

    available_funds = list(fund_returns["fund"].drop_duplicates())
    if set(allocations) != set(available_funds):
        raise ValueError("allocations must specify every available fund exactly once")
    allocation = pd.Series(allocations, dtype=float).reindex(available_funds)
    if not np.isfinite(allocation).all() or (allocation < 0).any():
        raise ValueError("allocation fractions must be finite and non-negative")
    if not np.isclose(allocation.sum(), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("allocation fractions must sum to one")

    returns_wide = fund_returns.pivot(index="date", columns="fund", values="daily_return")
    returns_wide = returns_wide.reindex(columns=available_funds).sort_index()
    returns_wide = returns_wide.dropna(how="any")
    if returns_wide.empty:
        raise AppArtifactError("selected fund return histories have no common dates")
    portfolio_daily_return = returns_wide.to_numpy() @ allocation.to_numpy()
    portfolio_growth = np.cumprod(1.0 + portfolio_daily_return)
    history = pd.DataFrame(
        {
            "date": returns_wide.index,
            "daily_return": portfolio_daily_return,
            "growth_of_1": portfolio_growth,
            "scenario_value": value * portfolio_growth,
        }
    )

    intersected_growth = (1.0 + returns_wide).cumprod().iloc[-1]
    summary = pd.DataFrame(
        {
            "fund": available_funds,
            "allocation_fraction": allocation.to_numpy(),
            "initial_allocation": value * allocation.to_numpy(),
            "fund_growth_of_1": intersected_growth.to_numpy(),
        }
    )
    summary["historical_ending_value"] = summary["initial_allocation"] * summary["fund_growth_of_1"]
    summary["historical_gain_loss"] = (
        summary["historical_ending_value"] - summary["initial_allocation"]
    )
    return AllocationScenario(
        summary=summary,
        history=history,
        initial_value=value,
        ending_value=float(history["scenario_value"].iloc[-1]),
    )


def apply_annual_management_fee(
    scenario: AllocationScenario,
    annual_management_fee: float,
    *,
    periods_per_year: int = 252,
) -> FeeAdjustedAllocationScenario:
    """Apply a hypothetical annual fee as a daily drag to an allocation scenario.

    This never changes fund returns or fund-level fact sheets.  Allocation mixes
    use their common equity-trading-date intersection, so the product-level fee
    uses 252 periods per year.
    """
    fee = float(annual_management_fee)
    if not np.isfinite(fee) or not 0.0 <= fee < 1.0:
        raise ValueError("annual_management_fee must be finite and between 0 and 1")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    history = scenario.history.copy()
    gross = pd.to_numeric(history["daily_return"], errors="coerce")
    if gross.isna().any() or not np.isfinite(gross).all() or (gross <= -1).any():
        raise AppArtifactError("allocation scenario contains invalid gross returns")

    if fee == 0.0:
        net = gross.copy()
    else:
        daily_fee_factor = (1.0 - fee) ** (1.0 / periods_per_year)
        net = (1.0 + gross) * daily_fee_factor - 1.0
    net_growth = (1.0 + net).cumprod()
    history["fee_adjusted_daily_return"] = net
    history["fee_adjusted_growth_of_1"] = net_growth
    history["fee_adjusted_scenario_value"] = scenario.initial_value * net_growth
    return FeeAdjustedAllocationScenario(
        history=history,
        annual_management_fee=fee,
        gross_ending_value=scenario.ending_value,
        fee_adjusted_ending_value=float(net_growth.iloc[-1] * scenario.initial_value),
    )


__all__ = [
    "ARTIFACT_RELATIVE_PATHS",
    "EXPECTED_FUND_IDENTITIES",
    "EXPECTED_SECTORS",
    "FINBERT_ARTIFACT_RELATIVE_PATHS",
    "FINBERT_COMPARISON_COLUMNS",
    "FINBERT_DISAGREEMENT_COLUMNS",
    "FINBERT_SENTIMENT_COLUMNS",
    "FUND_LABELS",
    "FUND_RETURN_COLUMNS",
    "FUND_WEIGHT_COLUMNS",
    "MANUAL_VALIDATION_METRIC_COLUMNS",
    "MANUAL_VALIDATION_RELATIVE_PATHS",
    "PERFORMANCE_COLUMNS",
    "SENTIMENT_COLUMNS",
    "AllocationScenario",
    "AppArtifactError",
    "AppArtifacts",
    "FeeAdjustedAllocationScenario",
    "FinBERTAppArtifacts",
    "ManualValidationAppArtifacts",
    "apply_annual_management_fee",
    "calculate_allocation_scenario",
    "drawdown_history",
    "latest_holdings",
    "load_app_artifacts",
    "load_finbert_app_artifacts",
    "load_manual_validation_app_artifacts",
    "sentiment_model_history",
    "validate_app_artifacts",
    "validate_finbert_app_artifacts",
    "validate_manual_validation_app_artifacts",
]
