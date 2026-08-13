"""Core assessment evidence derived from the precise Part B CSV artifacts.

The functions in this module do not load hosted data and do not run models. They
validate and reshape the precomputed fund and sentiment artifacts for report
tables and figures. Report-facing tables use three significant digits; source
artifacts retain their original precision.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

CORE_FUNDS = (
    "combined_equal_weight",
    "combined_min_variance",
    "combined_active_sector_allocation",
    "combined_growth_sector_allocation",
    "combined_aggressive_sector_allocation",
    "equity_equal_weight",
    "equity_sentiment_tilt",
    "crypto_equal_weight",
    "crypto_min_variance",
)
COMBINED_FUNDS = (
    "combined_equal_weight",
    "combined_min_variance",
    "combined_active_sector_allocation",
    "combined_growth_sector_allocation",
    "combined_aggressive_sector_allocation",
)
FUSION_FUNDS = (
    "equity_equal_weight",
    "equity_sentiment_tilt",
)
FUND_LABELS = {
    "combined_equal_weight": "Combined 1/N",
    "combined_min_variance": "Combined minimum variance",
    "combined_active_sector_allocation": "Combined active sector allocation",
    "combined_growth_sector_allocation": "Balanced growth sector allocation",
    "combined_aggressive_sector_allocation": "Aggressive sector and crypto allocation",
    "equity_equal_weight": "Equity 1/N",
    "equity_sentiment_tilt": "Equity sentiment tilt",
    "crypto_equal_weight": "Crypto equal weight",
    "crypto_min_variance": "Crypto minimum variance",
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


class EvidenceValidationError(RuntimeError):
    """Raised when report evidence cannot be traced to valid core artifacts."""


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise EvidenceValidationError(f"{name} is missing columns: {missing}")


def _normalise_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise", utc=True).dt.tz_convert(None).dt.normalize()


def _round_significant(value: float, digits: int = 3) -> float:
    """Round a finite report value to a fixed number of significant digits."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise EvidenceValidationError("report-facing values must be finite")
    if numeric == 0.0:
        return 0.0
    decimals = digits - math.floor(math.log10(abs(numeric))) - 1
    return float(round(numeric, decimals))


def _ordered_core(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    _require_columns(frame, {"fund", "asset_family", "method"}, name)
    core = frame.loc[frame["fund"].isin(CORE_FUNDS)].copy()
    observed = set(core["fund"])
    expected = set(CORE_FUNDS)
    if observed != expected:
        raise EvidenceValidationError(
            f"{name} core funds differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    order = {fund: position for position, fund in enumerate(CORE_FUNDS)}
    core["_fund_order"] = core["fund"].map(order)
    return core.sort_values("_fund_order", kind="mergesort").drop(columns="_fund_order")


def build_performance_report_table(performance: pd.DataFrame) -> pd.DataFrame:
    """Create the four-core-fund performance table at report precision."""
    required = {
        "fund",
        "asset_family",
        "method",
        "sample_start_date",
        "sample_end_date",
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
    _require_columns(performance, required, "performance_metrics")
    core = _ordered_core(performance, "performance_metrics")
    if core.duplicated("fund").any():
        raise EvidenceValidationError("performance_metrics has duplicate core funds")
    core["sample_start_date"] = _normalise_date(core["sample_start_date"])
    core["sample_end_date"] = _normalise_date(core["sample_end_date"])
    if not core["annual_return_method"].eq("geometric").all():
        raise EvidenceValidationError("core annual returns must be geometric")
    expected_periods = core["asset_family"].map({"crypto": 365}).fillna(252)
    if not core["periods_per_year"].eq(expected_periods).all():
        raise EvidenceValidationError("report funds have invalid native annualisation")
    if not np.allclose(core["risk_free_rate_annual"], 0.0, atol=0.0, rtol=0.0):
        raise EvidenceValidationError("core Sharpe ratios must use a zero risk-free rate")

    report = pd.DataFrame(
        {
            "fund": core["fund"],
            "fund_label": core["fund"].map(FUND_LABELS),
            "asset_family": core["asset_family"],
            "method": core["method"],
            "sample_start_date": core["sample_start_date"],
            "sample_end_date": core["sample_end_date"],
            "observations": core["observations"].astype(int),
            "periods_per_year": core["periods_per_year"].astype(int),
            "risk_free_rate_pct": core["risk_free_rate_annual"].map(
                lambda value: _round_significant(100.0 * value)
            ),
            "final_growth_of_1_dollars": core["final_growth_of_1"].map(
                _round_significant
            ),
            "geometric_annual_return_pct": core["annualised_return"].map(
                lambda value: _round_significant(100.0 * value)
            ),
            "annualised_volatility_pct": core["annualised_volatility"].map(
                lambda value: _round_significant(100.0 * value)
            ),
            "sharpe_ratio": core["sharpe_ratio"].map(_round_significant),
            "maximum_drawdown_pct": core["maximum_drawdown"].map(
                lambda value: _round_significant(100.0 * value)
            ),
        }
    )
    report["calculation_definition"] = report["periods_per_year"].map(
        lambda periods: (
            f"Growth=product(1+r); geometric annual return=growth^({periods}/n)-1; "
            f"volatility=sample SD(r)*sqrt({periods}); Sharpe=mean(r)*{periods}/volatility at "
            "0% risk-free; drawdown=wealth/running peak (including starting $1)-1."
        )
    )
    report["source_artifact"] = "results/tables/performance_metrics.csv"
    return report.reset_index(drop=True)


def build_return_paths(fund_returns: pd.DataFrame) -> pd.DataFrame:
    """Return exact core wealth and drawdown paths from logged daily returns."""
    required = {
        "fund",
        "asset_family",
        "method",
        "date",
        "daily_return",
        "growth_of_1",
    }
    _require_columns(fund_returns, required, "fund_returns")
    core = _ordered_core(fund_returns, "fund_returns")
    core["date"] = _normalise_date(core["date"])
    core["daily_return"] = pd.to_numeric(core["daily_return"], errors="coerce")
    core["growth_of_1"] = pd.to_numeric(core["growth_of_1"], errors="coerce")
    if core.duplicated(["fund", "date"]).any():
        raise EvidenceValidationError("fund_returns has duplicate core fund-date rows")
    if not np.isfinite(core[["daily_return", "growth_of_1"]]).all().all():
        raise EvidenceValidationError("core return paths contain non-finite values")
    core = core.sort_values(["fund", "date"], kind="mergesort")
    expected_growth = core.groupby("fund", sort=False)["daily_return"].transform(
        lambda returns: (1.0 + returns).cumprod()
    )
    if not np.allclose(
        core["growth_of_1"],
        expected_growth,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise EvidenceValidationError("stored growth_of_1 does not match daily returns")
    core["drawdown"] = core.groupby("fund", sort=False)["growth_of_1"].transform(
        lambda wealth: wealth / wealth.cummax().clip(lower=1.0) - 1.0
    )
    core["fund_label"] = core["fund"].map(FUND_LABELS)
    return core[
        [
            "fund",
            "fund_label",
            "asset_family",
            "method",
            "date",
            "daily_return",
            "growth_of_1",
            "drawdown",
        ]
    ].reset_index(drop=True)


def build_intersected_comparison_paths(fund_returns: pd.DataFrame) -> pd.DataFrame:
    """Rebase core funds on their explicit shared date intersection for comparison."""
    paths = build_return_paths(fund_returns)
    date_sets = [set(group["date"]) for _, group in paths.groupby("fund", sort=True)]
    common_dates = set.intersection(*date_sets)
    if not common_dates:
        raise EvidenceValidationError("core fund histories have no common comparison dates")
    comparison = paths.loc[paths["date"].isin(common_dates)].copy()
    comparison = comparison.sort_values(["fund", "date"], kind="mergesort")
    comparison["growth_of_1"] = comparison.groupby("fund", sort=False)["daily_return"].transform(
        lambda values: (1.0 + values).cumprod()
    )
    comparison["drawdown"] = comparison.groupby("fund", sort=False)["growth_of_1"].transform(
        lambda wealth: wealth / wealth.cummax().clip(lower=1.0) - 1.0
    )
    comparison["comparison_start_date"] = comparison["date"].min()
    comparison["comparison_end_date"] = comparison["date"].max()
    return comparison.reset_index(drop=True)


def build_common_period_return_paths(fund_returns: pd.DataFrame) -> pd.DataFrame:
    """Rebase core funds over one chronological window on native calendars.

    The latest core-fund start and earliest core-fund end define the window.
    Equity-calendar funds retain their trading observations and crypto-only
    funds retain all seven-day observations inside that same date range.
    """
    paths = build_return_paths(fund_returns)
    ranges = paths.groupby("fund")["date"].agg(["min", "max"])
    comparison_start = ranges["min"].max()
    comparison_end = ranges["max"].min()
    if comparison_start > comparison_end:
        raise EvidenceValidationError("core fund histories have no common date window")

    comparison = paths.loc[
        paths["date"].between(comparison_start, comparison_end)
    ].copy()
    if set(comparison["fund"]) != set(CORE_FUNDS):
        raise EvidenceValidationError("common-period comparison is missing core funds")
    comparison = comparison.sort_values(["fund", "date"], kind="mergesort")
    comparison["growth_of_1"] = comparison.groupby("fund", sort=False)[
        "daily_return"
    ].transform(lambda values: (1.0 + values).cumprod())
    comparison["drawdown"] = comparison.groupby("fund", sort=False)[
        "growth_of_1"
    ].transform(lambda wealth: wealth / wealth.cummax().clip(lower=1.0) - 1.0)
    comparison["comparison_start_date"] = comparison_start
    comparison["comparison_end_date"] = comparison_end
    return comparison.reset_index(drop=True)


def build_common_period_performance_table(
    fund_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate report metrics over the shared window on native calendars."""
    paths = build_common_period_return_paths(fund_returns)
    records: list[dict[str, object]] = []
    for fund in CORE_FUNDS:
        group = paths.loc[paths["fund"].eq(fund)].sort_values("date")
        values = group["daily_return"].to_numpy(dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise EvidenceValidationError(f"invalid common-period returns for {fund}")
        periods = 365 if group["asset_family"].iloc[0] == "crypto" else 252
        wealth = np.cumprod(1.0 + values)
        volatility = float(np.std(values, ddof=1) * np.sqrt(periods))
        sharpe = float(
            np.mean(values) / np.std(values, ddof=1) * np.sqrt(periods)
        )
        running_peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
        maximum_drawdown = float(np.min(wealth / running_peak - 1.0))
        records.append(
            {
                "fund": fund,
                "fund_label": FUND_LABELS[fund],
                "asset_family": group["asset_family"].iloc[0],
                "method": group["method"].iloc[0],
                "sample_start_date": group["comparison_start_date"].iloc[0],
                "sample_end_date": group["comparison_end_date"].iloc[0],
                "observations": len(values),
                "periods_per_year": periods,
                "risk_free_rate_pct": 0.0,
                "final_growth_of_1_dollars": _round_significant(wealth[-1]),
                "geometric_annual_return_pct": _round_significant(
                    100.0 * (wealth[-1] ** (periods / len(values)) - 1.0)
                ),
                "annualised_volatility_pct": _round_significant(
                    100.0 * volatility
                ),
                "sharpe_ratio": _round_significant(sharpe),
                "maximum_drawdown_pct": _round_significant(
                    100.0 * maximum_drawdown
                ),
                "calculation_definition": (
                    f"Common chronological window; native calendar; "
                    f"growth=product(1+r); geometric annual return="
                    f"growth^({periods}/n)-1; volatility=sample SD(r)*sqrt({periods}); "
                    f"Sharpe=mean(r)*{periods}/volatility at 0% risk-free; "
                    "drawdown=wealth/running peak (including starting $1)-1."
                ),
                "source_artifact": "results/data/fund_returns.csv",
            }
        )
    return pd.DataFrame.from_records(records)


def build_combined_weight_history(fund_weights: pd.DataFrame) -> pd.DataFrame:
    """Aggregate combined-fund ticker weights into equity and crypto shares."""
    required = {
        "fund",
        "asset_family",
        "method",
        "decision_date",
        "ticker",
        "target_weight",
    }
    _require_columns(fund_weights, required, "fund_weights")
    weights = fund_weights.loc[fund_weights["fund"].isin(COMBINED_FUNDS)].copy()
    if set(weights["fund"]) != set(COMBINED_FUNDS):
        raise EvidenceValidationError("both combined fund methods are required")
    weights["decision_date"] = _normalise_date(weights["decision_date"])
    weights["target_weight"] = pd.to_numeric(
        weights["target_weight"], errors="coerce"
    )
    if weights.duplicated(["fund", "decision_date", "ticker"]).any():
        raise EvidenceValidationError("combined weights contain duplicate keys")
    if not np.isfinite(weights["target_weight"]).all():
        raise EvidenceValidationError("combined weights contain non-finite values")
    weights["holding_asset_class"] = np.where(
        weights["ticker"].str.endswith("-USD"), "Crypto", "Equity"
    )
    history = (
        weights.groupby(
            [
                "fund",
                "asset_family",
                "method",
                "decision_date",
                "holding_asset_class",
            ],
            as_index=False,
            sort=True,
        )["target_weight"]
        .sum()
        .sort_values(
            ["fund", "decision_date", "holding_asset_class"], kind="mergesort"
        )
    )
    sums = history.groupby(["fund", "decision_date"])["target_weight"].sum()
    if not np.allclose(sums, 1.0, atol=1e-12, rtol=1e-12):
        raise EvidenceValidationError("combined asset-class weights do not sum to one")
    class_counts = history.groupby(["fund", "decision_date"])[
        "holding_asset_class"
    ].nunique()
    if not class_counts.eq(2).all():
        raise EvidenceValidationError("combined funds must contain equity and crypto")
    history["target_weight_pct"] = 100.0 * history["target_weight"]
    history["fund_label"] = history["fund"].map(FUND_LABELS)
    return history.reset_index(drop=True)


def build_combined_ticker_weight_history(
    fund_weights: pd.DataFrame,
    *,
    top_ticker_count: int = 6,
) -> pd.DataFrame:
    """Retain the largest minimum-variance ticker weights and group the rest.

    The ranking is only a descriptive display choice: tickers are selected by
    their maximum logged target weight in the combined minimum-variance fund.
    It never feeds back into portfolio formation or performance calculations.
    """
    if top_ticker_count <= 0:
        raise ValueError("top_ticker_count must be positive")
    required = {
        "fund",
        "asset_family",
        "method",
        "decision_date",
        "ticker",
        "target_weight",
    }
    _require_columns(fund_weights, required, "fund_weights")
    weights = fund_weights.loc[fund_weights["fund"].isin(COMBINED_FUNDS)].copy()
    if set(weights["fund"]) != set(COMBINED_FUNDS):
        raise EvidenceValidationError("both combined fund methods are required")
    weights["decision_date"] = _normalise_date(weights["decision_date"])
    weights["target_weight"] = pd.to_numeric(
        weights["target_weight"], errors="coerce"
    )
    if weights.duplicated(["fund", "decision_date", "ticker"]).any():
        raise EvidenceValidationError("combined weights contain duplicate keys")
    if not np.isfinite(weights["target_weight"]).all():
        raise EvidenceValidationError("combined weights contain non-finite values")

    minimum_variance = weights.loc[
        weights["fund"].eq("combined_min_variance")
    ]
    peak_weights = (
        minimum_variance.groupby("ticker", as_index=False)["target_weight"]
        .max()
        .sort_values(["target_weight", "ticker"], ascending=[False, True])
    )
    top_tickers = tuple(peak_weights.head(top_ticker_count)["ticker"])
    if len(top_tickers) != top_ticker_count:
        raise EvidenceValidationError("not enough combined tickers for display selection")
    weights["display_holding"] = np.where(
        weights["ticker"].isin(top_tickers), weights["ticker"], "Other assets"
    )
    history = (
        weights.groupby(
            [
                "fund",
                "asset_family",
                "method",
                "decision_date",
                "display_holding",
            ],
            as_index=False,
            sort=True,
        )["target_weight"]
        .sum()
        .sort_values(["fund", "decision_date", "display_holding"], kind="mergesort")
    )
    sums = history.groupby(["fund", "decision_date"])["target_weight"].sum()
    if not np.allclose(sums, 1.0, atol=1e-12, rtol=1e-12):
        raise EvidenceValidationError("displayed combined ticker weights do not sum to one")
    category_order = (*top_tickers, "Other assets")
    history["display_holding"] = pd.Categorical(
        history["display_holding"], categories=category_order, ordered=True
    )
    history["target_weight_pct"] = 100.0 * history["target_weight"]
    history["fund_label"] = history["fund"].map(FUND_LABELS)
    history["selection_definition"] = (
        f"Top {top_ticker_count} tickers by maximum logged combined minimum-variance "
        "target weight; remaining tickers grouped as Other assets."
    )
    return history.sort_values(
        ["fund", "decision_date", "display_holding"], kind="mergesort"
    ).reset_index(drop=True)


def build_sector_sentiment_series(sector_index: pd.DataFrame) -> pd.DataFrame:
    """Validate and retain the retrospective ten-sector daily sentiment index."""
    required = {
        "date",
        "sector",
        "raw_sector_compound",
        "observed_ticker_count",
        "possible_ticker_count",
        "ticker_coverage_share",
        "has_observed_news",
    }
    _require_columns(sector_index, required, "sector_sentiment_index")
    index = sector_index[list(required)].copy()
    index["date"] = _normalise_date(index["date"])
    if index.duplicated(["date", "sector"]).any():
        raise EvidenceValidationError("sector sentiment contains duplicate date-sector rows")
    if set(index["sector"]) != set(EXPECTED_SECTORS):
        raise EvidenceValidationError("sector sentiment does not contain exactly ten sectors")
    no_news = ~index["has_observed_news"].astype(bool)
    if index.loc[no_news, "raw_sector_compound"].notna().any():
        raise EvidenceValidationError("missing-news sector days were converted to scores")
    observed = index.loc[~no_news, "raw_sector_compound"]
    if observed.isna().any() or not observed.between(-1.0, 1.0).all():
        raise EvidenceValidationError("observed raw sector scores must lie in [-1, 1]")
    index["sector"] = pd.Categorical(
        index["sector"], categories=EXPECTED_SECTORS, ordered=True
    )
    return index.sort_values(["sector", "date"], kind="mergesort").reset_index(
        drop=True
    )


def build_fusion_report_table(fusion_comparison: pd.DataFrame) -> pd.DataFrame:
    """Create the locked base-versus-one-day-sentiment comparison table."""
    required = {
        "role",
        "fund",
        "asset_family",
        "method",
        "sample_start_date",
        "sample_end_date",
        "observations",
        "rebalance_count",
        "periods_per_year",
        "risk_free_rate_annual",
        "transaction_cost_bps",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "average_monthly_target_turnover",
        "annualised_return_difference_vs_base",
        "annualised_volatility_difference_vs_base",
        "sharpe_ratio_difference_vs_base",
        "maximum_drawdown_difference_vs_base",
    }
    _require_columns(fusion_comparison, required, "fusion_comparison")
    selected = fusion_comparison.loc[
        fusion_comparison["fund"].isin(FUSION_FUNDS)
    ].copy()
    if set(selected["fund"]) != set(FUSION_FUNDS) or set(selected["role"]) != {
        "base",
        "enhanced",
    }:
        raise EvidenceValidationError("locked base and sentiment fusion rows are required")
    selected["sample_start_date"] = _normalise_date(selected["sample_start_date"])
    selected["sample_end_date"] = _normalise_date(selected["sample_end_date"])
    if selected[["sample_start_date", "sample_end_date"]].nunique().max() != 1:
        raise EvidenceValidationError("fusion funds do not share the same sample")
    if not selected["periods_per_year"].eq(252).all():
        raise EvidenceValidationError("fusion evidence must use 252 periods per year")
    if not np.allclose(selected["risk_free_rate_annual"], 0.0, atol=0.0, rtol=0.0):
        raise EvidenceValidationError("fusion evidence must use a 0% risk-free rate")
    if not np.allclose(selected["transaction_cost_bps"], 0.0, atol=0.0, rtol=0.0):
        raise EvidenceValidationError("fusion evidence must use 0 bps costs")
    order = {"base": 0, "enhanced": 1}
    selected["_role_order"] = selected["role"].map(order)
    selected = selected.sort_values("_role_order", kind="mergesort")

    report = pd.DataFrame(
        {
            "role": selected["role"],
            "fund": selected["fund"],
            "fund_label": selected["fund"].map(FUND_LABELS),
            "asset_family": selected["asset_family"],
            "method": selected["method"],
            "sample_start_date": selected["sample_start_date"],
            "sample_end_date": selected["sample_end_date"],
            "observations": selected["observations"].astype(int),
            "rebalances": selected["rebalance_count"].astype(int),
            "periods_per_year": selected["periods_per_year"].astype(int),
            "risk_free_rate_pct": selected["risk_free_rate_annual"].map(
                lambda value: _round_significant(100.0 * value)
            ),
            "transaction_cost_bps": selected["transaction_cost_bps"].map(
                _round_significant
            ),
            "geometric_annual_return_pct": selected["annualised_return"].map(
                lambda value: _round_significant(100.0 * value)
            ),
            "annualised_volatility_pct": selected["annualised_volatility"].map(
                lambda value: _round_significant(100.0 * value)
            ),
            "sharpe_ratio": selected["sharpe_ratio"].map(_round_significant),
            "maximum_drawdown_pct": selected["maximum_drawdown"].map(
                lambda value: _round_significant(100.0 * value)
            ),
            "average_monthly_target_turnover_pct": selected[
                "average_monthly_target_turnover"
            ].map(lambda value: _round_significant(100.0 * value)),
            "annual_return_difference_pct_points": selected[
                "annualised_return_difference_vs_base"
            ].map(lambda value: _round_significant(100.0 * value)),
            "volatility_difference_pct_points": selected[
                "annualised_volatility_difference_vs_base"
            ].map(lambda value: _round_significant(100.0 * value)),
            "sharpe_difference": selected["sharpe_ratio_difference_vs_base"].map(
                _round_significant
            ),
            "drawdown_difference_pct_points": selected[
                "maximum_drawdown_difference_vs_base"
            ].map(lambda value: _round_significant(100.0 * value)),
        }
    )
    report["calculation_definition"] = (
        "Matched equity samples and rebalances; geometric return and drawdown use "
        "the core definitions; turnover is half the L1 target-weight change; "
        "differences are enhanced minus base."
    )
    report["source_artifact"] = "results/tables/fusion_comparison.csv"
    return report.reset_index(drop=True)


__all__ = [
    "COMBINED_FUNDS",
    "CORE_FUNDS",
    "EXPECTED_SECTORS",
    "FUND_LABELS",
    "FUSION_FUNDS",
    "EvidenceValidationError",
    "build_combined_ticker_weight_history",
    "build_combined_weight_history",
    "build_fusion_report_table",
    "build_performance_report_table",
    "build_common_period_performance_table",
    "build_common_period_return_paths",
    "build_intersected_comparison_paths",
    "build_return_paths",
    "build_sector_sentiment_series",
]
