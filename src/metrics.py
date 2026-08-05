"""Performance calculations and app-artifact schema validation.

All core equity-calendar funds use the locked 252-day annualisation convention
and a zero risk-free rate.  File writing remains in ``scripts/run_part_b.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import portfolios

PERIODS_PER_YEAR = 252
CRYPTO_PERIODS_PER_YEAR = 365
RISK_FREE_RATE_ANNUAL = 0.0
WEIGHT_TOLERANCE = 1e-8
FUND_IDENTIFIERS = ["fund", "asset_family", "method"]


class ArtifactValidationError(RuntimeError):
    """Raised when fund artifacts are incomplete or mutually inconsistent."""


@dataclass(frozen=True)
class FundArtifacts:
    """Validated app-readable frames for returns, weights, and fact sheets."""

    fund_returns: pd.DataFrame
    fund_weights: pd.DataFrame
    performance_metrics: pd.DataFrame


def _normalise_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise", utc=True).dt.tz_convert(None).dt.normalize()


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ArtifactValidationError(f"{name} is missing columns: {missing}")


def _validate_identifiers(frame: pd.DataFrame, name: str) -> None:
    if frame[FUND_IDENTIFIERS].isna().any().any():
        raise ArtifactValidationError(f"{name} contains missing fund identifiers")
    for fund, group in frame.groupby("fund", sort=False):
        for column in ("asset_family", "method"):
            if group[column].nunique(dropna=False) != 1:
                raise ArtifactValidationError(
                    f"{name} has inconsistent {column} for fund {fund}"
                )


def prepare_fund_returns(fund_returns: pd.DataFrame) -> pd.DataFrame:
    """Validate daily fund returns and recalculate growth of one by fund."""
    required = {
        *FUND_IDENTIFIERS,
        "date",
        "decision_date",
        "daily_return",
    }
    _require_columns(fund_returns, required, "fund_returns")
    returns = fund_returns.copy()
    returns["date"] = _normalise_date(returns["date"])
    returns["decision_date"] = _normalise_date(returns["decision_date"])
    returns["daily_return"] = pd.to_numeric(returns["daily_return"], errors="coerce")
    _validate_identifiers(returns, "fund_returns")

    if returns.duplicated(["fund", "date"]).any():
        raise ArtifactValidationError("fund_returns contains duplicate fund-date rows")
    if not np.isfinite(returns["daily_return"]).all():
        raise ArtifactValidationError("fund_returns contains non-finite daily returns")
    if (returns["daily_return"] <= -1).any():
        raise ArtifactValidationError("daily returns at or below -100% invalidate wealth")
    if not (returns["decision_date"] < returns["date"]).all():
        raise ArtifactValidationError(
            "fund_returns contains a decision on or after its holding return"
        )

    returns = returns.sort_values(["fund", "date"]).reset_index(drop=True)
    returns["growth_of_1"] = returns.groupby("fund", sort=False)[
        "daily_return"
    ].transform(lambda values: (1 + values).cumprod())
    if not np.isfinite(returns["growth_of_1"]).all() or (
        returns["growth_of_1"] <= 0
    ).any():
        raise ArtifactValidationError("growth_of_1 must remain finite and positive")

    columns = [
        *FUND_IDENTIFIERS,
        "date",
        "decision_date",
        "daily_return",
        "growth_of_1",
    ]
    return returns[columns]


def prepare_fund_weights(
    fund_weights: pd.DataFrame,
    *,
    tolerance: float = WEIGHT_TOLERANCE,
) -> pd.DataFrame:
    """Validate target weights and label each fund's latest holdings."""
    required = {
        *FUND_IDENTIFIERS,
        "decision_date",
        "ticker",
        "target_weight",
    }
    _require_columns(fund_weights, required, "fund_weights")
    weights = fund_weights.copy()
    weights["decision_date"] = _normalise_date(weights["decision_date"])
    weights["target_weight"] = pd.to_numeric(
        weights["target_weight"],
        errors="coerce",
    )
    _validate_identifiers(weights, "fund_weights")

    keys = ["fund", "decision_date", "ticker"]
    if weights.duplicated(keys).any():
        raise ArtifactValidationError("fund_weights contains duplicate target weights")
    if not np.isfinite(weights["target_weight"]).all():
        raise ArtifactValidationError("fund_weights contains non-finite weights")
    if (weights["target_weight"] < -tolerance).any() or (
        weights["target_weight"] > 1 + tolerance
    ).any():
        raise ArtifactValidationError("fund_weights violates long-only bounds")

    sums = weights.groupby(["fund", "decision_date"])["target_weight"].sum()
    if ((sums - 1).abs() > tolerance).any():
        raise ArtifactValidationError("target weights do not sum to one")

    weights["date"] = weights["decision_date"]
    latest_dates = weights.groupby("fund")["decision_date"].transform("max")
    weights["is_current"] = weights["decision_date"].eq(latest_dates)
    current = weights.loc[weights["is_current"]]
    current_sums = current.groupby("fund")["target_weight"].sum()
    if set(current_sums.index) != set(weights["fund"].unique()) or (
        (current_sums - 1).abs() > tolerance
    ).any():
        raise ArtifactValidationError("current holdings are missing or incomplete")

    front = [
        *FUND_IDENTIFIERS,
        "date",
        "decision_date",
        "ticker",
        "target_weight",
        "is_current",
    ]
    remainder = [column for column in weights.columns if column not in front]
    return weights.sort_values(keys).reset_index(drop=True)[front + remainder]


def build_performance_metrics(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    *,
    periods_per_year: int = PERIODS_PER_YEAR,
    periods_per_year_by_family: dict[str, int] | None = None,
    risk_free_rate_annual: float = RISK_FREE_RATE_ANNUAL,
) -> pd.DataFrame:
    """Build one consistent fact-sheet performance row per fund."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if risk_free_rate_annual != 0.0:
        raise ValueError("the locked core design requires a 0% risk-free rate")

    returns = prepare_fund_returns(fund_returns)
    weights = prepare_fund_weights(fund_weights)
    current_dates = (
        weights.loc[weights["is_current"]]
        .groupby("fund")["decision_date"]
        .first()
    )

    records: list[dict[str, object]] = []
    for fund, group in returns.groupby("fund", sort=True):
        group = group.sort_values("date")
        fund_periods = (
            periods_per_year_by_family.get(group["asset_family"].iloc[0], periods_per_year)
            if periods_per_year_by_family is not None
            else periods_per_year
        )
        if fund_periods <= 0:
            raise ValueError("periods_per_year values must be positive")
        statistics = portfolios.performance_metrics(
            group["daily_return"],
            periods_per_year=fund_periods,
        )
        annualised_mean_excess = float(
            group["daily_return"].mean() * fund_periods
        )
        records.append(
            {
                "fund": fund,
                "asset_family": group["asset_family"].iloc[0],
                "method": group["method"].iloc[0],
                "as_of_date": group["date"].iloc[-1],
                "sample_start_date": group["date"].iloc[0],
                "sample_end_date": group["date"].iloc[-1],
                "current_holdings_date": current_dates.loc[fund],
                "observations": len(group),
                "periods_per_year": fund_periods,
                "risk_free_rate_annual": risk_free_rate_annual,
                "annual_return_method": "geometric",
                "final_growth_of_1": float(group["growth_of_1"].iloc[-1]),
                "annualised_return": statistics["annualised_return"],
                "annualised_volatility": statistics["annualised_volatility"],
                "annualised_mean_excess_return": annualised_mean_excess,
                "sharpe_ratio": statistics["sharpe_ratio"],
                "maximum_drawdown": statistics["maximum_drawdown"],
            }
        )
    performance = pd.DataFrame.from_records(records)
    value_columns = [
        "final_growth_of_1",
        "annualised_return",
        "annualised_volatility",
        "annualised_mean_excess_return",
        "sharpe_ratio",
        "maximum_drawdown",
    ]
    if performance.empty or not np.isfinite(
        performance[value_columns].to_numpy(dtype=float)
    ).all():
        raise ArtifactValidationError("performance metrics must be finite")
    expected_sharpe = (
        performance["annualised_mean_excess_return"]
        / performance["annualised_volatility"]
    )
    if not np.allclose(
        performance["sharpe_ratio"],
        expected_sharpe,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ArtifactValidationError("Sharpe ratios do not match the locked definition")
    return performance


def build_fund_artifacts(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    *,
    periods_per_year_by_family: dict[str, int] | None = None,
) -> FundArtifacts:
    """Prepare and reconcile all fund artifacts before any file is written."""
    returns = prepare_fund_returns(fund_returns)
    weights = prepare_fund_weights(fund_weights)
    performance = build_performance_metrics(
        returns,
        weights,
        periods_per_year_by_family=periods_per_year_by_family,
    )

    return_funds = set(returns["fund"])
    weight_funds = set(weights["fund"])
    performance_funds = set(performance["fund"])
    if not return_funds == weight_funds == performance_funds:
        raise ArtifactValidationError("fund names differ across artifacts")

    return_identity = returns[FUND_IDENTIFIERS].drop_duplicates().sort_values("fund")
    weight_identity = weights[FUND_IDENTIFIERS].drop_duplicates().sort_values("fund")
    metric_identity = performance[FUND_IDENTIFIERS].drop_duplicates().sort_values("fund")
    pd.testing.assert_frame_equal(
        return_identity.reset_index(drop=True),
        weight_identity.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        return_identity.reset_index(drop=True),
        metric_identity.reset_index(drop=True),
    )

    return_decisions = returns[["fund", "decision_date"]].drop_duplicates()
    weight_decisions = weights[["fund", "decision_date"]].drop_duplicates()
    decision_check = return_decisions.merge(
        weight_decisions,
        on=["fund", "decision_date"],
        how="left",
        indicator=True,
    )
    if not decision_check["_merge"].eq("both").all():
        raise ArtifactValidationError("return rows reference unknown target weights")

    return_end_dates = returns.groupby("fund")["date"].max().sort_index()
    metric_end_dates = performance.set_index("fund")["sample_end_date"].sort_index()
    if not return_end_dates.equals(metric_end_dates):
        raise ArtifactValidationError("performance sample dates differ from returns")
    return FundArtifacts(returns, weights, performance)


__all__ = [
    "PERIODS_PER_YEAR",
    "CRYPTO_PERIODS_PER_YEAR",
    "RISK_FREE_RATE_ANNUAL",
    "ArtifactValidationError",
    "FundArtifacts",
    "build_fund_artifacts",
    "build_performance_metrics",
    "prepare_fund_returns",
    "prepare_fund_weights",
]
