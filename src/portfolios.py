"""Portfolio construction and strict walk-forward backtesting.

This module contains reusable financial calculations only.  Project-level data
loading and fund orchestration remain in :mod:`scripts.run_part_b`.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

WEIGHT_TOLERANCE = 1e-8
VARIATION_TOLERANCE = 1e-6


class PortfolioOptimizationError(RuntimeError):
    """Raised when an optimiser fails or produces an invalid solution."""


class PortfolioValidationError(RuntimeError):
    """Raised when completed portfolio outputs fail an economic audit."""


@dataclass(frozen=True)
class WeightSolution:
    """Validated target weights and transparent optimisation metadata."""

    weights: pd.Series
    solver_success: bool
    solver_status: str
    solver_message: str
    objective_value: float
    covariance_scale: float


@dataclass(frozen=True)
class BacktestResult:
    """Daily returns, long-form weights, and one-row-per-rebalance audit."""

    fund_returns: pd.DataFrame
    fund_weights: pd.DataFrame
    rebalance_audit: pd.DataFrame


def _coerce_training_returns(training_returns: pd.DataFrame) -> pd.DataFrame:
    if training_returns.empty or training_returns.shape[1] == 0:
        raise ValueError("training_returns must contain observations and assets")
    if len(training_returns) < 2:
        raise ValueError("at least two return observations are required")
    if training_returns.columns.duplicated().any():
        raise ValueError("training return asset names must be unique")

    numeric = training_returns.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("training_returns must be finite; no imputation is allowed")
    return numeric.astype(float)


def _covariance_and_scale(
    training_returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    numeric = _coerce_training_returns(training_returns)
    covariance = _sample_covariance(numeric)

    magnitude = float(np.max(np.abs(covariance)))
    if magnitude <= np.finfo(float).eps:
        raise PortfolioOptimizationError(
            "covariance matrix has no measurable variation"
        )
    scale = max(1.0, 1.0 / magnitude)
    return covariance, covariance * scale, scale


def _sample_covariance(training_returns: pd.DataFrame) -> np.ndarray:
    """Return a finite symmetric sample covariance, including the zero matrix."""
    covariance = np.atleast_2d(
        training_returns.cov(ddof=1).to_numpy(dtype=float)
    )
    covariance = (covariance + covariance.T) / 2
    if not np.isfinite(covariance).all():
        raise PortfolioOptimizationError("covariance matrix is not finite")
    return covariance


def _validated_weights(
    values: Iterable[float],
    asset_names: pd.Index,
    *,
    tolerance: float = WEIGHT_TOLERANCE,
) -> pd.Series:
    weights = np.asarray(list(values), dtype=float)
    if len(weights) != len(asset_names):
        raise PortfolioOptimizationError("solver returned the wrong number of weights")
    if not np.isfinite(weights).all():
        raise PortfolioOptimizationError("solver returned non-finite weights")
    if float(weights.min()) < -tolerance or float(weights.max()) > 1 + tolerance:
        raise PortfolioOptimizationError("solver weights violate long-only bounds")
    if abs(float(weights.sum()) - 1.0) > tolerance:
        raise PortfolioOptimizationError("solver weights do not sum to one")

    # Remove only floating-point boundary noise already inside the validated
    # tolerance, then restore an exact fully invested sum.
    weights = np.clip(weights, 0.0, 1.0)
    weights = weights / weights.sum()
    if not np.isfinite(weights).all() or abs(float(weights.sum()) - 1.0) > tolerance:
        raise PortfolioOptimizationError("weight normalisation failed validation")
    return pd.Series(weights, index=asset_names, name="target_weight")


def equal_weight_solution(training_returns: pd.DataFrame) -> WeightSolution:
    """Return a validated 1/N benchmark on the supplied asset universe."""
    numeric = _coerce_training_returns(training_returns)
    covariance = _sample_covariance(numeric)
    raw_weights = np.full(numeric.shape[1], 1.0 / numeric.shape[1])
    weights = _validated_weights(raw_weights, numeric.columns)
    objective = float(weights.to_numpy() @ covariance @ weights.to_numpy())
    return WeightSolution(
        weights=weights,
        solver_success=True,
        solver_status="not_required",
        solver_message="deterministic 1/N weights",
        objective_value=objective,
        covariance_scale=1.0,
    )


def minimum_variance_solution(
    training_returns: pd.DataFrame,
    *,
    solver: Callable[..., Any] = minimize,
) -> WeightSolution:
    """Solve a scaled long-only, fully invested minimum-variance problem.

    The covariance matrix is multiplied by one positive scalar so its largest
    absolute element is at least one.  This improves numerical conditioning but
    cannot change the economic optimum.  The reported objective remains the raw
    daily portfolio variance, not the scaled solver objective.
    """
    numeric = _coerce_training_returns(training_returns)
    covariance, scaled_covariance, scale = _covariance_and_scale(numeric)
    number_of_assets = numeric.shape[1]
    initial_weights = np.full(number_of_assets, 1.0 / number_of_assets)

    def scaled_objective(weights: np.ndarray) -> float:
        return float(weights @ scaled_covariance @ weights)

    result = solver(
        scaled_objective,
        initial_weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * number_of_assets,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 1_000, "ftol": 1e-12, "disp": False},
    )
    status = str(getattr(result, "status", "unknown"))
    message = str(getattr(result, "message", "no solver message"))
    if not bool(getattr(result, "success", False)):
        raise PortfolioOptimizationError(
            f"minimum-variance solver failed (status={status}): {message}"
        )
    if not hasattr(result, "x"):
        raise PortfolioOptimizationError("minimum-variance solver returned no weights")

    weights = _validated_weights(result.x, numeric.columns)
    scaled_objective_value = scaled_objective(weights.to_numpy(dtype=float))
    reported_objective = getattr(result, "fun", None)
    if reported_objective is None or not np.isfinite(float(reported_objective)):
        raise PortfolioOptimizationError(
            "minimum-variance solver returned no finite objective value"
        )
    objective_tolerance = 1e-8 * max(1.0, abs(scaled_objective_value))
    if abs(float(reported_objective) - scaled_objective_value) > objective_tolerance:
        raise PortfolioOptimizationError(
            "minimum-variance solver objective does not match returned weights"
        )
    initial_objective = scaled_objective(initial_weights)
    if scaled_objective_value > initial_objective + objective_tolerance:
        raise PortfolioOptimizationError(
            "minimum-variance solver is worse than its feasible initial weights"
        )
    objective = float(weights.to_numpy() @ covariance @ weights.to_numpy())
    if not np.isfinite(objective) or objective < -WEIGHT_TOLERANCE:
        raise PortfolioOptimizationError("minimum-variance objective is invalid")
    return WeightSolution(
        weights=weights,
        solver_success=True,
        solver_status=status,
        solver_message=message,
        objective_value=objective,
        covariance_scale=scale,
    )


def monthly_rebalance_schedule(
    dates: Iterable[object],
    *,
    initial_window: int = 252,
) -> pd.DataFrame:
    """Return eligible month-end decisions and next-day holding starts."""
    if initial_window < 2:
        raise ValueError("initial_window must be at least two observations")
    calendar = pd.DatetimeIndex(pd.to_datetime(list(dates), utc=True))
    calendar = calendar.tz_convert(None).normalize().sort_values()
    if calendar.has_duplicates:
        raise ValueError("backtest dates must be unique")
    if len(calendar) <= initial_window:
        raise ValueError("return history is too short for the initial window")

    positions = pd.Series(np.arange(len(calendar)), index=calendar)
    month_ends = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).max()
    records: list[dict[str, object]] = []
    for decision_date in month_ends:
        position = int(positions.loc[decision_date])
        window_size = position + 1
        if window_size < initial_window or position + 1 >= len(calendar):
            continue
        records.append(
            {
                "decision_date": decision_date,
                "training_start_date": calendar[0],
                "training_end_date": decision_date,
                "first_holding_date": calendar[position + 1],
                "window_size": window_size,
            }
        )
    schedule = pd.DataFrame.from_records(records)
    if schedule.empty:
        raise ValueError("no eligible rebalance dates were produced")
    if not (schedule["training_end_date"] < schedule["first_holding_date"]).all():
        raise PortfolioValidationError("training overlaps the holding period")
    return schedule


def _prepare_return_panel(
    returns: pd.DataFrame,
    *,
    date_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    if date_col not in returns.columns:
        raise ValueError(f"returns is missing date column: {date_col}")
    panel = returns.copy()
    panel[date_col] = (
        pd.to_datetime(panel[date_col], utc=True).dt.tz_convert(None).dt.normalize()
    )
    if panel[date_col].duplicated().any():
        raise ValueError("return panel dates must be unique")
    panel = panel.sort_values(date_col).reset_index(drop=True)
    assets = [column for column in panel.columns if column != date_col]
    if not assets:
        raise ValueError("return panel must contain at least one asset")
    panel[assets] = panel[assets].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(panel[assets].to_numpy(dtype=float)).all():
        raise ValueError("return panel must be complete and finite")
    return panel, assets


def oos_backtest(
    returns: pd.DataFrame,
    *,
    fund: str,
    method: str,
    asset_family: str,
    initial_window: int = 252,
    date_col: str = "date",
    solver: Callable[..., Any] = minimize,
) -> BacktestResult:
    """Run a monthly expanding-window backtest with strict timing boundaries."""
    if method not in {"equal_weight", "min_variance"}:
        raise ValueError(f"unsupported portfolio method: {method}")
    panel, assets = _prepare_return_panel(returns, date_col=date_col)
    schedule = monthly_rebalance_schedule(
        panel[date_col],
        initial_window=initial_window,
    )

    date_positions = pd.Series(np.arange(len(panel)), index=panel[date_col])
    return_records: list[dict[str, object]] = []
    weight_records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []

    for rebalance_number, row in schedule.iterrows():
        decision_position = int(date_positions.loc[row["decision_date"]])
        training = panel.loc[:decision_position, assets]
        if len(training) != int(row["window_size"]):
            raise PortfolioValidationError("training window size audit mismatch")

        if method == "equal_weight":
            solution = equal_weight_solution(training)
        else:
            solution = minimum_variance_solution(training, solver=solver)

        first_holding_position = decision_position + 1
        if rebalance_number + 1 < len(schedule):
            next_decision = schedule.loc[rebalance_number + 1, "decision_date"]
            holding_end_position = int(date_positions.loc[next_decision])
        else:
            holding_end_position = len(panel) - 1
        holding = panel.loc[first_holding_position:holding_end_position]
        if holding.empty:
            raise PortfolioValidationError("rebalance has no holding-period returns")

        weights = solution.weights.reindex(assets)
        portfolio_returns = holding[assets].to_numpy(dtype=float) @ weights.to_numpy()
        for date, daily_return in zip(
            holding[date_col],
            portfolio_returns,
            strict=True,
        ):
            return_records.append(
                {
                    "date": date,
                    "fund": fund,
                    "method": method,
                    "asset_family": asset_family,
                    "decision_date": row["decision_date"],
                    "daily_return": float(daily_return),
                }
            )

        metadata = {
            "fund": fund,
            "method": method,
            "asset_family": asset_family,
            "decision_date": row["decision_date"],
            "training_start_date": row["training_start_date"],
            "training_end_date": row["training_end_date"],
            "first_holding_date": row["first_holding_date"],
            "window_size": int(row["window_size"]),
            "solver_success": solution.solver_success,
            "solver_status": solution.solver_status,
            "solver_message": solution.solver_message,
            "objective_value": solution.objective_value,
            "covariance_scale": solution.covariance_scale,
        }
        for ticker, target_weight in weights.items():
            weight_records.append(
                {
                    **metadata,
                    "ticker": ticker,
                    "target_weight": float(target_weight),
                }
            )
        audit_records.append(
            {
                **metadata,
                "target_weights": json.dumps(
                    {ticker: float(value) for ticker, value in weights.items()},
                    sort_keys=True,
                ),
            }
        )

    fund_returns = pd.DataFrame.from_records(return_records)
    fund_returns["growth_of_1"] = (1 + fund_returns["daily_return"]).cumprod()
    fund_weights = pd.DataFrame.from_records(weight_records)
    rebalance_audit = pd.DataFrame.from_records(audit_records)
    if fund_returns.duplicated(["fund", "date"]).any():
        raise PortfolioValidationError("fund return dates overlap across holdings")
    if not np.isfinite(fund_returns["daily_return"]).all():
        raise PortfolioValidationError("derived fund returns are not finite")
    return BacktestResult(fund_returns, fund_weights, rebalance_audit)


def concatenate_backtests(results: Iterable[BacktestResult]) -> BacktestResult:
    """Combine independently validated funds into common long-form outputs."""
    items = list(results)
    if not items:
        raise ValueError("at least one backtest result is required")
    return BacktestResult(
        fund_returns=pd.concat(
            [item.fund_returns for item in items],
            ignore_index=True,
        ),
        fund_weights=pd.concat(
            [item.fund_weights for item in items],
            ignore_index=True,
        ),
        rebalance_audit=pd.concat(
            [item.rebalance_audit for item in items],
            ignore_index=True,
        ),
    )


def validate_weight_variation(
    fund_weights: pd.DataFrame,
    *,
    dynamic_fund: str,
    benchmark_fund: str,
    tolerance: float = VARIATION_TOLERANCE,
) -> dict[str, float]:
    """Require dynamic weights to change over time and differ from 1/N."""
    required = {"fund", "decision_date", "ticker", "target_weight"}
    missing = sorted(required.difference(fund_weights.columns))
    if missing:
        raise ValueError(f"fund_weights is missing columns: {missing}")

    dynamic = fund_weights.loc[fund_weights["fund"].eq(dynamic_fund)].pivot(
        index="decision_date",
        columns="ticker",
        values="target_weight",
    )
    if len(dynamic) < 2:
        raise PortfolioValidationError("dynamic fund needs at least two rebalances")
    max_rebalance_change = float(dynamic.diff().abs().to_numpy()[1:].max())

    comparison = fund_weights.loc[
        fund_weights["fund"].isin([dynamic_fund, benchmark_fund])
    ].pivot(
        index=["decision_date", "ticker"],
        columns="fund",
        values="target_weight",
    )
    if dynamic_fund not in comparison or benchmark_fund not in comparison:
        raise PortfolioValidationError("funds do not share comparable weight keys")
    comparable = comparison[[dynamic_fund, benchmark_fund]].dropna()
    if comparable.empty:
        raise PortfolioValidationError("funds have no overlapping weights")
    max_method_difference = float(
        (comparable[dynamic_fund] - comparable[benchmark_fund]).abs().max()
    )
    if max_rebalance_change <= tolerance:
        raise PortfolioValidationError(
            "dynamic weights are stalled across rebalances"
        )
    if max_method_difference <= tolerance:
        raise PortfolioValidationError(
            "dynamic weights do not differ from the 1/N benchmark"
        )
    return {
        "max_rebalance_weight_change": max_rebalance_change,
        "max_method_weight_difference": max_method_difference,
    }


def performance_metrics(
    daily_returns: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Calculate the locked geometric return, risk, Sharpe, and drawdown metrics."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    values = pd.to_numeric(daily_returns, errors="coerce").to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("daily_returns must be non-empty and finite")
    if len(values) < 2:
        raise ValueError("at least two daily returns are required")
    wealth = np.cumprod(1 + values)
    if (wealth <= 0).any():
        raise ValueError("growth of one must remain positive")
    annualised_return = float(wealth[-1] ** (periods_per_year / len(values)) - 1)
    annualised_volatility = float(np.std(values, ddof=1) * np.sqrt(periods_per_year))
    sharpe = (
        float(np.mean(values) / np.std(values, ddof=1) * np.sqrt(periods_per_year))
        if annualised_volatility > 0
        else float("nan")
    )
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    drawdown = wealth / running_peak - 1
    return {
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": float(drawdown.min()),
    }


__all__ = [
    "BacktestResult",
    "PortfolioOptimizationError",
    "PortfolioValidationError",
    "WeightSolution",
    "concatenate_backtests",
    "equal_weight_solution",
    "minimum_variance_solution",
    "monthly_rebalance_schedule",
    "oos_backtest",
    "performance_metrics",
    "validate_weight_variation",
]
