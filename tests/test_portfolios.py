"""Unit tests for portfolio optimisers and validation boundaries."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from src import portfolios


def _training_returns() -> pd.DataFrame:
    rng = np.random.default_rng(5545)
    return pd.DataFrame(
        {
            "LOW": rng.normal(0.0002, 0.005, 300),
            "MEDIUM": rng.normal(0.0003, 0.012, 300),
            "HIGH": rng.normal(0.0004, 0.025, 300),
        }
    )


def test_equal_weight_is_finite_long_only_and_fully_invested() -> None:
    solution = portfolios.equal_weight_solution(_training_returns())

    assert np.isfinite(solution.weights).all()
    assert (solution.weights >= 0).all()
    assert solution.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert solution.weights.to_numpy() == pytest.approx(np.full(3, 1 / 3))
    assert solution.solver_status == "not_required"


def test_equal_weight_accepts_a_zero_covariance_training_sample() -> None:
    constant = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]})

    solution = portfolios.equal_weight_solution(constant)

    assert solution.weights.to_numpy() == pytest.approx([0.5, 0.5])
    assert solution.objective_value == pytest.approx(0.0, abs=0.0)


def test_minimum_variance_is_valid_scaled_and_improves_objective() -> None:
    returns = _training_returns()
    equal = portfolios.equal_weight_solution(returns)
    minimum = portfolios.minimum_variance_solution(returns)

    assert minimum.solver_success
    assert np.isfinite(minimum.weights).all()
    assert (minimum.weights >= -1e-12).all()
    assert minimum.weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert minimum.objective_value <= equal.objective_value + 1e-12
    assert minimum.covariance_scale > 1.0
    assert not np.allclose(minimum.weights, equal.weights, atol=1e-6)


def test_failed_solver_is_rejected_without_fallback_weights() -> None:
    def failed_solver(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            success=False,
            status=9,
            message="iteration limit reached",
            x=np.full(3, 1 / 3),
        )

    with pytest.raises(
        portfolios.PortfolioOptimizationError,
        match=r"solver failed.*iteration limit reached",
    ):
        portfolios.minimum_variance_solution(
            _training_returns(),
            solver=failed_solver,
        )


def test_successful_solver_with_inconsistent_objective_is_rejected() -> None:
    def inconsistent_solver(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            success=True,
            status=0,
            message="claimed success",
            x=np.full(3, 1 / 3),
            fun=-1.0,
        )

    with pytest.raises(
        portfolios.PortfolioOptimizationError,
        match="objective does not match returned weights",
    ):
        portfolios.minimum_variance_solution(
            _training_returns(),
            solver=inconsistent_solver,
        )


def test_stalled_dynamic_weight_history_is_rejected() -> None:
    decisions = pd.to_datetime(["2023-01-31", "2023-02-28"])
    records: list[dict[str, object]] = []
    for fund in ("dynamic", "benchmark"):
        for decision in decisions:
            for ticker in ("A", "B"):
                records.append(
                    {
                        "fund": fund,
                        "decision_date": decision,
                        "ticker": ticker,
                        "target_weight": 0.5,
                    }
                )

    with pytest.raises(
        portfolios.PortfolioValidationError,
        match="stalled across rebalances",
    ):
        portfolios.validate_weight_variation(
            pd.DataFrame.from_records(records),
            dynamic_fund="dynamic",
            benchmark_fund="benchmark",
        )


def test_performance_metrics_match_independent_definitions() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, 0.005])
    metrics = portfolios.performance_metrics(returns, periods_per_year=252)
    wealth = (1 + returns).cumprod()
    expected_return = wealth.iloc[-1] ** (252 / len(returns)) - 1
    expected_volatility = returns.std(ddof=1) * np.sqrt(252)
    expected_sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    expected_drawdown = (
        wealth / wealth.cummax().clip(lower=1.0) - 1
    ).min()

    assert metrics["annualised_return"] == pytest.approx(expected_return)
    assert metrics["annualised_volatility"] == pytest.approx(expected_volatility)
    assert metrics["sharpe_ratio"] == pytest.approx(expected_sharpe)
    assert metrics["maximum_drawdown"] == pytest.approx(expected_drawdown)


def test_drawdown_includes_starting_wealth_as_the_first_peak() -> None:
    metrics = portfolios.performance_metrics(pd.Series([-0.10, 0.20]))

    assert metrics["maximum_drawdown"] == pytest.approx(-0.10, abs=1e-15)


def test_performance_metrics_require_two_observations() -> None:
    with pytest.raises(ValueError, match="at least two daily returns"):
        portfolios.performance_metrics(pd.Series([0.01]))
