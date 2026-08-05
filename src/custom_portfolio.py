"""Build-time asset returns and app-time custom portfolio arithmetic.

The deployed app reads the precomputed long-form return artifact produced here.
It never loads raw prices or rebuilds a backtest.  A custom scenario is a
transparent constant-mix historical simulation on the equity trading calendar.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INVESTABLE_ASSET_RELATIVE_PATH = Path("results/data/investable_asset_returns.csv")
INVESTABLE_ASSET_COLUMNS = (
    "date",
    "ticker",
    "asset_group",
    "sector",
    "daily_return",
)
ASSET_GROUPS = ("Stock", "Crypto")
WEIGHT_TOLERANCE = 1e-8


class CustomPortfolioError(RuntimeError):
    """Raised when the custom-portfolio artifact or scenario is inconsistent."""


@dataclass(frozen=True)
class CustomPortfolioScenario:
    """Historical path, metrics and current holdings for a user-defined mix."""

    history: pd.DataFrame
    holdings: pd.DataFrame
    asset_mix: pd.DataFrame
    initial_value: float
    ending_value: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    maximum_drawdown: float
    periods_per_year: int = 252


def build_investable_asset_returns(
    combined_asset_returns: pd.DataFrame,
    equity_sector_map: pd.DataFrame,
) -> pd.DataFrame:
    """Convert separately calculated, equity-calendar-aligned returns to long form."""
    if "date" not in combined_asset_returns:
        raise ValueError("combined_asset_returns must contain date")
    if not {"ticker", "sector"}.issubset(equity_sector_map.columns):
        raise ValueError("equity_sector_map must contain ticker and sector")
    if combined_asset_returns["date"].duplicated().any():
        raise ValueError("combined_asset_returns must be unique by date")

    sector_map = (
        equity_sector_map[["ticker", "sector"]]
        .drop_duplicates()
        .set_index("ticker")["sector"]
    )
    if sector_map.index.duplicated().any():
        raise ValueError("equity_sector_map must map each ticker once")

    long = combined_asset_returns.melt(
        id_vars="date",
        var_name="ticker",
        value_name="daily_return",
    )
    long["asset_group"] = np.where(
        long["ticker"].isin(sector_map.index), "Stock", "Crypto"
    )
    long["sector"] = long["ticker"].map(sector_map).fillna("Crypto")
    long = long.loc[:, INVESTABLE_ASSET_COLUMNS].sort_values(
        ["ticker", "date"], kind="mergesort"
    )
    return validate_investable_asset_returns(long)


def validate_investable_asset_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and type the app-facing individual-asset return contract."""
    missing = sorted(set(INVESTABLE_ASSET_COLUMNS).difference(frame.columns))
    if missing:
        raise CustomPortfolioError(
            f"investable_asset_returns.csv is missing required columns: {missing}"
        )
    result = frame.loc[:, INVESTABLE_ASSET_COLUMNS].copy()
    try:
        result["date"] = (
            pd.to_datetime(result["date"], errors="raise", utc=True)
            .dt.tz_convert(None)
            .dt.normalize()
        )
    except (TypeError, ValueError) as exc:
        raise CustomPortfolioError(f"invalid asset-return date: {exc}") from exc
    if result[["date", "ticker", "asset_group", "sector"]].isna().any().any():
        raise CustomPortfolioError("asset-return keys and labels must not be missing")
    if result.duplicated(["ticker", "date"]).any():
        raise CustomPortfolioError("asset returns contain duplicate ticker + date keys")
    result["daily_return"] = pd.to_numeric(result["daily_return"], errors="coerce")
    if not np.isfinite(result["daily_return"].to_numpy(dtype=float)).all():
        raise CustomPortfolioError("asset returns contain non-finite daily returns")
    if (result["daily_return"] <= -1).any():
        raise CustomPortfolioError("asset returns contain values at or below -100%")
    if set(result["asset_group"].unique()) != set(ASSET_GROUPS):
        raise CustomPortfolioError("asset returns must contain Stock and Crypto")
    identity_counts = result.groupby("ticker")[["asset_group", "sector"]].nunique()
    if (identity_counts != 1).any().any():
        raise CustomPortfolioError("ticker metadata changes within the asset artifact")
    date_summary = result.groupby("ticker")["date"].agg(["min", "max", "size"])
    if len(date_summary.drop_duplicates()) != 1:
        raise CustomPortfolioError("all investable assets must share one return sample")
    expected = result.sort_values(["ticker", "date"], kind="mergesort").reset_index(
        drop=True
    )
    if not result.reset_index(drop=True).equals(expected):
        raise CustomPortfolioError("asset returns are not sorted by ticker and date")
    return expected


def load_investable_asset_returns(project_root: Path) -> pd.DataFrame:
    """Read and validate the project-relative individual-asset return artifact."""
    path = Path(project_root) / INVESTABLE_ASSET_RELATIVE_PATH
    if not path.is_file():
        raise CustomPortfolioError(
            "Required custom-portfolio artifact is missing: "
            f"{INVESTABLE_ASSET_RELATIVE_PATH.as_posix()}. Rebuild project artifacts."
        )
    try:
        return validate_investable_asset_returns(pd.read_csv(path))
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise CustomPortfolioError(f"Could not read {path.name}: {exc}") from exc


def calculate_custom_portfolio(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float],
    initial_value: float,
    *,
    tolerance: float = WEIGHT_TOLERANCE,
    periods_per_year: int = 252,
) -> CustomPortfolioScenario:
    """Calculate one long-only, fully invested constant-mix historical scenario."""
    returns = validate_investable_asset_returns(asset_returns)
    value = float(initial_value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("initial_value must be finite and positive")
    if not 2 <= len(weights) <= 15:
        raise ValueError("select between 2 and 15 distinct assets")

    allocation = pd.Series(weights, dtype=float)
    if allocation.index.duplicated().any():
        raise ValueError("each selected asset must be unique")
    available = set(returns["ticker"].unique())
    unknown = sorted(set(allocation.index).difference(available))
    if unknown:
        raise ValueError(f"selected assets are unavailable: {unknown}")
    if not np.isfinite(allocation.to_numpy()).all():
        raise ValueError("weights must be finite")
    if (allocation < 0).any() or (allocation > 1).any():
        raise ValueError("weights must remain between zero and one")
    if not np.isclose(allocation.sum(), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("weights must sum to one")

    tickers = list(allocation.index)
    wide = returns.loc[returns["ticker"].isin(tickers)].pivot(
        index="date", columns="ticker", values="daily_return"
    )
    wide = wide.reindex(columns=tickers).sort_index()
    if wide.isna().any().any():
        raise CustomPortfolioError("selected asset returns do not share a complete sample")
    daily = wide.to_numpy(dtype=float) @ allocation.to_numpy(dtype=float)
    if (daily <= -1).any() or not np.isfinite(daily).all():
        raise CustomPortfolioError("calculated portfolio returns are invalid")
    growth = np.cumprod(1.0 + daily)
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], growth)))[1:]
    drawdown = growth / running_peak - 1.0
    history = pd.DataFrame(
        {
            "date": wide.index,
            "daily_return": daily,
            "growth_of_1": growth,
            "portfolio_value": value * growth,
            "drawdown": drawdown,
        }
    )

    observations = len(history)
    annualised_return = float(growth[-1] ** (periods_per_year / observations) - 1.0)
    annualised_volatility = float(np.std(daily, ddof=1) * np.sqrt(periods_per_year))
    annualised_mean_return = float(np.mean(daily) * periods_per_year)
    sharpe_ratio = (
        annualised_mean_return / annualised_volatility
        if annualised_volatility > 0
        else 0.0
    )

    metadata = (
        returns[["ticker", "asset_group", "sector"]]
        .drop_duplicates("ticker")
        .set_index("ticker")
        .reindex(tickers)
    )
    holdings = metadata.reset_index()
    holdings["weight"] = allocation.to_numpy()
    holdings["initial_allocation"] = value * holdings["weight"]
    holdings = holdings.sort_values("weight", ascending=False).reset_index(drop=True)
    asset_mix = (
        holdings.groupby("asset_group", as_index=False, sort=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    return CustomPortfolioScenario(
        history=history,
        holdings=holdings,
        asset_mix=asset_mix,
        initial_value=value,
        ending_value=float(value * growth[-1]),
        annualised_return=annualised_return,
        annualised_volatility=annualised_volatility,
        sharpe_ratio=float(sharpe_ratio),
        maximum_drawdown=float(drawdown.min()),
        periods_per_year=periods_per_year,
    )


__all__ = [
    "ASSET_GROUPS",
    "INVESTABLE_ASSET_COLUMNS",
    "INVESTABLE_ASSET_RELATIVE_PATH",
    "CustomPortfolioError",
    "CustomPortfolioScenario",
    "build_investable_asset_returns",
    "calculate_custom_portfolio",
    "load_investable_asset_returns",
    "validate_investable_asset_returns",
]
