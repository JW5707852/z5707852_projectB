"""A predeclared sector-risk-budget fund with a lagged sentiment satellite.

The strategy is deliberately a distinct, exploratory product rather than a
retuning of the existing equity sentiment tilt.  It uses only data available at
each monthly decision date: a 252-day expanding return history and the
already-lagged tradable sector signal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import portfolios

FUND = "combined_active_sector_allocation"
METHOD = "active_sector_sentiment_risk_budget"
ASSET_FAMILY = "combined"
CORE_EQUITY_WEIGHT = 0.70
SENTIMENT_SATELLITE_WEIGHT = 0.20
CRYPTO_SLEEVE_WEIGHT = 0.10
TOP_SECTOR_COUNT = 2
SECTOR_WEIGHT_CAP = 0.20
STOCK_WEIGHT_CAP = 0.05
CRYPTO_ASSET_WEIGHT_CAP = 0.02
VOLATILITY_LOOKBACK = 252
TRANSACTION_COST_BPS = 0.0
WEIGHT_TOLERANCE = 1e-10


@dataclass(frozen=True)
class SectorAllocationSpec:
    """Predeclared, fully specified sleeves and limits for one fund."""

    fund: str
    method: str
    core_equity_weight: float
    sentiment_satellite_weight: float
    crypto_sleeve_weight: float
    top_sector_count: int
    sector_weight_cap: float
    stock_weight_cap: float
    crypto_asset_weight_cap: float


ACTIVE_SECTOR_SPEC = SectorAllocationSpec(
    fund=FUND,
    method=METHOD,
    core_equity_weight=CORE_EQUITY_WEIGHT,
    sentiment_satellite_weight=SENTIMENT_SATELLITE_WEIGHT,
    crypto_sleeve_weight=CRYPTO_SLEEVE_WEIGHT,
    top_sector_count=TOP_SECTOR_COUNT,
    sector_weight_cap=SECTOR_WEIGHT_CAP,
    stock_weight_cap=STOCK_WEIGHT_CAP,
    crypto_asset_weight_cap=CRYPTO_ASSET_WEIGHT_CAP,
)
GROWTH_FUND = "combined_growth_sector_allocation"
GROWTH_METHOD = "growth_sector_sentiment_risk_budget"
GROWTH_SECTOR_SPEC = SectorAllocationSpec(
    fund=GROWTH_FUND,
    method=GROWTH_METHOD,
    core_equity_weight=0.80,
    sentiment_satellite_weight=0.15,
    crypto_sleeve_weight=0.05,
    top_sector_count=3,
    sector_weight_cap=0.15,
    stock_weight_cap=0.03,
    crypto_asset_weight_cap=0.005,
)
AGGRESSIVE_FUND = "combined_aggressive_sector_allocation"
AGGRESSIVE_METHOD = "aggressive_sector_sentiment_risk_budget"
AGGRESSIVE_SECTOR_SPEC = SectorAllocationSpec(
    fund=AGGRESSIVE_FUND,
    method=AGGRESSIVE_METHOD,
    core_equity_weight=0.50,
    sentiment_satellite_weight=0.30,
    crypto_sleeve_weight=0.20,
    top_sector_count=3,
    sector_weight_cap=0.25,
    stock_weight_cap=0.05,
    crypto_asset_weight_cap=0.02,
)


class ActiveSectorValidationError(RuntimeError):
    """Raised when the active sector strategy cannot be reproduced safely."""


def _normalise_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise", utc=True).dt.tz_convert(None).dt.normalize()


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ActiveSectorValidationError(f"{name} is missing columns: {missing}")


def _capped_proportional_allocation(
    raw: pd.Series,
    *,
    total: float,
    caps: pd.Series,
) -> pd.Series:
    """Allocate ``total`` in proportion to positive raw values under caps."""
    if total < 0.0 or not np.isfinite(total):
        raise ActiveSectorValidationError("allocation total must be finite and non-negative")
    if not raw.index.equals(caps.index):
        raise ActiveSectorValidationError("allocation values and caps must share an index")
    values = pd.to_numeric(raw, errors="coerce").astype(float)
    limits = pd.to_numeric(caps, errors="coerce").astype(float)
    if (
        not np.isfinite(values.to_numpy()).all()
        or not np.isfinite(limits.to_numpy()).all()
        or (values <= 0.0).any()
        or (limits < 0.0).any()
    ):
        raise ActiveSectorValidationError(
            "allocation values must be positive and caps non-negative"
        )
    if float(limits.sum()) + WEIGHT_TOLERANCE < total:
        raise ActiveSectorValidationError("sector caps cannot fund the required allocation")

    result = pd.Series(0.0, index=values.index)
    remaining = values.index.copy()
    remaining_total = float(total)
    while len(remaining):
        share = values.loc[remaining] / float(values.loc[remaining].sum())
        proposal = remaining_total * share
        capacity = limits.loc[remaining] - result.loc[remaining]
        constrained = proposal > capacity + WEIGHT_TOLERANCE
        if not constrained.any():
            result.loc[remaining] += proposal
            break
        constrained_names = proposal.index[constrained]
        result.loc[constrained_names] += capacity.loc[constrained_names]
        remaining_total = float(total - result.sum())
        remaining = remaining.difference(constrained_names)
        if remaining_total < -WEIGHT_TOLERANCE:
            raise ActiveSectorValidationError("capped allocation overfunded the target")

    if not np.isclose(float(result.sum()), total, atol=WEIGHT_TOLERANCE, rtol=0.0):
        raise ActiveSectorValidationError("capped allocation does not conserve its total")
    if (result > limits + WEIGHT_TOLERANCE).any():
        raise ActiveSectorValidationError("capped allocation exceeds a sector limit")
    return result


def _validate_inputs(
    combined_returns: pd.DataFrame,
    sector_index: pd.DataFrame,
    ticker_sectors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    _require_columns(combined_returns, {"date"}, "combined_returns")
    _require_columns(ticker_sectors, {"ticker", "sector"}, "ticker_sectors")
    _require_columns(
        sector_index,
        {"date", "sector", "tradable_sector_zscore", "tradable_signal_source_date"},
        "sector_index",
    )
    panel = combined_returns.copy()
    panel["date"] = _normalise_date(panel["date"])
    if panel["date"].duplicated().any():
        raise ActiveSectorValidationError("combined_returns has duplicate dates")
    panel = panel.sort_values("date").reset_index(drop=True)
    assets = [column for column in panel.columns if column != "date"]
    if not assets:
        raise ActiveSectorValidationError("combined_returns has no assets")
    panel[assets] = panel[assets].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(panel[assets].to_numpy(dtype=float)).all():
        raise ActiveSectorValidationError("combined returns must be finite")

    mapping = ticker_sectors.copy()
    if mapping.duplicated("ticker").any() or mapping[["ticker", "sector"]].isna().any().any():
        raise ActiveSectorValidationError("ticker sectors must be one-to-one and complete")
    mapping = mapping.sort_values("ticker").reset_index(drop=True)
    equity_tickers = mapping["ticker"].tolist()
    crypto_tickers = sorted(ticker for ticker in assets if ticker.endswith("-USD"))
    if set(equity_tickers).intersection(crypto_tickers):
        raise ActiveSectorValidationError("equity and crypto universes overlap")
    if set(equity_tickers).union(crypto_tickers) != set(assets):
        raise ActiveSectorValidationError(
            "combined universe does not match the supplied sector map"
        )
    counts = mapping.groupby("sector")["ticker"].size()
    if len(counts) != 10 or not counts.eq(5).all():
        raise ActiveSectorValidationError("the strategy requires ten sectors with five stocks each")
    if len(crypto_tickers) != 10:
        raise ActiveSectorValidationError("the strategy requires ten cryptocurrencies")

    signals = sector_index.copy()
    signals["date"] = _normalise_date(signals["date"])
    signals["tradable_signal_source_date"] = pd.to_datetime(
        signals["tradable_signal_source_date"], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()
    if signals.duplicated(["date", "sector"]).any():
        raise ActiveSectorValidationError("sector index has duplicate date-sector keys")
    if set(signals["sector"].unique()) != set(counts.index):
        raise ActiveSectorValidationError("sector index does not match the equity sector map")
    return panel, signals, mapping, equity_tickers, crypto_tickers


def build_active_sector_allocation(
    combined_returns: pd.DataFrame,
    sector_index: pd.DataFrame,
    ticker_sectors: pd.DataFrame,
    *,
    initial_window: int = VOLATILITY_LOOKBACK,
    strategy: SectorAllocationSpec = ACTIVE_SECTOR_SPEC,
) -> portfolios.BacktestResult:
    """Build one fixed sector allocation from only prior data."""
    if initial_window != VOLATILITY_LOOKBACK:
        raise ActiveSectorValidationError(
            f"volatility lookback is locked at {VOLATILITY_LOOKBACK} trading days"
        )
    panel, signals, mapping, equity_tickers, crypto_tickers = _validate_inputs(
        combined_returns,
        sector_index,
        ticker_sectors,
    )
    schedule = portfolios.monthly_rebalance_schedule(
        panel["date"], initial_window=initial_window
    )
    sectors = sorted(mapping["sector"].unique())
    sector_tickers = {
        sector: mapping.loc[mapping["sector"].eq(sector), "ticker"].tolist()
        for sector in sectors
    }
    positions = pd.Series(np.arange(len(panel)), index=panel["date"])
    return_records: list[dict[str, object]] = []
    weight_records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []

    for rebalance_number, schedule_row in schedule.iterrows():
        decision_date = schedule_row["decision_date"]
        decision_position = int(positions.loc[decision_date])
        training_start_position = decision_position - VOLATILITY_LOOKBACK + 1
        training = panel.loc[
            training_start_position:decision_position,
            [*equity_tickers, *crypto_tickers],
        ]
        if len(training) != VOLATILITY_LOOKBACK:
            raise ActiveSectorValidationError("training window differs from the locked lookback")
        sector_returns = pd.DataFrame(
            {
                sector: training[sector_tickers[sector]].mean(axis=1)
                for sector in sectors
            }
        )
        sector_volatility = sector_returns.std(ddof=1)
        if not np.isfinite(sector_volatility.to_numpy()).all() or (sector_volatility <= 0).any():
            raise ActiveSectorValidationError("sector volatility must be positive and finite")
        inverse_volatility = 1.0 / sector_volatility

        decision_signals = signals.loc[
            signals["date"].eq(decision_date),
            ["sector", "tradable_sector_zscore", "tradable_signal_source_date"],
        ].set_index("sector").reindex(sectors)
        scores = pd.to_numeric(decision_signals["tradable_sector_zscore"], errors="coerce")
        sources = decision_signals["tradable_signal_source_date"]
        signal_was_missing = scores.isna()
        for sector in scores.index[signal_was_missing]:
            prior = signals.loc[
                signals["sector"].eq(sector)
                & signals["date"].lt(decision_date)
                & signals["tradable_sector_zscore"].notna()
            ].sort_values("date")
            if prior.empty:
                raise ActiveSectorValidationError(
                    "a missing active-sector signal has no prior usable value"
                )
            latest = prior.iloc[-1]
            scores.loc[sector] = float(latest["tradable_sector_zscore"])
            sources.loc[sector] = latest["tradable_signal_source_date"]
        if sources.isna().any() or not (sources < decision_date).all():
            raise ActiveSectorValidationError(
                "each active-sector decision requires a strictly lagged signal for every sector"
            )
        ranked = scores.sort_values(ascending=False, kind="mergesort")
        selected_sectors = tuple(ranked.index[: strategy.top_sector_count])
        satellite = pd.Series(0.0, index=sectors)
        satellite.loc[list(selected_sectors)] = (
            strategy.sentiment_satellite_weight / strategy.top_sector_count
        )
        capacities = pd.Series(strategy.sector_weight_cap, index=sectors) - satellite
        core = _capped_proportional_allocation(
            inverse_volatility,
            total=strategy.core_equity_weight,
            caps=capacities,
        )
        sector_weights = core + satellite
        if not np.isclose(
            float(sector_weights.sum()),
            strategy.core_equity_weight + strategy.sentiment_satellite_weight,
            atol=WEIGHT_TOLERANCE,
            rtol=0.0,
        ):
            raise ActiveSectorValidationError("equity sleeves do not conserve their allocation")
        if (sector_weights > strategy.sector_weight_cap + WEIGHT_TOLERANCE).any():
            raise ActiveSectorValidationError("sector allocation exceeds the fixed cap")

        target = pd.Series(0.0, index=sorted([*equity_tickers, *crypto_tickers]))
        for sector in sectors:
            target.loc[sector_tickers[sector]] = sector_weights.loc[sector] / 5.0
        target.loc[crypto_tickers] = strategy.crypto_sleeve_weight / len(crypto_tickers)
        if not np.isclose(float(target.sum()), 1.0, atol=WEIGHT_TOLERANCE, rtol=0.0):
            raise ActiveSectorValidationError("target weights do not sum to one")
        if (target < -WEIGHT_TOLERANCE).any() or (
            target > strategy.stock_weight_cap + WEIGHT_TOLERANCE
        ).any():
            raise ActiveSectorValidationError("target weights exceed the stock cap")
        if (
            target.loc[crypto_tickers]
            > strategy.crypto_asset_weight_cap + WEIGHT_TOLERANCE
        ).any():
            raise ActiveSectorValidationError("a crypto target exceeds its cap")

        first_holding_position = decision_position + 1
        if rebalance_number + 1 < len(schedule):
            next_decision = schedule.loc[rebalance_number + 1, "decision_date"]
            holding_end_position = int(positions.loc[next_decision])
        else:
            holding_end_position = len(panel) - 1
        holding = panel.loc[first_holding_position:holding_end_position]
        if holding.empty:
            raise ActiveSectorValidationError("a rebalance has no holding-period returns")
        portfolio_returns = holding[target.index].to_numpy(dtype=float) @ target.to_numpy()
        for date, daily_return in zip(holding["date"], portfolio_returns, strict=True):
            return_records.append(
                {
                    "date": date,
                    "fund": strategy.fund,
                    "method": strategy.method,
                    "asset_family": ASSET_FAMILY,
                    "decision_date": decision_date,
                    "daily_return": float(daily_return),
                }
            )

        covariance = training[target.index].cov(ddof=1).to_numpy(dtype=float)
        metadata = {
            "fund": strategy.fund,
            "method": strategy.method,
            "asset_family": ASSET_FAMILY,
            "decision_date": decision_date,
            "training_start_date": panel.loc[training_start_position, "date"],
            "training_end_date": decision_date,
            "first_holding_date": schedule_row["first_holding_date"],
            "window_size": VOLATILITY_LOOKBACK,
            "solver_success": True,
            "solver_status": "not_required",
            "solver_message": (
                "fixed sector inverse-volatility core with lagged sentiment satellite"
            ),
            "objective_value": float(target.to_numpy() @ covariance @ target.to_numpy()),
            "covariance_scale": 1.0,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "active_core_weight": strategy.core_equity_weight,
            "active_satellite_weight": strategy.sentiment_satellite_weight,
            "active_crypto_sleeve_weight": strategy.crypto_sleeve_weight,
            "active_top_sector_count": strategy.top_sector_count,
            "active_stock_weight_cap": strategy.stock_weight_cap,
            "active_sector_weight_cap": strategy.sector_weight_cap,
            "active_crypto_asset_weight_cap": strategy.crypto_asset_weight_cap,
            "active_volatility_lookback": VOLATILITY_LOOKBACK,
        }
        sector_rank = ranked.index.to_series().reset_index(drop=True)
        ranks = {sector: position + 1 for position, sector in enumerate(sector_rank)}
        sector_lookup = mapping.set_index("ticker")["sector"].to_dict()
        for ticker, value in target.items():
            sector = sector_lookup.get(ticker, "Crypto")
            is_equity = ticker in sector_lookup
            weight_records.append(
                {
                    **metadata,
                    "ticker": ticker,
                    "target_weight": float(value),
                    "base_target_weight": (
                        float(core.loc[sector] / 5.0) if is_equity else float(value)
                    ),
                    "sector": sector,
                    "tradable_sector_zscore": float(scores.loc[sector]) if is_equity else np.nan,
                    "applied_sector_zscore": float(scores.loc[sector]) if is_equity else np.nan,
                    "tradable_signal_source_date": sources.loc[sector] if is_equity else pd.NaT,
                    "signal_was_missing": bool(signal_was_missing.loc[sector])
                    if is_equity
                    else False,
                    "active_sector_inverse_volatility": (
                        float(inverse_volatility.loc[sector]) if is_equity else np.nan
                    ),
                    "active_sector_rank": float(ranks[sector]) if is_equity else np.nan,
                    "active_selected_sector": (
                        bool(sector in selected_sectors) if is_equity else False
                    ),
                    "active_sector_target_weight": (
                        float(sector_weights.loc[sector])
                        if is_equity
                        else strategy.crypto_sleeve_weight
                    ),
                }
            )
        audit_records.append(
            {
                **metadata,
                "target_weights": json.dumps(target.to_dict(), sort_keys=True),
                "selected_sectors": json.dumps(list(selected_sectors)),
                "sector_scores": json.dumps(scores.to_dict(), sort_keys=True),
                "sector_target_weights": json.dumps(sector_weights.to_dict(), sort_keys=True),
            }
        )

    fund_returns = pd.DataFrame.from_records(return_records).sort_values("date")
    fund_returns["growth_of_1"] = (1.0 + fund_returns["daily_return"]).cumprod()
    fund_weights = pd.DataFrame.from_records(weight_records).sort_values(
        ["decision_date", "ticker"]
    )
    audit = pd.DataFrame.from_records(audit_records).sort_values("decision_date")
    return portfolios.BacktestResult(
        fund_returns.reset_index(drop=True),
        fund_weights.reset_index(drop=True),
        audit.reset_index(drop=True),
    )


def active_sector_exposure(fund_weights: pd.DataFrame) -> pd.DataFrame:
    """Summarise logged sector and crypto sleeve weights by decision date."""
    _require_columns(
        fund_weights,
        {"fund", "decision_date", "sector", "target_weight"},
        "fund_weights",
    )
    selected = fund_weights.loc[fund_weights["fund"].eq(FUND)].copy()
    if selected.empty:
        raise ActiveSectorValidationError("active-sector weights are absent")
    exposure = (
        selected.groupby(["decision_date", "sector"], as_index=False)["target_weight"]
        .sum()
        .sort_values(["decision_date", "sector"])
        .reset_index(drop=True)
    )
    sums = exposure.groupby("decision_date")["target_weight"].sum()
    if not np.allclose(sums, 1.0, atol=WEIGHT_TOLERANCE, rtol=0.0):
        raise ActiveSectorValidationError("active-sector exposures do not sum to one")
    return exposure


def build_growth_sector_allocation(
    combined_returns: pd.DataFrame,
    sector_index: pd.DataFrame,
    ticker_sectors: pd.DataFrame,
    *,
    initial_window: int = VOLATILITY_LOOKBACK,
) -> portfolios.BacktestResult:
    """Build the fixed 80/15/5 balanced-growth allocation without look-ahead."""
    return build_active_sector_allocation(
        combined_returns,
        sector_index,
        ticker_sectors,
        initial_window=initial_window,
        strategy=GROWTH_SECTOR_SPEC,
    )


def build_aggressive_sector_allocation(
    combined_returns: pd.DataFrame,
    sector_index: pd.DataFrame,
    ticker_sectors: pd.DataFrame,
    *,
    initial_window: int = VOLATILITY_LOOKBACK,
) -> portfolios.BacktestResult:
    """Build the fixed 50/30/20 aggressive allocation without look-ahead."""
    return build_active_sector_allocation(
        combined_returns,
        sector_index,
        ticker_sectors,
        initial_window=initial_window,
        strategy=AGGRESSIVE_SECTOR_SPEC,
    )


def sector_allocation_exposure(
    fund_weights: pd.DataFrame,
    *,
    fund: str,
) -> pd.DataFrame:
    """Summarise one sector-allocation fund's logged sleeve exposures."""
    _require_columns(
        fund_weights,
        {"fund", "decision_date", "sector", "target_weight"},
        "fund_weights",
    )
    selected = fund_weights.loc[fund_weights["fund"].eq(fund)].copy()
    if selected.empty:
        raise ActiveSectorValidationError("sector-allocation weights are absent")
    exposure = (
        selected.groupby(["decision_date", "sector"], as_index=False)["target_weight"]
        .sum()
        .sort_values(["decision_date", "sector"])
        .reset_index(drop=True)
    )
    sums = exposure.groupby("decision_date")["target_weight"].sum()
    if not np.allclose(sums, 1.0, atol=WEIGHT_TOLERANCE, rtol=0.0):
        raise ActiveSectorValidationError("sector-allocation exposures do not sum to one")
    return exposure


__all__ = [
    "ACTIVE_SECTOR_SPEC",
    "AGGRESSIVE_FUND",
    "AGGRESSIVE_METHOD",
    "AGGRESSIVE_SECTOR_SPEC",
    "ASSET_FAMILY",
    "CORE_EQUITY_WEIGHT",
    "CRYPTO_ASSET_WEIGHT_CAP",
    "CRYPTO_SLEEVE_WEIGHT",
    "FUND",
    "GROWTH_FUND",
    "GROWTH_METHOD",
    "GROWTH_SECTOR_SPEC",
    "METHOD",
    "SECTOR_WEIGHT_CAP",
    "SENTIMENT_SATELLITE_WEIGHT",
    "STOCK_WEIGHT_CAP",
    "TOP_SECTOR_COUNT",
    "TRANSACTION_COST_BPS",
    "VOLATILITY_LOOKBACK",
    "ActiveSectorValidationError",
    "SectorAllocationSpec",
    "active_sector_exposure",
    "build_active_sector_allocation",
    "build_aggressive_sector_allocation",
    "build_growth_sector_allocation",
    "sector_allocation_exposure",
]
