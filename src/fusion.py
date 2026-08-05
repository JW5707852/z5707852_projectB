"""Look-ahead-safe fusion of sector sentiment with equity fund weights.

The locked fusion rule consumes the precomputed, lagged sector signal produced
by :mod:`src.sentiment`.  It never scores text, changes the signal parameters,
or applies sentiment to crypto assets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import portfolios, sentiment

BASE_FUND = "equity_equal_weight"
TILTED_FUND = "equity_sentiment_tilt"
TILTED_METHOD = "sentiment_tilt"
EXPLORATORY_FUND = "equity_sentiment_21d_coverage_tilt"
EXPLORATORY_METHOD = "exploratory_sentiment_21d_coverage_tilt"
EXPLORATORY_METHOD_LABEL = sentiment.EXPLORATORY_METHOD_LABEL
EXPLORATORY_SIGNAL_WINDOW = sentiment.EXPLORATORY_SIGNAL_WINDOW
TILT_STRENGTH = 0.10
MIN_HISTORY = 60
ZSCORE_CLIP = 2.0
SIGNAL_LAG = 1
TRANSACTION_COST_BPS = 0.0
WEIGHT_TOLERANCE = 1e-10


class FusionValidationError(RuntimeError):
    """Raised when the fusion inputs or matched comparison are inconsistent."""


@dataclass(frozen=True)
class FusionEvidence:
    """Matched performance, turnover, and sector-exposure evidence."""

    comparison: pd.DataFrame
    sector_exposure: pd.DataFrame
    turnover: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FusionValidationError(f"{name} is missing columns: {missing}")


def _normalise_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise", utc=True).dt.tz_convert(None).dt.normalize()


def _validate_locked_signal(sector_signal: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "sector",
        "tradable_sector_zscore",
        "tradable_signal_source_date",
        "signal_prior_observations",
        "min_history_observations",
        "zscore_clip_bound",
        "signal_lag_trading_days",
    }
    _require_columns(sector_signal, required, "sector_signal")
    signal = sector_signal.copy()
    signal["date"] = _normalise_date(signal["date"])
    signal["tradable_signal_source_date"] = pd.to_datetime(
        signal["tradable_signal_source_date"], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()
    signal["tradable_sector_zscore"] = pd.to_numeric(
        signal["tradable_sector_zscore"], errors="coerce"
    )
    signal["signal_prior_observations"] = pd.to_numeric(
        signal["signal_prior_observations"], errors="coerce"
    )
    if signal.duplicated(["date", "sector"]).any():
        raise FusionValidationError("sector_signal must be unique by date and sector")
    if not signal["min_history_observations"].eq(MIN_HISTORY).all():
        raise FusionValidationError("sector signal does not use the locked 60-observation history")
    if not signal["zscore_clip_bound"].eq(ZSCORE_CLIP).all():
        raise FusionValidationError("sector signal does not use the locked [-2, 2] clip")
    if not signal["signal_lag_trading_days"].eq(SIGNAL_LAG).all():
        raise FusionValidationError("sector signal does not use the locked one-day lag")

    available = signal["tradable_sector_zscore"].notna()
    values = signal.loc[available, "tradable_sector_zscore"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (np.abs(values) > ZSCORE_CLIP + 1e-12).any():
        raise FusionValidationError("tradable sector z-scores must be finite and clipped")
    if signal.loc[available, "tradable_signal_source_date"].isna().any():
        raise FusionValidationError("available signals must retain their source date")
    if not (
        signal.loc[available, "tradable_signal_source_date"]
        < signal.loc[available, "date"]
    ).all():
        raise FusionValidationError("sentiment signal is not available before use")
    if (signal.loc[available, "signal_prior_observations"] < MIN_HISTORY).any():
        raise FusionValidationError("tradable signal has insufficient prior history")
    return signal.sort_values(["date", "sector"]).reset_index(drop=True)


def _validate_ticker_sectors(ticker_sectors: pd.DataFrame) -> pd.DataFrame:
    _require_columns(ticker_sectors, {"ticker", "sector"}, "ticker_sectors")
    mapping = ticker_sectors[["ticker", "sector"]].drop_duplicates().copy()
    if mapping[["ticker", "sector"]].isna().any().any():
        raise FusionValidationError("ticker sector mapping contains missing values")
    if (mapping.groupby("ticker")["sector"].nunique() != 1).any():
        raise FusionValidationError("a ticker maps to more than one sector")
    return mapping.sort_values("ticker").reset_index(drop=True)


def apply_sentiment_tilt(
    base_backtest: portfolios.BacktestResult,
    equity_returns: pd.DataFrame,
    sector_signal: pd.DataFrame,
    ticker_sectors: pd.DataFrame,
    *,
    tilt_strength: float = TILT_STRENGTH,
) -> portfolios.BacktestResult:
    """Apply the fixed monthly sector tilt to ``equity_equal_weight``.

    The base fund's decision dates and holding rows are authoritative.  For a
    sector signal ``z_s`` available on a decision date, the unnormalised stock
    weight is ``w_i * (1 + 0.10 * z_s)``.  Missing signals receive multiplier
    one, and all weights are renormalised before being applied to the base
    fund's exact next holding period.
    """
    if tilt_strength != TILT_STRENGTH:
        raise FusionValidationError("the locked tilt strength is 0.10 and may not be retuned")
    signal = _validate_locked_signal(sector_signal)
    mapping = _validate_ticker_sectors(ticker_sectors)

    base_returns = base_backtest.fund_returns.loc[
        base_backtest.fund_returns["fund"].eq(BASE_FUND)
    ].copy()
    base_weights = base_backtest.fund_weights.loc[
        base_backtest.fund_weights["fund"].eq(BASE_FUND)
    ].copy()
    base_audit = base_backtest.rebalance_audit.loc[
        base_backtest.rebalance_audit["fund"].eq(BASE_FUND)
    ].copy()
    if base_returns.empty or base_weights.empty or base_audit.empty:
        raise FusionValidationError(f"base backtest must contain {BASE_FUND}")
    if not base_returns["asset_family"].eq("equity").all() or not base_weights[
        "asset_family"
    ].eq("equity").all():
        raise FusionValidationError("sentiment fusion may be applied only to equities")
    if not base_returns["method"].eq("equal_weight").all() or not base_weights[
        "method"
    ].eq("equal_weight").all():
        raise FusionValidationError("fusion baseline must use equal-weight holdings")

    for frame, columns in (
        (base_returns, ["date", "decision_date"]),
        (base_weights, ["decision_date"]),
        (
            base_audit,
            [
                "decision_date",
                "training_start_date",
                "training_end_date",
                "first_holding_date",
            ],
        ),
    ):
        for column in columns:
            frame[column] = _normalise_date(frame[column])
    if base_returns.duplicated(["fund", "date"]).any():
        raise FusionValidationError("base fund returns contain duplicate dates")
    if base_weights.duplicated(["fund", "decision_date", "ticker"]).any():
        raise FusionValidationError("base target weights contain duplicate keys")

    base_tickers = sorted(base_weights["ticker"].unique())
    mapped_tickers = set(mapping["ticker"])
    if not set(base_tickers).issubset(mapped_tickers):
        missing = sorted(set(base_tickers).difference(mapped_tickers))
        raise FusionValidationError(f"equity sector mapping is missing tickers: {missing}")
    mapping = mapping.loc[mapping["ticker"].isin(base_tickers)].copy()
    if len(mapping) != len(base_tickers):
        raise FusionValidationError("the fusion universe is not one-to-one with equities")

    panel = equity_returns.copy()
    _require_columns(panel, {"date", *base_tickers}, "equity_returns")
    panel["date"] = _normalise_date(panel["date"])
    if panel["date"].duplicated().any():
        raise FusionValidationError("equity return dates must be unique")
    panel = panel.sort_values("date").set_index("date")
    panel[base_tickers] = panel[base_tickers].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(panel[base_tickers].to_numpy(dtype=float)).all():
        raise FusionValidationError("equity returns must be complete and finite")

    return_records: list[dict[str, object]] = []
    weight_records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []
    signal_columns = [
        "sector",
        "tradable_sector_zscore",
        "tradable_signal_source_date",
        "signal_prior_observations",
    ]

    for decision_date, base_group in base_weights.groupby("decision_date", sort=True):
        base_group = base_group.sort_values("ticker").copy()
        if set(base_group["ticker"]) != set(base_tickers):
            raise FusionValidationError("base equity universe changes across rebalances")
        base_values = base_group["target_weight"].to_numpy(dtype=float)
        if not np.isfinite(base_values).all() or (base_values < -WEIGHT_TOLERANCE).any():
            raise FusionValidationError("base weights are invalid")
        if abs(float(base_values.sum()) - 1.0) > WEIGHT_TOLERANCE:
            raise FusionValidationError("base weights do not sum to one")

        decision_signal = signal.loc[signal["date"].eq(decision_date), signal_columns]
        tilted = base_group.merge(mapping, on="ticker", how="left", validate="many_to_one")
        tilted = tilted.merge(
            decision_signal,
            on="sector",
            how="left",
            validate="many_to_one",
        )
        tilted["base_target_weight"] = tilted["target_weight"]
        tilted["signal_was_missing"] = tilted["tradable_sector_zscore"].isna()
        tilted["applied_sector_zscore"] = tilted["tradable_sector_zscore"].fillna(0.0)
        tilted["tilt_multiplier"] = np.where(
            tilted["signal_was_missing"],
            1.0,
            1.0 + tilt_strength * tilted["tradable_sector_zscore"],
        )
        if not tilted["tilt_multiplier"].between(0.8, 1.2, inclusive="both").all():
            raise FusionValidationError("tilt multiplier falls outside [0.8, 1.2]")
        unnormalised = tilted["base_target_weight"] * tilted["tilt_multiplier"]
        normaliser = float(unnormalised.sum())
        if not np.isfinite(normaliser) or normaliser <= 0:
            raise FusionValidationError("tilted weights cannot be normalised")
        tilted["target_weight"] = unnormalised / normaliser
        target = tilted.set_index("ticker")["target_weight"].reindex(base_tickers)
        if not np.isfinite(target).all() or (target < -WEIGHT_TOLERANCE).any():
            raise FusionValidationError("tilted weights violate long-only constraints")
        if abs(float(target.sum()) - 1.0) > WEIGHT_TOLERANCE:
            raise FusionValidationError("tilted weights do not sum to one")

        holding = base_returns.loc[base_returns["decision_date"].eq(decision_date)].copy()
        if holding.empty:
            raise FusionValidationError("base decision has no holding-period returns")
        holding = holding.sort_values("date")
        asset_slice = panel.loc[holding["date"], base_tickers]
        base_vector = base_group.set_index("ticker")["target_weight"].reindex(base_tickers)
        reproduced_base = asset_slice.to_numpy(dtype=float) @ base_vector.to_numpy(dtype=float)
        if not np.allclose(
            reproduced_base,
            holding["daily_return"].to_numpy(dtype=float),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise FusionValidationError("base logged weights do not reproduce base returns")
        tilted_returns = asset_slice.to_numpy(dtype=float) @ target.to_numpy(dtype=float)
        for date, daily_return in zip(holding["date"], tilted_returns, strict=True):
            return_records.append(
                {
                    "date": date,
                    "fund": TILTED_FUND,
                    "method": TILTED_METHOD,
                    "asset_family": "equity",
                    "decision_date": decision_date,
                    "daily_return": float(daily_return),
                }
            )

        audit = base_audit.loc[base_audit["decision_date"].eq(decision_date)]
        if len(audit) != 1:
            raise FusionValidationError("base rebalance audit is not unique")
        audit_row = audit.iloc[0]
        training = panel.loc[:decision_date, base_tickers]
        covariance = training.cov(ddof=1).to_numpy(dtype=float)
        objective = float(target.to_numpy() @ covariance @ target.to_numpy())
        metadata = {
            "fund": TILTED_FUND,
            "method": TILTED_METHOD,
            "asset_family": "equity",
            "decision_date": decision_date,
            "training_start_date": audit_row["training_start_date"],
            "training_end_date": audit_row["training_end_date"],
            "first_holding_date": audit_row["first_holding_date"],
            "window_size": int(audit_row["window_size"]),
            "solver_success": True,
            "solver_status": "not_required",
            "solver_message": "fixed equal-weight sector sentiment tilt",
            "objective_value": objective,
            "covariance_scale": 1.0,
            "tilt_strength": tilt_strength,
            "min_history_observations": MIN_HISTORY,
            "zscore_clip_bound": ZSCORE_CLIP,
            "signal_lag_trading_days": SIGNAL_LAG,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
        }
        for row in tilted.itertuples(index=False):
            weight_records.append(
                {
                    **metadata,
                    "ticker": row.ticker,
                    "target_weight": float(row.target_weight),
                    "base_target_weight": float(row.base_target_weight),
                    "sector": row.sector,
                    "tradable_sector_zscore": (
                        float(row.tradable_sector_zscore)
                        if pd.notna(row.tradable_sector_zscore)
                        else np.nan
                    ),
                    "applied_sector_zscore": float(row.applied_sector_zscore),
                    "tradable_signal_source_date": row.tradable_signal_source_date,
                    "signal_prior_observations": row.signal_prior_observations,
                    "signal_was_missing": bool(row.signal_was_missing),
                    "tilt_multiplier": float(row.tilt_multiplier),
                }
            )
        sector_rows = tilted.drop_duplicates("sector").sort_values("sector")
        audit_records.append(
            {
                **metadata,
                "target_weights": json.dumps(
                    {ticker: float(value) for ticker, value in target.items()},
                    sort_keys=True,
                ),
                "available_sector_signals": json.dumps(
                    {
                        row.sector: (
                            float(row.tradable_sector_zscore)
                            if pd.notna(row.tradable_sector_zscore)
                            else None
                        )
                        for row in sector_rows.itertuples(index=False)
                    },
                    sort_keys=True,
                ),
                "missing_sector_signal_count": int(
                    sector_rows["signal_was_missing"].sum()
                ),
            }
        )

    fund_returns = pd.DataFrame.from_records(return_records).sort_values("date")
    fund_returns["growth_of_1"] = (1 + fund_returns["daily_return"]).cumprod()
    fund_weights = pd.DataFrame.from_records(weight_records).sort_values(
        ["decision_date", "ticker"]
    )
    rebalance_audit = pd.DataFrame.from_records(audit_records).sort_values(
        "decision_date"
    )
    if tuple(fund_returns["date"]) != tuple(base_returns.sort_values("date")["date"]):
        raise FusionValidationError("base and tilted funds do not share the same dates")
    if tuple(fund_returns["decision_date"]) != tuple(
        base_returns.sort_values("date")["decision_date"]
    ):
        raise FusionValidationError("base and tilted funds do not share rebalances")
    return portfolios.BacktestResult(
        fund_returns.reset_index(drop=True),
        fund_weights.reset_index(drop=True),
        rebalance_audit.reset_index(drop=True),
    )


def _validate_exploratory_signal(sector_signal: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "sector",
        "signal_window_trading_days",
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
        "min_history_observations",
        "zscore_clip_bound",
        "zscore_std_ddof",
        "exploratory_method_label",
    }
    _require_columns(sector_signal, required, "exploratory sector signal")
    signal = sector_signal.copy()
    signal["date"] = _normalise_date(signal["date"])
    for column in (
        "signal_window_start_date",
        "signal_window_end_date",
        "latest_raw_news_date_used",
    ):
        signal[column] = pd.to_datetime(
            signal[column], errors="coerce", utc=True
        ).dt.tz_convert(None).dt.normalize()
    numeric = [
        "trailing_coverage_weighted_sentiment",
        "effective_coverage",
        "expanding_prior_mean",
        "expanding_prior_std",
        "expanding_prior_observations",
        "raw_trailing_zscore",
        "clipped_trailing_zscore",
        "coverage_adjusted_zscore",
    ]
    signal[numeric] = signal[numeric].apply(pd.to_numeric, errors="coerce")
    if signal.duplicated(["date", "sector"]).any():
        raise FusionValidationError("exploratory signal must be unique by date and sector")
    if not signal["signal_window_trading_days"].eq(EXPLORATORY_SIGNAL_WINDOW).all():
        raise FusionValidationError("exploratory signal window must be fixed at 21 days")
    if not signal["min_history_observations"].eq(MIN_HISTORY).all():
        raise FusionValidationError("exploratory signal must use 60 prior observations")
    if not signal["zscore_clip_bound"].eq(ZSCORE_CLIP).all():
        raise FusionValidationError("exploratory signal must use the locked [-2, 2] clip")
    if not signal["zscore_std_ddof"].eq(1).all():
        raise FusionValidationError("exploratory signal must use sample standard deviation")
    if not signal["exploratory_method_label"].eq(EXPLORATORY_METHOD_LABEL).all():
        raise FusionValidationError("exploratory method label is inconsistent")
    if not signal["effective_coverage"].between(0.0, 1.0).all():
        raise FusionValidationError("effective coverage must remain in [0, 1]")
    dated_windows = signal.dropna(subset=["signal_window_end_date"])
    if not (dated_windows["signal_window_end_date"] < dated_windows["date"]).all():
        raise FusionValidationError("exploratory window includes current or future data")
    news_sources = signal.dropna(subset=["latest_raw_news_date_used"])
    if not (news_sources["latest_raw_news_date_used"] < news_sources["date"]).all():
        raise FusionValidationError("exploratory signal uses current or future news")
    available = signal["coverage_adjusted_zscore"].notna()
    values = signal.loc[available, "coverage_adjusted_zscore"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (np.abs(values) > ZSCORE_CLIP + 1e-12).any():
        raise FusionValidationError("coverage-adjusted z-scores are invalid")
    if (signal.loc[available, "expanding_prior_observations"] < MIN_HISTORY).any():
        raise FusionValidationError("exploratory signal has insufficient history")
    return signal.sort_values(["date", "sector"]).reset_index(drop=True)


def apply_exploratory_coverage_tilt(
    base_backtest: portfolios.BacktestResult,
    equity_returns: pd.DataFrame,
    sector_signal: pd.DataFrame,
    ticker_sectors: pd.DataFrame,
    *,
    tilt_strength: float = TILT_STRENGTH,
) -> portfolios.BacktestResult:
    """Apply the predeclared 21-day coverage-adjusted signal to equity only."""
    if tilt_strength != TILT_STRENGTH:
        raise FusionValidationError("exploratory tilt strength is locked at 0.10")
    signal = _validate_exploratory_signal(sector_signal)
    mapping = _validate_ticker_sectors(ticker_sectors)

    base_returns = base_backtest.fund_returns.loc[
        base_backtest.fund_returns["fund"].eq(BASE_FUND)
    ].copy()
    base_weights = base_backtest.fund_weights.loc[
        base_backtest.fund_weights["fund"].eq(BASE_FUND)
    ].copy()
    base_audit = base_backtest.rebalance_audit.loc[
        base_backtest.rebalance_audit["fund"].eq(BASE_FUND)
    ].copy()
    if base_returns.empty or base_weights.empty or base_audit.empty:
        raise FusionValidationError(f"base backtest must contain {BASE_FUND}")
    for frame, columns in (
        (base_returns, ["date", "decision_date"]),
        (base_weights, ["decision_date"]),
        (
            base_audit,
            [
                "decision_date",
                "training_start_date",
                "training_end_date",
                "first_holding_date",
            ],
        ),
    ):
        for column in columns:
            frame[column] = _normalise_date(frame[column])
    if not base_returns["asset_family"].eq("equity").all():
        raise FusionValidationError("exploratory fusion may be applied only to equities")

    tickers = sorted(base_weights["ticker"].unique())
    mapping = mapping.loc[mapping["ticker"].isin(tickers)].copy()
    if len(mapping) != len(tickers) or set(mapping["ticker"]) != set(tickers):
        raise FusionValidationError("exploratory fusion requires every equity sector")
    panel = equity_returns.copy()
    _require_columns(panel, {"date", *tickers}, "equity_returns")
    panel["date"] = _normalise_date(panel["date"])
    if panel["date"].duplicated().any():
        raise FusionValidationError("equity return dates must be unique")
    panel = panel.sort_values("date").set_index("date")
    panel[tickers] = panel[tickers].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(panel[tickers].to_numpy(dtype=float)).all():
        raise FusionValidationError("equity returns must be complete and finite")

    signal_fields = [
        "sector",
        "signal_window_trading_days",
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
        "exploratory_method_label",
    ]
    return_records: list[dict[str, object]] = []
    weight_records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []

    for decision_date, base_group in base_weights.groupby("decision_date", sort=True):
        base_group = base_group.sort_values("ticker").copy()
        if set(base_group["ticker"]) != set(tickers):
            raise FusionValidationError("base equity universe changes across rebalances")
        base_values = base_group["target_weight"].to_numpy(dtype=float)
        if not np.isfinite(base_values).all() or (base_values < -WEIGHT_TOLERANCE).any():
            raise FusionValidationError("base weights are invalid")
        if abs(float(base_values.sum()) - 1.0) > WEIGHT_TOLERANCE:
            raise FusionValidationError("base weights do not sum to one")

        decision_signal = signal.loc[signal["date"].eq(decision_date), signal_fields]
        tilted = base_group.merge(mapping, on="ticker", how="left", validate="many_to_one")
        tilted = tilted.merge(
            decision_signal, on="sector", how="left", validate="many_to_one"
        )
        if tilted["exploratory_method_label"].isna().all():
            raise FusionValidationError("decision date is missing from exploratory panel")
        tilted["base_target_weight"] = tilted["target_weight"]
        tilted["signal_was_missing"] = tilted["coverage_adjusted_zscore"].isna()
        tilted["tilt_multiplier"] = np.where(
            tilted["signal_was_missing"],
            1.0,
            1.0 + tilt_strength * tilted["coverage_adjusted_zscore"],
        )
        if not tilted["tilt_multiplier"].between(0.8, 1.2, inclusive="both").all():
            raise FusionValidationError("exploratory multiplier falls outside [0.8, 1.2]")
        tilted["unnormalised_weight"] = (
            tilted["base_target_weight"] * tilted["tilt_multiplier"]
        )
        normaliser = float(tilted["unnormalised_weight"].sum())
        if not np.isfinite(normaliser) or normaliser <= 0.0:
            raise FusionValidationError("exploratory weights cannot be normalised")
        tilted["target_weight"] = tilted["unnormalised_weight"] / normaliser
        target = tilted.set_index("ticker")["target_weight"].reindex(tickers)
        if not np.isfinite(target).all() or (target < -WEIGHT_TOLERANCE).any():
            raise FusionValidationError("exploratory weights violate long-only constraints")
        if abs(float(target.sum()) - 1.0) > WEIGHT_TOLERANCE:
            raise FusionValidationError("exploratory weights do not sum to one")

        holding = base_returns.loc[base_returns["decision_date"].eq(decision_date)].copy()
        if holding.empty:
            raise FusionValidationError("base decision has no holding-period returns")
        holding = holding.sort_values("date")
        asset_slice = panel.loc[holding["date"], tickers]
        exploratory_returns = asset_slice.to_numpy(dtype=float) @ target.to_numpy(dtype=float)
        for date, daily_return in zip(holding["date"], exploratory_returns, strict=True):
            return_records.append(
                {
                    "date": date,
                    "fund": EXPLORATORY_FUND,
                    "method": EXPLORATORY_METHOD,
                    "asset_family": "equity",
                    "decision_date": decision_date,
                    "daily_return": float(daily_return),
                }
            )

        audit = base_audit.loc[base_audit["decision_date"].eq(decision_date)]
        if len(audit) != 1:
            raise FusionValidationError("base rebalance audit is not unique")
        audit_row = audit.iloc[0]
        training = panel.loc[:decision_date, tickers]
        covariance = training.cov(ddof=1).to_numpy(dtype=float)
        objective = float(target.to_numpy() @ covariance @ target.to_numpy())
        metadata = {
            "fund": EXPLORATORY_FUND,
            "method": EXPLORATORY_METHOD,
            "asset_family": "equity",
            "decision_date": decision_date,
            "training_start_date": audit_row["training_start_date"],
            "training_end_date": audit_row["training_end_date"],
            "first_holding_date": audit_row["first_holding_date"],
            "window_size": int(audit_row["window_size"]),
            "solver_success": True,
            "solver_status": "not_required",
            "solver_message": "fixed exploratory coverage-adjusted sentiment tilt",
            "objective_value": objective,
            "covariance_scale": 1.0,
            "tilt_strength": tilt_strength,
            "min_history_observations": MIN_HISTORY,
            "zscore_clip_bound": ZSCORE_CLIP,
            "signal_lag_trading_days": SIGNAL_LAG,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "signal_window_trading_days": EXPLORATORY_SIGNAL_WINDOW,
            "exploratory_method_label": EXPLORATORY_METHOD_LABEL,
        }
        for row in tilted.itertuples(index=False):
            record = {
                **metadata,
                "ticker": row.ticker,
                "target_weight": float(row.target_weight),
                "base_target_weight": float(row.base_target_weight),
                "unnormalised_weight": float(row.unnormalised_weight),
                "sector": row.sector,
                "signal_window_start_date": row.signal_window_start_date,
                "signal_window_end_date": row.signal_window_end_date,
                "latest_raw_news_date_used": row.latest_raw_news_date_used,
                "trailing_coverage_weighted_sentiment": row.trailing_coverage_weighted_sentiment,
                "effective_coverage": row.effective_coverage,
                "expanding_prior_mean": row.expanding_prior_mean,
                "expanding_prior_std": row.expanding_prior_std,
                "expanding_prior_observations": row.expanding_prior_observations,
                "raw_trailing_zscore": row.raw_trailing_zscore,
                "clipped_trailing_zscore": row.clipped_trailing_zscore,
                "coverage_adjusted_zscore": row.coverage_adjusted_zscore,
                "signal_was_missing": bool(row.signal_was_missing),
                "tilt_multiplier": float(row.tilt_multiplier),
            }
            weight_records.append(record)
        sector_rows = tilted.drop_duplicates("sector").sort_values("sector")
        audit_records.append(
            {
                **metadata,
                "target_weights": json.dumps(
                    {ticker: float(value) for ticker, value in target.items()},
                    sort_keys=True,
                ),
                "available_sector_signals": json.dumps(
                    {
                        row.sector: (
                            float(row.coverage_adjusted_zscore)
                            if pd.notna(row.coverage_adjusted_zscore)
                            else None
                        )
                        for row in sector_rows.itertuples(index=False)
                    },
                    sort_keys=True,
                ),
                "missing_sector_signal_count": int(
                    sector_rows["signal_was_missing"].sum()
                ),
            }
        )

    fund_returns = pd.DataFrame.from_records(return_records).sort_values("date")
    fund_returns["growth_of_1"] = (1.0 + fund_returns["daily_return"]).cumprod()
    fund_weights = pd.DataFrame.from_records(weight_records).sort_values(
        ["decision_date", "ticker"]
    )
    rebalance_audit = pd.DataFrame.from_records(audit_records).sort_values(
        "decision_date"
    )
    expected_returns = base_returns.sort_values("date")
    if tuple(fund_returns["date"]) != tuple(expected_returns["date"]):
        raise FusionValidationError("base and exploratory funds do not share dates")
    if tuple(fund_returns["decision_date"]) != tuple(expected_returns["decision_date"]):
        raise FusionValidationError("base and exploratory funds do not share rebalances")
    return portfolios.BacktestResult(
        fund_returns.reset_index(drop=True),
        fund_weights.reset_index(drop=True),
        rebalance_audit.reset_index(drop=True),
    )


def target_weight_turnover(fund_weights: pd.DataFrame) -> pd.DataFrame:
    """Calculate half-L1 changes in monthly target weights by fund.

    The initial allocation is labelled and assigned zero so it is not mistaken
    for a rebalance.  Transaction costs remain zero and are not deducted.
    """
    required = {
        "fund",
        "asset_family",
        "method",
        "decision_date",
        "ticker",
        "target_weight",
    }
    _require_columns(fund_weights, required, "fund_weights")
    weights = fund_weights.copy()
    weights["decision_date"] = _normalise_date(weights["decision_date"])
    records: list[dict[str, object]] = []
    for fund, group in weights.groupby("fund", sort=True):
        identity = group[["asset_family", "method"]].drop_duplicates()
        if len(identity) != 1:
            raise FusionValidationError(f"inconsistent identifiers for {fund}")
        pivot = group.pivot(
            index="decision_date", columns="ticker", values="target_weight"
        ).sort_index()
        if pivot.isna().any().any():
            raise FusionValidationError(f"asset universe changes within {fund}")
        changes = 0.5 * pivot.diff().abs().sum(axis=1)
        changes.iloc[0] = 0.0
        for position, (decision_date, value) in enumerate(changes.items()):
            records.append(
                {
                    "fund": fund,
                    "asset_family": identity["asset_family"].iloc[0],
                    "method": identity["method"].iloc[0],
                    "decision_date": decision_date,
                    "target_weight_turnover": float(value),
                    "is_initial_allocation": position == 0,
                    "transaction_cost_bps": TRANSACTION_COST_BPS,
                }
            )
    return pd.DataFrame.from_records(records)


def compare_sector_exposure(
    fund_weights: pd.DataFrame,
    ticker_sectors: pd.DataFrame,
    *,
    base_fund: str = BASE_FUND,
    tilted_fund: str = TILTED_FUND,
) -> pd.DataFrame:
    """Return matched base-versus-tilted target sector exposures."""
    mapping = _validate_ticker_sectors(ticker_sectors)
    selected = fund_weights.loc[fund_weights["fund"].isin([base_fund, tilted_fund])].copy()
    if set(selected["fund"]) != {base_fund, tilted_fund}:
        raise FusionValidationError("both base and tilted weights are required")
    selected["decision_date"] = _normalise_date(selected["decision_date"])
    selected = selected.drop(columns="sector", errors="ignore").merge(
        mapping,
        on="ticker",
        how="left",
        validate="many_to_one",
    )
    if selected["sector"].isna().any():
        raise FusionValidationError("sector exposure has unmapped equity tickers")
    long = (
        selected.groupby(["fund", "decision_date", "sector"], as_index=False)[
            "target_weight"
        ]
        .sum()
        .rename(columns={"target_weight": "target_sector_weight"})
    )
    sums = long.groupby(["fund", "decision_date"])["target_sector_weight"].sum()
    if not np.allclose(sums, 1.0, atol=WEIGHT_TOLERANCE, rtol=WEIGHT_TOLERANCE):
        raise FusionValidationError("sector exposures do not sum to one")
    wide = long.pivot(
        index=["decision_date", "sector"],
        columns="fund",
        values="target_sector_weight",
    ).reset_index()
    if wide[[base_fund, tilted_fund]].isna().any().any():
        raise FusionValidationError("base and tilted sector exposures are not matched")
    wide = wide.rename(
        columns={
            base_fund: "base_sector_weight",
            tilted_fund: "tilted_sector_weight",
        }
    )
    wide["sector_weight_difference"] = (
        wide["tilted_sector_weight"] - wide["base_sector_weight"]
    )
    wide["base_fund"] = base_fund
    wide["tilted_fund"] = tilted_fund
    return wide[
        [
            "base_fund",
            "tilted_fund",
            "decision_date",
            "sector",
            "base_sector_weight",
            "tilted_sector_weight",
            "sector_weight_difference",
        ]
    ].sort_values(["decision_date", "sector"]).reset_index(drop=True)


def build_fusion_evidence(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    ticker_sectors: pd.DataFrame,
    *,
    base_fund: str = BASE_FUND,
    tilted_fund: str = TILTED_FUND,
) -> FusionEvidence:
    """Build a fair, matched before-versus-after comparison."""
    selected_returns = fund_returns.loc[
        fund_returns["fund"].isin([base_fund, tilted_fund])
    ].copy()
    if set(selected_returns["fund"]) != {base_fund, tilted_fund}:
        raise FusionValidationError("both base and tilted returns are required")
    selected_returns["date"] = _normalise_date(selected_returns["date"])
    selected_returns["decision_date"] = _normalise_date(
        selected_returns["decision_date"]
    )
    ordered = {
        fund: group.sort_values("date")
        for fund, group in selected_returns.groupby("fund", sort=False)
    }
    base = ordered[base_fund]
    tilted = ordered[tilted_fund]
    for column in ("date", "decision_date"):
        if tuple(base[column]) != tuple(tilted[column]):
            raise FusionValidationError(f"base and tilted {column} samples differ")

    turnover = target_weight_turnover(
        fund_weights.loc[fund_weights["fund"].isin([base_fund, tilted_fund])]
    )
    records: list[dict[str, object]] = []
    metric_names = [
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
    ]
    for role, fund in (("base", base_fund), ("enhanced", tilted_fund)):
        group = ordered[fund]
        statistics = portfolios.performance_metrics(group["daily_return"], 252)
        fund_turnover = turnover.loc[
            turnover["fund"].eq(fund) & ~turnover["is_initial_allocation"]
        ]
        records.append(
            {
                "role": role,
                "fund": fund,
                "asset_family": group["asset_family"].iloc[0],
                "method": group["method"].iloc[0],
                "sample_start_date": group["date"].iloc[0],
                "sample_end_date": group["date"].iloc[-1],
                "observations": len(group),
                "rebalance_count": int(
                    fund_weights.loc[fund_weights["fund"].eq(fund), "decision_date"].nunique()
                ),
                "periods_per_year": 252,
                "risk_free_rate_annual": 0.0,
                "transaction_cost_bps": TRANSACTION_COST_BPS,
                **statistics,
                "average_monthly_target_turnover": float(
                    fund_turnover["target_weight_turnover"].mean()
                ),
                "total_target_turnover": float(
                    fund_turnover["target_weight_turnover"].sum()
                ),
            }
        )
    comparison = pd.DataFrame.from_records(records)
    base_row = comparison.loc[comparison["role"].eq("base")].iloc[0]
    for metric in [*metric_names, "average_monthly_target_turnover", "total_target_turnover"]:
        comparison[f"{metric}_difference_vs_base"] = comparison[metric] - base_row[metric]
    exposure = compare_sector_exposure(
        fund_weights,
        ticker_sectors,
        base_fund=base_fund,
        tilted_fund=tilted_fund,
    )
    return FusionEvidence(comparison, exposure, turnover)


LOCKED_SIGNAL_METHOD = "locked_prior_day_expanding_zscore"
EXPLORATORY_SIGNAL_METHOD = "exploratory_21d_coverage_adjusted_zscore"
PREDICTABILITY_FOLDS = (
    ("through_2021", pd.Timestamp("2021-01-04"), pd.Timestamp("2021-12-31")),
    ("through_2022", pd.Timestamp("2021-01-04"), pd.Timestamp("2022-12-30")),
    ("through_2023", pd.Timestamp("2021-01-04"), pd.Timestamp("2023-12-29")),
)


def _cross_sectional_rank_ic(group: pd.DataFrame) -> tuple[float, int]:
    usable = group.dropna(subset=["available_signal", "sector_excess_return"])
    if len(usable) < 3:
        return np.nan, len(usable)
    signal_rank = usable["available_signal"].rank(method="average")
    return_rank = usable["sector_excess_return"].rank(method="average")
    value = signal_rank.corr(return_rank)
    return (float(value) if pd.notna(value) else np.nan), len(usable)


def build_sentiment_predictability(
    base_backtest: portfolios.BacktestResult,
    equity_returns: pd.DataFrame,
    ticker_sectors: pd.DataFrame,
    locked_signal: pd.DataFrame,
    exploratory_signal: pd.DataFrame,
) -> pd.DataFrame:
    """Relate decision-time sector signals to the next actual holding period.

    Each monthly rank IC is cross-sectional across sectors.  No observations
    are shuffled and no parameter is estimated or selected from the returns.
    """
    mapping = _validate_ticker_sectors(ticker_sectors)
    locked = _validate_locked_signal(locked_signal)
    exploratory = _validate_exploratory_signal(exploratory_signal)
    if {"ticker_coverage_share", "has_observed_news"}.difference(locked.columns):
        raise FusionValidationError("locked signal lacks source-day coverage fields")

    source_coverage = locked[
        ["date", "sector", "ticker_coverage_share", "has_observed_news"]
    ].rename(
        columns={
            "date": "tradable_signal_source_date",
            "ticker_coverage_share": "effective_coverage",
            "has_observed_news": "source_has_observed_news",
        }
    )
    locked_available = locked[
        [
            "date",
            "sector",
            "tradable_sector_zscore",
            "tradable_signal_source_date",
        ]
    ].merge(
        source_coverage,
        on=["tradable_signal_source_date", "sector"],
        how="left",
        validate="many_to_one",
    )
    locked_available = locked_available.rename(
        columns={"tradable_sector_zscore": "available_signal"}
    )
    locked_available["signal_method"] = LOCKED_SIGNAL_METHOD
    locked_available["news_coverage_measure"] = "source-day ticker coverage share"

    exploratory_available = exploratory[
        ["date", "sector", "coverage_adjusted_zscore", "effective_coverage"]
    ].rename(columns={"coverage_adjusted_zscore": "available_signal"})
    exploratory_available["signal_method"] = EXPLORATORY_SIGNAL_METHOD
    exploratory_available["news_coverage_measure"] = (
        "21-day observed ticker opportunities / total ticker opportunities"
    )
    signals = pd.concat(
        [
            locked_available[
                [
                    "date",
                    "sector",
                    "signal_method",
                    "available_signal",
                    "effective_coverage",
                    "news_coverage_measure",
                ]
            ],
            exploratory_available[
                [
                    "date",
                    "sector",
                    "signal_method",
                    "available_signal",
                    "effective_coverage",
                    "news_coverage_measure",
                ]
            ],
        ],
        ignore_index=True,
    )

    base_returns = base_backtest.fund_returns.loc[
        base_backtest.fund_returns["fund"].eq(BASE_FUND)
    ].copy()
    if base_returns.empty:
        raise FusionValidationError(f"base backtest must contain {BASE_FUND}")
    base_returns["date"] = _normalise_date(base_returns["date"])
    base_returns["decision_date"] = _normalise_date(base_returns["decision_date"])
    tickers = sorted(base_backtest.fund_weights.loc[
        base_backtest.fund_weights["fund"].eq(BASE_FUND), "ticker"
    ].unique())
    mapping = mapping.loc[mapping["ticker"].isin(tickers)].copy()
    if len(mapping) != len(tickers):
        raise FusionValidationError("predictability mapping does not cover equities")
    sectors = sorted(mapping["sector"].unique())

    panel = equity_returns.copy()
    _require_columns(panel, {"date", *tickers}, "equity_returns")
    panel["date"] = _normalise_date(panel["date"])
    panel = panel.sort_values("date").set_index("date")
    panel[tickers] = panel[tickers].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(panel[tickers].to_numpy(dtype=float)).all():
        raise FusionValidationError("predictability returns must be finite")

    records: list[dict[str, object]] = []
    for decision_date, holding in base_returns.groupby("decision_date", sort=True):
        holding_dates = pd.DatetimeIndex(holding["date"].sort_values())
        if holding_dates.empty or not (holding_dates > decision_date).all():
            raise FusionValidationError("predictability returns are not after decision")
        holding_returns = panel.loc[holding_dates, tickers]
        market_daily = holding_returns.mean(axis=1)
        market_period_return = float((1.0 + market_daily).prod() - 1.0)
        decision_signals = signals.loc[signals["date"].eq(decision_date)]
        if set(decision_signals["sector"]) != set(sectors):
            raise FusionValidationError("decision signal does not cover every sector row")

        for sector in sectors:
            sector_tickers = mapping.loc[mapping["sector"].eq(sector), "ticker"].tolist()
            sector_daily = holding_returns[sector_tickers].mean(axis=1)
            sector_period_return = float((1.0 + sector_daily).prod() - 1.0)
            sector_excess = (
                (1.0 + sector_period_return) / (1.0 + market_period_return) - 1.0
            )
            for signal_row in decision_signals.loc[
                decision_signals["sector"].eq(sector)
            ].itertuples(index=False):
                records.append(
                    {
                        "decision_date": decision_date,
                        "first_holding_date": holding_dates[0],
                        "last_holding_date": holding_dates[-1],
                        "sector": sector,
                        "signal_method": signal_row.signal_method,
                        "available_signal": signal_row.available_signal,
                        "effective_coverage": signal_row.effective_coverage,
                        "news_coverage_measure": signal_row.news_coverage_measure,
                        "sector_holding_period_return": sector_period_return,
                        "market_holding_period_return": market_period_return,
                        "sector_excess_return": sector_excess,
                    }
                )
    result = pd.DataFrame.from_records(records).sort_values(
        ["decision_date", "signal_method", "sector"]
    )
    expected_rows = (
        base_returns["decision_date"].nunique()
        * len(sectors)
        * signals["signal_method"].nunique()
    )
    if len(result) != expected_rows:
        raise FusionValidationError(
            f"predictability row conservation failed: {len(result)} != {expected_rows}"
        )

    ic_records: list[dict[str, object]] = []
    for (decision_date, method), group in result.groupby(
        ["decision_date", "signal_method"], sort=True
    ):
        rank_ic, count = _cross_sectional_rank_ic(group)
        ic_records.append(
            {
                "decision_date": decision_date,
                "signal_method": method,
                "monthly_spearman_rank_ic": rank_ic,
                "usable_sector_count_for_ic": count,
            }
        )
    result = result.merge(
        pd.DataFrame.from_records(ic_records),
        on=["decision_date", "signal_method"],
        how="left",
        validate="many_to_one",
    )
    return result.reset_index(drop=True)


def _spearman_over_time(group: pd.DataFrame) -> float:
    usable = group.dropna(subset=["available_signal", "sector_excess_return"])
    if len(usable) < 3:
        return np.nan
    value = usable["available_signal"].rank().corr(
        usable["sector_excess_return"].rank()
    )
    return float(value) if pd.notna(value) else np.nan


def build_predictability_summary(predictability: pd.DataFrame) -> pd.DataFrame:
    """Summarise rank evidence over predeclared time-ordered expanding folds."""
    required = {
        "decision_date",
        "first_holding_date",
        "sector",
        "signal_method",
        "available_signal",
        "effective_coverage",
        "sector_excess_return",
        "monthly_spearman_rank_ic",
    }
    _require_columns(predictability, required, "predictability")
    data = predictability.copy()
    data["decision_date"] = _normalise_date(data["decision_date"])
    data["first_holding_date"] = _normalise_date(data["first_holding_date"])
    records: list[dict[str, object]] = []

    for fold_name, fold_start, fold_end in PREDICTABILITY_FOLDS:
        fold = data.loc[
            data["first_holding_date"].between(fold_start, fold_end, inclusive="both")
        ]
        for method, method_group in fold.groupby("signal_method", sort=True):
            usable = method_group.dropna(
                subset=["available_signal", "sector_excess_return"]
            )
            monthly = method_group[
                ["decision_date", "monthly_spearman_rank_ic"]
            ].drop_duplicates("decision_date")
            monthly_ic = monthly["monthly_spearman_rank_ic"].dropna()
            common = {
                "temporal_fold": fold_name,
                "period_start_date": fold_start,
                "period_end_date": fold_end,
                "signal_method": method,
                "monthly_decisions": method_group["decision_date"].nunique(),
                "signal_coverage_share": float(
                    method_group["available_signal"].notna().mean()
                ),
                "mean_effective_news_coverage": float(
                    method_group["effective_coverage"].mean()
                ),
                "final_period_used_for_parameter_selection": False,
            }
            records.append(
                {
                    **common,
                    "summary_scope": "overall",
                    "sector": "ALL",
                    "usable_sector_observations": len(usable),
                    "usable_monthly_rank_ic_observations": len(monthly_ic),
                    "mean_monthly_rank_ic": float(monthly_ic.mean()),
                    "median_monthly_rank_ic": float(monthly_ic.median()),
                    "proportion_monthly_rank_ic_above_zero": float(
                        monthly_ic.gt(0).mean()
                    ),
                    "sector_time_series_spearman": np.nan,
                }
            )
            for sector, sector_group in method_group.groupby("sector", sort=True):
                sector_usable = sector_group.dropna(
                    subset=["available_signal", "sector_excess_return"]
                )
                records.append(
                    {
                        **common,
                        "monthly_decisions": sector_group[
                            "decision_date"
                        ].nunique(),
                        "signal_coverage_share": float(
                            sector_group["available_signal"].notna().mean()
                        ),
                        "mean_effective_news_coverage": float(
                            sector_group["effective_coverage"].mean()
                        ),
                        "summary_scope": "sector",
                        "sector": sector,
                        "usable_sector_observations": len(sector_usable),
                        "usable_monthly_rank_ic_observations": np.nan,
                        "mean_monthly_rank_ic": np.nan,
                        "median_monthly_rank_ic": np.nan,
                        "proportion_monthly_rank_ic_above_zero": np.nan,
                        "sector_time_series_spearman": _spearman_over_time(
                            sector_group
                        ),
                    }
                )
    return pd.DataFrame.from_records(records).sort_values(
        ["period_end_date", "signal_method", "summary_scope", "sector"]
    ).reset_index(drop=True)


__all__ = [
    "BASE_FUND",
    "EXPLORATORY_FUND",
    "EXPLORATORY_METHOD",
    "EXPLORATORY_METHOD_LABEL",
    "EXPLORATORY_SIGNAL_METHOD",
    "EXPLORATORY_SIGNAL_WINDOW",
    "LOCKED_SIGNAL_METHOD",
    "MIN_HISTORY",
    "PREDICTABILITY_FOLDS",
    "SIGNAL_LAG",
    "TILTED_FUND",
    "TILTED_METHOD",
    "TILT_STRENGTH",
    "TRANSACTION_COST_BPS",
    "WEIGHT_TOLERANCE",
    "ZSCORE_CLIP",
    "FusionEvidence",
    "FusionValidationError",
    "apply_exploratory_coverage_tilt",
    "apply_sentiment_tilt",
    "build_fusion_evidence",
    "build_predictability_summary",
    "build_sentiment_predictability",
    "compare_sector_exposure",
    "target_weight_turnover",
]
