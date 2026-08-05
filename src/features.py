"""Return construction and headline-calendar alignment for Project B.

Asset returns are calculated on native calendars before alignment.  Headline
assembly remains row-level so the later sentiment build can score each distinct
raw title exactly once before ticker-day and sector aggregation.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src import etl


def daily_returns(
    prices: pd.DataFrame,
    price_col: str = "adjClose",
    *,
    asset_class: str | None = None,
) -> pd.DataFrame:
    """Calculate simple returns within each ticker's native price calendar."""
    etl.require_columns(prices, {"ticker", "date", price_col}, "prices")

    ordered = prices.copy()
    ordered["date"] = etl.normalise_date(ordered["date"])
    if ordered[["ticker", "date"]].isna().any(axis=1).any():
        raise ValueError("prices contains a missing ticker or date")
    if ordered.duplicated(["ticker", "date"]).any():
        raise ValueError("prices must be unique by ticker and date")

    ordered = ordered.sort_values(["ticker", "date"]).reset_index(drop=True)
    ordered[price_col] = pd.to_numeric(ordered[price_col], errors="coerce")
    finite_prices = np.isfinite(ordered[price_col].to_numpy(dtype=float))
    if not finite_prices.all():
        raise ValueError(f"prices contains non-finite {price_col} values")
    if (ordered[price_col] <= 0).any():
        raise ValueError(f"prices contains non-positive {price_col} values")

    prior_col = f"prior_{price_col}"
    ordered[prior_col] = ordered.groupby("ticker")[price_col].shift()
    ordered["daily_return"] = ordered.groupby("ticker")[price_col].pct_change(
        fill_method=None
    )
    non_leading = ordered[prior_col].notna()
    if not np.isfinite(
        ordered.loc[non_leading, "daily_return"].to_numpy(dtype=float)
    ).all():
        raise ValueError("calculated daily returns contain non-finite values")
    if asset_class is not None:
        ordered["asset_class"] = asset_class

    columns = ["date", "ticker"]
    if "sector" in ordered.columns:
        columns.append("sector")
    if asset_class is not None:
        columns.append("asset_class")
    columns.extend([prior_col, price_col, "daily_return"])
    return ordered[columns]


def combined_returns_panel(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join separately calculated crypto returns to equity trading dates."""
    required = {"date", "ticker", "daily_return"}
    etl.require_columns(equity_returns, required, "equity_returns")
    etl.require_columns(crypto_returns, required, "crypto_returns")

    equity = equity_returns.copy()
    crypto = crypto_returns.copy()
    equity["date"] = etl.normalise_date(equity["date"])
    crypto["date"] = etl.normalise_date(crypto["date"])
    for name, frame in (("equity_returns", equity), ("crypto_returns", crypto)):
        if frame.duplicated(["ticker", "date"]).any():
            raise ValueError(f"{name} must be unique by ticker and date")

    equity_wide = equity.pivot(
        index="date",
        columns="ticker",
        values="daily_return",
    ).sort_index()
    crypto_wide = crypto.pivot(
        index="date",
        columns="ticker",
        values="daily_return",
    ).sort_index()
    collisions = sorted(set(equity_wide.columns).intersection(crypto_wide.columns))
    if collisions:
        raise ValueError(f"ticker names overlap across asset classes: {collisions}")

    combined = equity_wide.join(crypto_wide, how="left")
    combined.columns.name = None
    return combined.reset_index()


def return_missingness_audit(
    return_panel: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Report leading and unexpected non-finite returns for every asset column."""
    etl.require_columns(return_panel, {date_col}, "return_panel")
    asset_columns = [column for column in return_panel.columns if column != date_col]
    if not asset_columns:
        raise ValueError("return_panel must contain at least one asset column")

    dates = etl.normalise_date(return_panel[date_col]).reset_index(drop=True)
    records: list[dict[str, object]] = []
    for ticker in asset_columns:
        numeric = pd.to_numeric(return_panel[ticker], errors="coerce")
        values = numeric.to_numpy(dtype=float)
        valid = np.isfinite(values)
        missing_count = int((~valid).sum())
        if valid.any():
            valid_positions = np.flatnonzero(valid)
            first_valid = int(valid_positions[0])
            last_valid = int(valid_positions[-1])
            leading_missing = first_valid
            unexpected_missing = int((~valid[first_valid:]).sum())
            first_valid_date: pd.Timestamp | pd.NaT = dates.iloc[first_valid]
            last_valid_date: pd.Timestamp | pd.NaT = dates.iloc[last_valid]
        else:
            leading_missing = 0
            unexpected_missing = len(values)
            first_valid_date = pd.NaT
            last_valid_date = pd.NaT
        records.append(
            {
                "ticker": ticker,
                "observations": len(values),
                "missing_returns": missing_count,
                "leading_missing_returns": leading_missing,
                "unexpected_missing_returns": unexpected_missing,
                "first_valid_date": first_valid_date,
                "last_valid_date": last_valid_date,
            }
        )
    return pd.DataFrame.from_records(records)


def complete_return_panel(
    return_panel: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Remove only leading structural gaps and reject later missing returns.

    The first equity date has no equity return because no prior equity price is
    available.  This function starts the usable sample at the first date where
    every asset return is finite.  Any non-finite value from that point onward is
    an unexpected data issue and stops the build; no value is imputed.
    """
    etl.require_columns(return_panel, {date_col}, "return_panel")
    panel = return_panel.copy()
    panel[date_col] = etl.normalise_date(panel[date_col])
    if panel[date_col].duplicated().any():
        raise ValueError("return_panel dates must be unique")
    panel = panel.sort_values(date_col).reset_index(drop=True)

    asset_columns = [column for column in panel.columns if column != date_col]
    if not asset_columns:
        raise ValueError("return_panel must contain at least one asset column")
    numeric = panel[asset_columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    complete_rows = finite.all(axis=1)
    if not complete_rows.any():
        raise ValueError("return_panel has no date with complete finite returns")

    first_complete = int(np.flatnonzero(complete_rows)[0])
    if not complete_rows[first_complete:].all():
        bad_position = first_complete + int(
            np.flatnonzero(~complete_rows[first_complete:])[0]
        )
        bad_assets = [
            asset_columns[index]
            for index in np.flatnonzero(~finite[bad_position])
        ]
        bad_date = panel.loc[bad_position, date_col].strftime("%Y-%m-%d")
        raise ValueError(
            "unexpected non-finite live returns on "
            f"{bad_date} for {bad_assets[:10]}"
        )

    usable = panel.loc[first_complete:].copy().reset_index(drop=True)
    usable[asset_columns] = numeric.loc[first_complete:].reset_index(drop=True)
    return usable


def map_headlines_to_trading_days(
    headlines: pd.DataFrame,
    equity_calendar: Iterable[object],
) -> pd.DataFrame:
    """Map each headline to the same or next equity trading day."""
    etl.require_columns(
        headlines,
        {"date", "ticker", "sector", "title", "text_raw"},
        "headlines",
    )
    calendar_series = pd.Series(list(equity_calendar), dtype="object")
    if calendar_series.empty:
        raise ValueError("equity_calendar must contain at least one date")
    calendar = pd.DatetimeIndex(etl.normalise_date(calendar_series))
    calendar = calendar.sort_values().unique()

    mapped = headlines.copy()
    mapped["date"] = etl.normalise_date(mapped["date"])
    source_dates = mapped["date"].to_numpy(dtype="datetime64[ns]")
    calendar_values = calendar.to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(calendar_values, source_dates, side="left")
    within_sample = positions < len(calendar_values)

    trading_dates = np.full(
        len(mapped),
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )
    trading_dates[within_sample] = calendar_values[positions[within_sample]]
    mapped["trading_date"] = pd.to_datetime(trading_dates)

    status = np.full(len(mapped), "outside_equity_sample", dtype=object)
    same_day = within_sample & (trading_dates == source_dates)
    status[within_sample & ~same_day] = "next_trading_day"
    status[same_day] = "same_trading_day"
    mapped["mapping_status"] = status
    mapped["calendar_days_forward"] = (
        mapped["trading_date"] - mapped["date"]
    ).dt.days.astype("Int64")
    return mapped


def assemble_headline_panel(
    headlines: pd.DataFrame,
    equity_calendar: Iterable[object],
) -> pd.DataFrame:
    """Return the mapped row-level headline panel for later title scoring.

    This data-foundation step deliberately does not concatenate, clean, or score
    title text.  One clean headline remains one row so the sentiment model can
    score each distinct title once and validate its join cardinality.
    """
    mapped = map_headlines_to_trading_days(headlines, equity_calendar)
    return mapped.sort_values(
        ["trading_date", "ticker", "date", "title"],
        na_position="last",
    ).reset_index(drop=True)


__all__ = [
    "assemble_headline_panel",
    "combined_returns_panel",
    "complete_return_panel",
    "daily_returns",
    "map_headlines_to_trading_days",
    "return_missingness_audit",
]
