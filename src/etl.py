"""Clean and audit the supplied Project B datasets.

All hosted data enters through :mod:`src.data_access`.  Cleaning is deliberately
conservative: exact duplicate keys may be collapsed, conflicting price keys stop
the workflow, and original headline text is preserved for later VADER scoring.
"""
from __future__ import annotations

import pandas as pd

from src import data_access

CRYPTO_END_DATE = pd.Timestamp("2023-12-31")
PRICE_KEY = ["ticker", "date"]
NEWS_KEY = ["ticker", "date", "title"]
OHLC_COLUMNS = ["open", "high", "low", "close"]
PRICE_COLUMNS = [*OHLC_COLUMNS, "adjClose"]


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    """Raise a clear error when a required input column is absent."""
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def normalise_date(series: pd.Series) -> pd.Series:
    """Convert aware or naive timestamps to timezone-naive UTC calendar dates."""
    return pd.to_datetime(series, utc=True).dt.tz_convert(None).dt.normalize()


def _require_nonmissing_key(
    frame: pd.DataFrame,
    key: list[str],
    name: str,
) -> None:
    missing_rows = int(frame[key].isna().any(axis=1).sum())
    if missing_rows:
        raise ValueError(f"{name} contains {missing_rows} rows with a missing key")


def conflicting_duplicate_keys(frame: pd.DataFrame, key: list[str]) -> int:
    """Count duplicate keys whose non-key values disagree."""
    duplicate_rows = frame.loc[frame.duplicated(key, keep=False)]
    if duplicate_rows.empty:
        return 0
    return sum(
        len(group.drop_duplicates()) > 1
        for _, group in duplicate_rows.groupby(key, dropna=False, sort=False)
    )


def clean_price_panel(
    prices: pd.DataFrame,
    *,
    dataset: str,
    end_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Normalise and uniquely key a price panel without hiding conflicts.

    Exact duplicate ticker-date observations are safe to collapse.  If two
    observations share a ticker-date key but disagree elsewhere, the build stops
    rather than choosing one silently.
    """
    required = set(PRICE_KEY + PRICE_COLUMNS + ["volume"])
    require_columns(prices, required, dataset)

    clean = prices.copy()
    clean["date"] = normalise_date(clean["date"])
    _require_nonmissing_key(clean, PRICE_KEY, dataset)
    if end_date is not None:
        clean = clean.loc[clean["date"] <= pd.Timestamp(end_date)].copy()

    conflicts = conflicting_duplicate_keys(clean, PRICE_KEY)
    if conflicts:
        raise ValueError(
            f"{dataset} contains {conflicts} conflicting ticker-date keys"
        )

    clean = (
        clean.drop_duplicates(PRICE_KEY, keep="first")
        .sort_values(PRICE_KEY)
        .reset_index(drop=True)
    )
    if clean.duplicated(PRICE_KEY).any():
        raise AssertionError(f"{dataset} is not unique by ticker and date")
    return clean


def clean_news_headlines(headlines: pd.DataFrame) -> pd.DataFrame:
    """Remove exact headline duplicates while preserving the original title."""
    required = {"date", "ticker", "sector", "title", "url", "publisher"}
    require_columns(headlines, required, "news_headlines")

    clean = headlines.copy()
    clean["date"] = normalise_date(clean["date"])
    _require_nonmissing_key(clean, NEWS_KEY, "news_headlines")
    clean["text_raw"] = clean["title"].copy()
    clean = (
        clean.drop_duplicates(NEWS_KEY, keep="first")
        .sort_values(["date", "ticker", "title"])
        .reset_index(drop=True)
    )
    if clean.duplicated(NEWS_KEY).any():
        raise AssertionError(
            "news_headlines is not unique by ticker, date, and title"
        )
    return clean


def load_clean_equities() -> pd.DataFrame:
    """Load and clean the supplied equity panel."""
    return clean_price_panel(
        data_access.load_equity_prices(),
        dataset="equity_prices",
    )


def load_clean_crypto() -> pd.DataFrame:
    """Load and clean crypto prices, excluding the ten 2024 stray rows."""
    return clean_price_panel(
        data_access.load_crypto_prices(),
        dataset="crypto_prices",
        end_date=CRYPTO_END_DATE,
    )


def load_clean_news() -> pd.DataFrame:
    """Load and clean the supplied headline panel."""
    return clean_news_headlines(data_access.load_news_headlines())


def _missing_date_audit(
    prices: pd.DataFrame,
    *,
    dataset: str,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    expected = pd.DatetimeIndex(calendar).sort_values().unique()
    if len(expected) == 0:
        raise ValueError(f"{dataset} calendar is empty")

    for ticker, group in prices.groupby("ticker", sort=True):
        observed = pd.DatetimeIndex(group["date"].unique())
        missing = expected.difference(observed)
        records.append(
            {
                "dataset": dataset,
                "ticker": ticker,
                "observations": len(observed),
                "expected_dates": len(expected),
                "missing_dates": len(missing),
                "coverage_pct": 100 * len(observed) / len(expected),
                "missing_dates_preview": ";".join(
                    date.strftime("%Y-%m-%d") for date in missing[:10]
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def audit_missing_price_dates(
    equities: pd.DataFrame,
    crypto: pd.DataFrame,
) -> pd.DataFrame:
    """Audit prices against the appropriate equity or calendar-day calendar."""
    require_columns(equities, {"ticker", "date"}, "equities")
    require_columns(crypto, {"ticker", "date"}, "crypto")
    equity_calendar = pd.DatetimeIndex(equities["date"].unique())
    crypto_calendar = pd.date_range(
        crypto["date"].min(),
        crypto["date"].max(),
        freq="D",
    )
    return pd.concat(
        [
            _missing_date_audit(
                equities,
                dataset="equity_prices",
                calendar=equity_calendar,
            ),
            _missing_date_audit(
                crypto,
                dataset="crypto_prices",
                calendar=crypto_calendar,
            ),
        ],
        ignore_index=True,
    )


__all__ = [
    "CRYPTO_END_DATE",
    "NEWS_KEY",
    "PRICE_KEY",
    "audit_missing_price_dates",
    "clean_news_headlines",
    "clean_price_panel",
    "conflicting_duplicate_keys",
    "load_clean_crypto",
    "load_clean_equities",
    "load_clean_news",
    "normalise_date",
    "require_columns",
]
