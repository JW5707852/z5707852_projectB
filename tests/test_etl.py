"""Focused tests for the Part B data foundation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src import etl, features


def _price_rows(
    ticker: str,
    dates: list[str],
    adjusted: list[float],
    *,
    closes: list[float] | None = None,
) -> pd.DataFrame:
    close_values = adjusted if closes is None else closes
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(dates),
            "open": close_values,
            "high": [value * 1.01 for value in close_values],
            "low": [value * 0.99 for value in close_values],
            "close": close_values,
            "adjClose": adjusted,
            "volume": [100] * len(dates),
        }
    )


def _return_prices(
    ticker: str,
    dates: list[str],
    adjusted: list[float],
    *,
    closes: list[float] | None = None,
) -> pd.DataFrame:
    frame = {
        "ticker": ticker,
        "date": pd.to_datetime(dates),
        "adjClose": adjusted,
    }
    if closes is not None:
        frame["close"] = closes
    return pd.DataFrame(frame)


def _news_rows(dates: list[str], titles: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates, utc=True),
            "ticker": ["ABC"] * len(dates),
            "sector": ["Tech"] * len(dates),
            "title": titles,
            "url": [f"url-{index}" for index in range(len(dates))],
            "publisher": [None] * len(dates),
        }
    )


def test_daily_returns_use_adjusted_close_within_ticker() -> None:
    prices = pd.concat(
        [
            _return_prices(
                "A",
                ["2023-01-02", "2023-01-03"],
                [100.0, 110.0],
                closes=[10.0, 50.0],
            ),
            _return_prices(
                "B",
                ["2023-01-02", "2023-01-03"],
                [50.0, 40.0],
                closes=[200.0, 201.0],
            ),
        ],
        ignore_index=True,
    )

    result = features.daily_returns(prices)
    observed = result.dropna(subset=["daily_return"]).set_index("ticker")

    assert observed.loc["A", "daily_return"] == pytest.approx(0.10)
    assert observed.loc["B", "daily_return"] == pytest.approx(-0.20)
    assert result["daily_return"].isna().sum() == 2


@pytest.mark.parametrize("invalid_price", [0.0, -1.0])
def test_daily_returns_reject_non_positive_adjusted_prices(
    invalid_price: float,
) -> None:
    prices = _return_prices(
        "A",
        ["2023-01-02", "2023-01-03"],
        [100.0, invalid_price],
    )

    with pytest.raises(ValueError, match="non-positive adjClose"):
        features.daily_returns(prices)


def test_crypto_returns_are_calculated_before_equity_calendar_alignment() -> None:
    equity = features.daily_returns(
        _return_prices("EQ", ["2023-01-02", "2023-01-03"], [100.0, 101.0])
    )
    crypto = features.daily_returns(
        _return_prices(
            "COIN-USD",
            ["2023-01-01", "2023-01-02", "2023-01-03"],
            [10.0, 12.0, 18.0],
        )
    )

    combined = features.combined_returns_panel(equity, crypto).set_index("date")

    assert pd.Timestamp("2023-01-01") not in combined.index
    assert combined.loc["2023-01-02", "COIN-USD"] == pytest.approx(0.20)
    assert combined.loc["2023-01-03", "COIN-USD"] == pytest.approx(0.50)


def test_crypto_cutoff_and_exact_price_duplicate_rule() -> None:
    prices = _price_rows(
        "COIN-USD",
        ["2023-12-31", "2024-01-01", "2023-12-31"],
        [100.0, 101.0, 100.0],
    )

    clean = etl.clean_price_panel(
        prices,
        dataset="crypto_prices",
        end_date=etl.CRYPTO_END_DATE,
    )

    assert len(clean) == 1
    assert clean["date"].max() == pd.Timestamp("2023-12-31")
    assert not clean.duplicated(["ticker", "date"]).any()


def test_conflicting_price_duplicate_keys_stop_cleaning() -> None:
    prices = _price_rows(
        "EQ",
        ["2023-12-29", "2023-12-29"],
        [100.0, 101.0],
    )

    with pytest.raises(ValueError, match="conflicting ticker-date"):
        etl.clean_price_panel(prices, dataset="equity_prices")


def test_news_deduplication_preserves_title_as_text_raw() -> None:
    news = _news_rows(
        ["2023-01-07", "2023-01-07", "2023-01-07"],
        ["Profit Jumps!", "Profit Jumps!", "A second headline"],
    )

    clean = etl.clean_news_headlines(news)

    assert len(clean) == 2
    assert clean["text_raw"].equals(clean["title"])
    assert set(clean["text_raw"]) == {"Profit Jumps!", "A second headline"}
    assert not clean.duplicated(["ticker", "date", "title"]).any()


def test_aware_news_and_naive_prices_normalise_to_same_date() -> None:
    prices = _price_rows(
        "EQ",
        ["2023-01-07 15:00:00"],
        [100.0],
    )
    news = _news_rows(
        ["2023-01-07T15:00:00+00:00"],
        ["Case and punctuation remain!"],
    )

    clean_prices = etl.clean_price_panel(prices, dataset="equity_prices")
    clean_news = etl.clean_news_headlines(news)

    assert clean_prices.loc[0, "date"] == pd.Timestamp("2023-01-07")
    assert clean_news.loc[0, "date"] == pd.Timestamp("2023-01-07")
    assert clean_prices["date"].dt.tz is None
    assert clean_news["date"].dt.tz is None
    assert clean_news.loc[0, "text_raw"] == "Case and punctuation remain!"


def test_headlines_map_same_day_forward_and_outside_sample() -> None:
    clean = etl.clean_news_headlines(
        _news_rows(
            ["2023-01-06", "2023-01-07", "2023-01-10"],
            ["Same day", "Weekend headline", "After sample"],
        )
    )

    mapped = features.map_headlines_to_trading_days(
        clean,
        pd.to_datetime(["2023-01-06", "2023-01-09"]),
    )

    assert mapped["mapping_status"].tolist() == [
        "same_trading_day",
        "next_trading_day",
        "outside_equity_sample",
    ]
    assert mapped.loc[1, "trading_date"] == pd.Timestamp("2023-01-09")
    assert pd.isna(mapped.loc[2, "trading_date"])
    in_sample = mapped.dropna(subset=["trading_date"])
    assert (in_sample["trading_date"] >= in_sample["date"]).all()


def test_missing_return_audit_drops_only_leading_gaps_and_never_imputes() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-01-02", "2023-01-03", "2023-01-04"]
            ),
            "EQ": [np.nan, 0.01, 0.02],
            "COIN-USD": [0.03, 0.04, 0.05],
        }
    )

    audit = features.return_missingness_audit(panel).set_index("ticker")
    usable = features.complete_return_panel(panel)

    assert audit.loc["EQ", "leading_missing_returns"] == 1
    assert audit["unexpected_missing_returns"].sum() == 0
    assert usable["date"].min() == pd.Timestamp("2023-01-03")
    assert np.isfinite(usable.drop(columns="date").to_numpy()).all()

    broken = panel.copy()
    broken.loc[2, "COIN-USD"] = np.nan
    with pytest.raises(ValueError, match="unexpected non-finite live returns"):
        features.complete_return_panel(broken)


@pytest.fixture(scope="module")
def official_clean_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        etl.load_clean_equities(),
        etl.load_clean_crypto(),
        etl.load_clean_news(),
    )


def test_official_clean_data_counts_ranges_and_unique_keys(
    official_clean_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    equities, crypto, news = official_clean_data

    assert len(equities) == 50_300
    assert equities["date"].min() == pd.Timestamp("2020-01-02")
    assert equities["date"].max() == pd.Timestamp("2023-12-29")
    assert not equities.duplicated(["ticker", "date"]).any()

    assert len(crypto) == 14_610
    assert crypto["date"].min() == pd.Timestamp("2020-01-01")
    assert crypto["date"].max() == pd.Timestamp("2023-12-31")
    assert not crypto.duplicated(["ticker", "date"]).any()

    assert len(news) == 146_836
    assert news["date"].min() == pd.Timestamp("2020-01-01")
    assert news["date"].max() == pd.Timestamp("2023-12-31")
    assert news["text_raw"].equals(news["title"])
    assert not news.duplicated(["ticker", "date", "title"]).any()


def test_official_usable_combined_return_panel_has_no_missing_returns(
    official_clean_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    equities, crypto, _ = official_clean_data
    equity_returns = features.daily_returns(equities, asset_class="equity")
    crypto_returns = features.daily_returns(crypto, asset_class="crypto")
    combined = features.combined_returns_panel(equity_returns, crypto_returns)
    audit = features.return_missingness_audit(combined)
    usable = features.complete_return_panel(combined)

    assert equity_returns["daily_return"].notna().sum() == 50_250
    assert crypto_returns["daily_return"].notna().sum() == 14_600
    assert combined.shape == (1_006, 61)
    assert usable.shape == (1_005, 61)
    assert audit["unexpected_missing_returns"].sum() == 0
    assert np.isfinite(usable.drop(columns="date").to_numpy()).all()
