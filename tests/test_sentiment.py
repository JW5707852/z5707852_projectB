"""Focused tests for VADER scoring, aggregation order, coverage, and timing."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from src import etl, features, sentiment


class SpyAnalyzer:
    """Deterministic analyzer that records every unchanged input string."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[str] = []

    def polarity_scores(self, text: str) -> dict[str, Any]:
        self.calls.append(text)
        return {"compound": self.scores[text]}


def test_distinct_titles_are_scored_once_and_joined_without_row_change() -> None:
    panel = pd.DataFrame(
        {
            "title": ["Profit JUMPS!", "Profit JUMPS!", "Risk does not fall."],
            "text_raw": [
                "Profit JUMPS!",
                "Profit JUMPS!",
                "Risk does not fall.",
            ],
            "ticker": ["AAA", "BBB", "AAA"],
        }
    )
    analyzer = SpyAnalyzer(
        {"Profit JUMPS!": 0.8, "Risk does not fall.": -0.4}
    )

    cache = sentiment.score_distinct_titles(panel, analyzer=analyzer)
    scored = sentiment.score_headlines(panel, title_score_cache=cache)

    assert analyzer.calls == ["Profit JUMPS!", "Risk does not fall."]
    assert len(cache) == 2
    assert len(scored) == len(panel)
    assert scored["text_raw"].equals(panel["text_raw"])
    assert scored["vader_compound"].tolist() == [0.8, 0.8, -0.4]
    assert not scored.duplicated().any()


def test_duplicate_title_cache_is_rejected_before_join() -> None:
    panel = pd.DataFrame(
        {"title": ["Good"], "text_raw": ["Good"], "ticker": ["AAA"]}
    )
    invalid_cache = pd.DataFrame(
        {"text_raw": ["Good", "Good"], "vader_compound": [0.2, 0.3]}
    )

    with pytest.raises(
        sentiment.SentimentValidationError,
        match="unique by text_raw",
    ):
        sentiment.score_headlines(panel, title_score_cache=invalid_cache)


def test_standard_vader_receives_unchanged_case_punctuation_and_negation() -> None:
    panel = pd.DataFrame(
        {
            "title": ["This is GOOD!", "This is good.", "This is not good."],
            "text_raw": ["This is GOOD!", "This is good.", "This is not good."],
        }
    )
    cache = sentiment.score_distinct_titles(
        panel,
        analyzer=sentiment.get_vader_analyzer(),
    )
    scores = cache.set_index("text_raw")["vader_compound"]

    assert list(cache["text_raw"]) == list(panel["text_raw"])
    assert scores["This is GOOD!"] > scores["This is good."]
    assert scores["This is not good."] < 0
    metadata = sentiment.vader_metadata()
    assert metadata["text_input_column"] == "text_raw"
    assert metadata["text_preprocessing"] == "none"
    assert metadata["lexicon_extension"] == "none"


def test_weekend_headline_maps_forward_to_next_equity_trading_day() -> None:
    clean = etl.clean_news_headlines(
        pd.DataFrame(
            {
                "date": ["2023-01-06", "2023-01-07", "2023-01-09"],
                "ticker": ["AAA", "AAA", "AAA"],
                "sector": ["Tech", "Tech", "Tech"],
                "title": ["Friday", "Saturday", "Monday"],
                "url": ["https://example.com/1", "https://example.com/2", "https://example.com/3"],
                "publisher": ["Example", "Example", "Example"],
            }
        )
    )
    mapped = features.map_headlines_to_trading_days(
        clean,
        pd.to_datetime(["2023-01-06", "2023-01-09", "2023-01-10"]),
    )

    assert mapped["trading_date"].tolist() == list(
        pd.to_datetime(["2023-01-06", "2023-01-09", "2023-01-09"])
    )
    assert mapped["mapping_status"].tolist() == [
        "same_trading_day",
        "next_trading_day",
        "same_trading_day",
    ]


def test_ticker_day_then_equal_weight_sector_aggregation_and_coverage() -> None:
    date = pd.Timestamp("2023-01-03")
    headline_scores = pd.DataFrame(
        {
            "trading_date": [date, date, date, date],
            "ticker": ["AAA", "AAA", "AAA", "BBB"],
            "sector": ["Tech", "Tech", "Tech", "Tech"],
            "text_raw": ["A1", "A2", "A3", "B1"],
            "vader_compound": [0.9, 0.6, 0.3, -0.2],
        }
    )
    ticker_days = sentiment.aggregate_ticker_days(headline_scores)
    universe = pd.DataFrame(
        {"ticker": ["AAA", "BBB", "CCC"], "sector": ["Tech"] * 3}
    )
    index = sentiment.sector_sentiment_index(
        ticker_days,
        pd.to_datetime(["2023-01-03", "2023-01-04"]),
        universe,
    )

    ticker_scores = ticker_days.set_index("ticker")["ticker_day_compound"]
    assert ticker_scores["AAA"] == pytest.approx(0.6)
    assert ticker_scores["BBB"] == pytest.approx(-0.2)
    first = index.loc[index["date"].eq("2023-01-03")].iloc[0]
    second = index.loc[index["date"].eq("2023-01-04")].iloc[0]
    assert first["raw_sector_compound"] == pytest.approx(0.2)
    assert first["raw_sector_compound"] != pytest.approx(
        headline_scores["vader_compound"].mean()
    )
    assert first["observed_ticker_count"] == 2
    assert first["possible_ticker_count"] == 3
    assert first["ticker_coverage_share"] == pytest.approx(2 / 3)
    assert second["observed_ticker_count"] == 0
    assert second["ticker_coverage_share"] == 0
    assert np.isnan(second["raw_sector_compound"])


def test_tradable_signal_uses_prior_history_and_is_lagged_one_trading_day() -> None:
    dates = pd.bdate_range("2023-01-02", periods=5)
    ticker_days = pd.DataFrame(
        {
            "trading_date": dates[:4],
            "ticker": ["AAA"] * 4,
            "sector": ["Tech"] * 4,
            "ticker_day_compound": [0.0, 0.5, 1.0, -0.5],
            "headline_count": [1] * 4,
        }
    )
    universe = pd.DataFrame({"ticker": ["AAA"], "sector": ["Tech"]})
    index = sentiment.sector_sentiment_index(
        ticker_days,
        dates,
        universe,
        min_history=2,
        zscore_clip=2.0,
        signal_lag=1,
    ).set_index("date")

    assert np.isnan(index.loc[dates[2], "tradable_sector_zscore"])
    assert index.loc[dates[3], "raw_zscore_clipped"] == pytest.approx(-2.0)
    assert index.loc[dates[3], "tradable_sector_zscore"] == pytest.approx(2.0)
    assert index.loc[dates[3], "tradable_signal_source_date"] == dates[2]
    assert index.loc[dates[3], "signal_prior_observations"] == 2
    assert index.loc[dates[3], "tradable_signal_source_date"] < dates[3]


def test_zero_day_signal_lag_is_rejected() -> None:
    ticker_days = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "ticker": ["AAA", "AAA"],
            "sector": ["Tech", "Tech"],
            "ticker_day_compound": [0.1, 0.2],
            "headline_count": [1, 1],
        }
    )
    universe = pd.DataFrame({"ticker": ["AAA"], "sector": ["Tech"]})

    with pytest.raises(ValueError, match="at least one trading day"):
        sentiment.sector_sentiment_index(
            ticker_days,
            pd.to_datetime(["2023-01-03", "2023-01-04"]),
            universe,
            min_history=2,
            signal_lag=0,
        )


def _exploratory_sector_fixture(periods: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=periods)
    position = np.arange(periods)
    observed = np.where(position % 6 == 0, 0, position % 5 + 1)
    raw = np.sin(position / 7.0)
    raw = np.where(observed == 0, np.nan, raw)
    return pd.DataFrame(
        {
            "date": dates,
            "sector": "Technology",
            "raw_sector_compound": raw,
            "observed_ticker_count": observed,
            "possible_ticker_count": 5,
            "ticker_coverage_share": observed / 5.0,
            "has_observed_news": observed > 0,
        }
    )


def _direct_trailing_value(panel: pd.DataFrame, position: int) -> tuple[float, float]:
    prior = panel.iloc[max(0, position - 21) : position]
    observed = prior.loc[prior["has_observed_news"]]
    denominator = float(observed["ticker_coverage_share"].sum())
    trailing = (
        float(
            (
                observed["raw_sector_compound"]
                * observed["ticker_coverage_share"]
            ).sum()
            / denominator
        )
        if denominator > 0
        else np.nan
    )
    coverage = float(prior["observed_ticker_count"].sum()) / (21 * 5)
    return trailing, coverage


def test_exploratory_window_excludes_current_and_missing_days_are_not_zero() -> None:
    panel = _exploratory_sector_fixture()
    actual = sentiment.build_coverage_adjusted_trailing_signal(panel)
    position = 50
    expected_trailing, expected_coverage = _direct_trailing_value(panel, position)
    row = actual.iloc[position]

    assert row["signal_window_start_date"] == panel.iloc[position - 21]["date"]
    assert row["signal_window_end_date"] == panel.iloc[position - 1]["date"]
    assert row["signal_window_end_date"] < row["date"]
    assert row["latest_raw_news_date_used"] < row["date"]
    assert row["trailing_coverage_weighted_sentiment"] == pytest.approx(
        expected_trailing, abs=1e-12
    )
    assert row["effective_coverage"] == pytest.approx(
        expected_coverage, abs=1e-12
    )
    assert 0.0 <= row["effective_coverage"] <= 1.0

    current_and_future_changed = panel.copy()
    changed_rows = (
        (current_and_future_changed.index >= position)
        & current_and_future_changed["has_observed_news"]
    )
    current_and_future_changed.loc[changed_rows, "raw_sector_compound"] = 0.999
    changed = sentiment.build_coverage_adjusted_trailing_signal(
        current_and_future_changed
    )
    pd.testing.assert_frame_equal(
        actual.loc[:position, [
            "trailing_coverage_weighted_sentiment",
            "effective_coverage",
            "coverage_adjusted_zscore",
        ]],
        changed.loc[:position, [
            "trailing_coverage_weighted_sentiment",
            "effective_coverage",
            "coverage_adjusted_zscore",
        ]],
        check_exact=True,
    )


def test_exploratory_expanding_statistics_are_strictly_prior_with_ddof_one() -> None:
    panel = _exploratory_sector_fixture()
    actual = sentiment.build_coverage_adjusted_trailing_signal(panel)
    position = 75
    prior_trailing = np.asarray(
        [
            _direct_trailing_value(panel, index)[0]
            for index in range(1, position)
        ],
        dtype=float,
    )
    prior_trailing = prior_trailing[np.isfinite(prior_trailing)]
    current_trailing, current_coverage = _direct_trailing_value(panel, position)
    expected_mean = float(np.mean(prior_trailing))
    expected_std = float(np.std(prior_trailing, ddof=1))
    expected_raw_z = (current_trailing - expected_mean) / expected_std
    expected_clipped = float(np.clip(expected_raw_z, -2.0, 2.0))
    expected_adjusted = expected_clipped * np.sqrt(current_coverage)
    row = actual.iloc[position]

    assert row["expanding_prior_observations"] == len(prior_trailing)
    assert row["expanding_prior_mean"] == pytest.approx(expected_mean, abs=1e-12)
    assert row["expanding_prior_std"] == pytest.approx(expected_std, abs=1e-12)
    assert row["raw_trailing_zscore"] == pytest.approx(expected_raw_z, abs=1e-12)
    assert row["clipped_trailing_zscore"] == pytest.approx(
        expected_clipped, abs=1e-12
    )
    assert row["coverage_adjusted_zscore"] == pytest.approx(
        expected_adjusted, abs=1e-12
    )
    assert row["zscore_std_ddof"] == 1


def test_exploratory_parameters_are_locked_against_final_period_search() -> None:
    panel = _exploratory_sector_fixture()
    with pytest.raises(ValueError, match="locked at 21"):
        sentiment.build_coverage_adjusted_trailing_signal(panel, signal_window=20)
    with pytest.raises(ValueError, match="locked at 60"):
        sentiment.build_coverage_adjusted_trailing_signal(panel, min_history=59)
