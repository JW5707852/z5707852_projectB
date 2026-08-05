"""Focused pure-logic tests for the Phase 2 FinBERT artifact build."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src import finbert_phase2


def _score_cache(titles: list[str]) -> pd.DataFrame:
    rows = []
    for index, title in enumerate(titles):
        if index % 3 == 0:
            probabilities = (0.8, 0.1, 0.1)
            label = "positive"
        elif index % 3 == 1:
            probabilities = (0.1, 0.8, 0.1)
            label = "negative"
        else:
            probabilities = (0.1, 0.1, 0.8)
            label = "neutral"
        rows.append(
            {
                "text_raw": title,
                "probability_positive": probabilities[0],
                "probability_negative": probabilities[1],
                "probability_neutral": probabilities[2],
                "finbert_score": probabilities[0] - probabilities[1],
                "finbert_label": label,
            }
        )
    return pd.DataFrame(rows)


def test_join_preserves_clean_rows_and_infers_duplicate_title_once() -> None:
    titles = ["Profit rises", "Profit rises", "Loss widens"]
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-02"]),
            "trading_date": pd.to_datetime(["2023-01-03"] * 3),
            "ticker": ["AAA", "AAA", "BBB"],
            "sector": ["Tech", "Tech", "Energy"],
            "title": titles,
            "text_raw": titles,
        }
    )
    scored, audit = finbert_phase2.join_title_scores(
        panel, _score_cache(["Profit rises", "Loss widens"])
    )
    assert len(scored) == len(panel)
    assert audit == {
        "clean_rows_before_join": 3,
        "rows_after_join": 3,
        "matched_rows": 3,
        "unmatched_rows": 0,
        "multiplied_rows": 0,
        "outside_equity_sample_rows": 0,
    }
    assert scored["text_raw"].tolist() == titles


def test_sector_index_equal_weights_ticker_days_and_keeps_no_news_missing() -> None:
    ticker_days = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2023-01-03"] * 2),
            "ticker": ["AAA", "BBB"],
            "sector": ["Tech", "Tech"],
            "ticker_day_finbert": [0.8, -0.2],
            "ticker_day_probability_positive": [0.85, 0.10],
            "ticker_day_probability_negative": [0.05, 0.30],
            "ticker_day_probability_neutral": [0.10, 0.60],
            "headline_count": [10, 1],
        }
    )
    calendar = pd.to_datetime(["2023-01-03", "2023-01-04"])
    universe = pd.DataFrame({"ticker": ["AAA", "BBB"], "sector": ["Tech", "Tech"]})
    actual = finbert_phase2.sector_sentiment_index(ticker_days, calendar, universe)
    first = actual.iloc[0]
    second = actual.iloc[1]
    assert first["raw_sector_finbert"] == pytest.approx(0.3)
    assert first["raw_sector_finbert"] != (10 * 0.8 - 0.2) / 11
    assert first["observed_ticker_count"] == 2
    assert first["headline_count"] == 11
    assert bool(first["has_observed_news"])
    assert pd.isna(second["raw_sector_finbert"])
    assert second["observed_ticker_count"] == 0
    assert not bool(second["has_observed_news"])


def test_comparison_uses_only_matched_date_sector_rows_and_reports_counts() -> None:
    headline = pd.DataFrame(
        {
            "sector": ["Tech", "Tech", "Energy"],
            "vader_compound": [0.5, -0.5, 0.0],
            "finbert_score": [0.7, -0.6, 0.1],
            "finbert_label": ["positive", "negative", "neutral"],
        }
    )
    vader = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-03"]),
            "sector": ["Tech", "Tech", "Energy"],
            "raw_sector_compound": [0.2, np.nan, -0.1],
        }
    )
    finbert = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]),
            "sector": ["Tech", "Tech", "Energy"],
            "raw_sector_finbert": [0.3, 0.2, -0.2],
            "sector_probability_positive": [0.6, 0.5, 0.1],
            "sector_probability_negative": [0.2, 0.2, 0.7],
            "sector_probability_neutral": [0.2, 0.3, 0.2],
        }
    )
    table, matched = finbert_phase2.model_comparison_table(headline, vader, finbert)
    assert len(matched) == 1
    row = table.loc[
        table["observation_unit"].eq("matched_date_sector") & table["sector"].eq("All")
    ].iloc[0]
    assert row["paired_observation_count"] == 1
    assert "not accuracy" in row["interpretation_status"]
    headline_rows = table["observation_unit"].eq("clean_headline_row")
    assert table.loc[headline_rows, "pearson_correlation"].isna().all()
    assert table.loc[headline_rows, "spearman_correlation"].isna().all()
    correlation_rows = table["pearson_correlation"].notna()
    assert table.loc[correlation_rows, "observation_unit"].eq("matched_date_sector").all()


def test_coverage_reconciliation_ignores_datetime_storage_resolution_only() -> None:
    base = pd.DataFrame(
        {
            "date": pd.Series(["2023-01-03"], dtype="datetime64[us]"),
            "sector": ["Tech"],
            "observed_ticker_count": [2],
            "headline_count": [5],
            "possible_ticker_count": [3],
            "ticker_coverage_share": [2 / 3],
            "has_observed_news": [True],
        }
    )
    same = base.copy()
    same["date"] = same["date"].astype("datetime64[ns]")
    assert finbert_phase2.validate_coverage_reconciliation(base, same) == {
        "sector_grid_rows_compared": 1,
        "coverage_mismatch_count": 0,
    }
    changed = same.copy()
    changed.loc[0, "headline_count"] = 6
    with np.testing.assert_raises_regex(finbert_phase2.FinBERTValidationError, "headline_count"):
        finbert_phase2.validate_coverage_reconciliation(base, changed)


def _sampling_population(size: int = 90) -> pd.DataFrame:
    phrases = (
        "not expected to improve",
        "dividend guidance revised",
        "revenue rises 20%",
        "quarterly earnings announcement",
        "ordinary update",
        "debt outlook changes",
    )
    records = []
    for index in range(size):
        text = f"{phrases[index % len(phrases)]} case {index}"
        finbert_label = "negative" if index % 2 == 0 else "positive"
        if finbert_label == "negative":
            probabilities = (0.1, 0.8, 0.1)
            finbert_score = -0.7
            vader = 0.6
        else:
            probabilities = (0.8, 0.1, 0.1)
            finbert_score = 0.7
            vader = -0.6
        records.append(
            {
                "date": pd.Timestamp(2022 + index % 2, 1 + index % 12, 1),
                "trading_date": pd.Timestamp(2022 + index % 2, 1 + index % 12, 3),
                "ticker": f"T{index % 8}",
                "sector": "Tech" if index % 2 else "Energy",
                "title": text,
                "text_raw": text,
                "vader_compound": vader,
                "probability_positive": probabilities[0],
                "probability_negative": probabilities[1],
                "probability_neutral": probabilities[2],
                "finbert_score": finbert_score,
                "finbert_label": finbert_label,
            }
        )
    return pd.DataFrame.from_records(records)


def test_manual_review_separates_purposes_weights_and_blank_student_fields() -> None:
    review = finbert_phase2.manual_review_template(
        _sampling_population(), representative_size=20, diagnostic_size=12, seed=5545
    )
    counts = review["sample_purpose"].value_counts().to_dict()
    assert counts == {
        "representative_evaluation": 20,
        "disagreement_enriched_diagnosis": 12,
    }
    representative = review.loc[review["sample_purpose"].eq("representative_evaluation")]
    diagnostic = review.loc[review["sample_purpose"].eq("disagreement_enriched_diagnosis")]
    assert representative["sampling_weight"].notna().all()
    assert diagnostic["sampling_weight"].isna().all()
    assert review[list(finbert_phase2.STUDENT_REVIEW_FIELDS)].eq("").all().all()
    assert review["validation_status"].eq("pending student review").all()
    assert not review["text_raw"].duplicated().any()
