"""Focused Phase 3 FinBERT loader, chart, blinding, and app tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from src.app_artifacts import (
    ARTIFACT_RELATIVE_PATHS,
    EXPECTED_FUND_IDENTITIES,
    EXPECTED_SECTORS,
    FINBERT_ARTIFACT_RELATIVE_PATHS,
    AppArtifactError,
    load_app_artifacts,
    load_finbert_app_artifacts,
    sentiment_model_history,
    validate_finbert_app_artifacts,
)
from src.app_charts import MODEL_COLORS, sentiment_model_comparison_figure
from src.finbert_review import (
    BLIND_REVIEW_COLUMNS,
    STUDENT_REVIEW_COLUMNS,
    audit_review_mapping,
    build_blind_review,
)
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "streamlit_app.py"
AUDIT_PATH = PROJECT_ROOT / "results/tables/sentiment_manual_review_template.csv"
BLIND_PATH = PROJECT_ROOT / "results/tables/sentiment_manual_review_blind.csv"


@pytest.fixture(scope="module")
def phase3_artifacts():
    core = load_app_artifacts(PROJECT_ROOT)
    finbert = load_finbert_app_artifacts(PROJECT_ROOT, core.sector_sentiment)
    return core, finbert


def test_optional_finbert_artifacts_validate_without_title_cache(
    phase3_artifacts,
) -> None:
    core, finbert = phase3_artifacts
    assert set(FINBERT_ARTIFACT_RELATIVE_PATHS) == {
        "sector_sentiment",
        "model_comparison",
        "disagreements",
        "metadata",
    }
    assert all(
        "finbert_title_scores" not in path.as_posix()
        for path in FINBERT_ARTIFACT_RELATIVE_PATHS.values()
    )
    assert len(finbert.sector_sentiment) == 10_060
    assert len(finbert.model_comparison) == 22
    assert len(finbert.disagreements) == 200
    assert set(finbert.sector_sentiment["sector"]) == set(EXPECTED_SECTORS)
    assert finbert.metadata["manual_review_status"] == "pending student review"
    assert finbert.manual_validation is not None
    assert finbert.manual_validation.metadata["phase_status"] == "PASS"

    history = sentiment_model_history(core.sector_sentiment, finbert.sector_sentiment)
    assert len(history) == 10_060
    assert not history.duplicated(["date", "sector"]).any()
    assert history["vader_score"].isna().equals(history["finbert_score"].isna())


def test_missing_or_malformed_finbert_artifacts_do_not_invalidate_core(
    tmp_path: Path,
    phase3_artifacts,
) -> None:
    for relative in ARTIFACT_RELATIVE_PATHS.values():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, target)
    core = load_app_artifacts(tmp_path)
    assert len(core.performance_metrics) == len(EXPECTED_FUND_IDENTITIES)
    with pytest.raises(AppArtifactError, match="FinBERT robustness artifacts are unavailable"):
        load_finbert_app_artifacts(tmp_path, core.sector_sentiment)

    _, finbert = phase3_artifacts
    invalid_comparison = finbert.model_comparison.copy()
    invalid_comparison.loc[0, "observation_unit"] = "mixed_unit"
    with pytest.raises(AppArtifactError, match="FinBERT robustness artifacts are unavailable"):
        validate_finbert_app_artifacts(
            finbert.sector_sentiment,
            invalid_comparison,
            finbert.disagreements,
            finbert.metadata,
            core.sector_sentiment,
        )
    assert len(core.fund_returns["fund"].unique()) == len(EXPECTED_FUND_IDENTITIES)


def test_model_comparison_chart_preserves_gaps_and_uses_stable_colours(
    phase3_artifacts,
) -> None:
    core, finbert = phase3_artifacts
    history = sentiment_model_history(core.sector_sentiment, finbert.sector_sentiment)
    selected = history.loc[history["sector"].eq("Tech")].copy()
    selected["display_vader"] = selected["vader_score"]
    selected["display_finbert"] = selected["finbert_score"]
    figure = sentiment_model_comparison_figure(selected, y_title="Daily sentiment score")
    assert [trace.name for trace in figure.data] == ["VADER", "FinBERT"]
    assert [trace.line.color for trace in figure.data] == [
        MODEL_COLORS["VADER"],
        MODEL_COLORS["FinBERT"],
    ]
    assert all(trace.connectgaps is False for trace in figure.data)
    assert figure.layout.hovermode == "x unified"
    assert figure.layout.xaxis.rangeslider.visible is True
    assert list(figure.layout.yaxis.range) == [-1.02, 1.02]


def test_blind_review_is_deterministic_complete_and_contains_no_model_leakage() -> None:
    audit = pd.read_csv(AUDIT_PATH, keep_default_na=False)
    expected = build_blind_review(audit)
    actual = pd.read_csv(BLIND_PATH, keep_default_na=False)
    assert actual.equals(expected)
    assert tuple(actual.columns) == BLIND_REVIEW_COLUMNS
    assert len(actual) == 150
    assert actual["review_id"].is_unique
    assert actual["text_raw"].is_unique
    assert set(actual["text_raw"]) == set(audit["text_raw"])
    assert actual[list(STUDENT_REVIEW_COLUMNS)].eq("").all().all()
    forbidden = {
        "vader_compound",
        "vader_label",
        "finbert_score",
        "finbert_label",
        "disagreement_type",
        "sample_purpose",
        "sampling_stratum",
        "sampling_weight",
    }
    assert forbidden.isdisjoint(actual.columns)

    mapping = audit_review_mapping(audit)
    joined = actual[["review_id"]].merge(
        mapping[["review_id", "sample_purpose"]],
        on="review_id",
        how="left",
        validate="one_to_one",
    )
    assert joined["sample_purpose"].value_counts().to_dict() == {
        "representative_evaluation": 100,
        "disagreement_enriched_diagnosis": 50,
    }


def test_phase3_streamlit_sentiment_view_and_runtime_dependency_boundary() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    app.segmented_control[0].set_value("Sector Sentiment").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Sector Overview",
        "Model Comparison",
    ]
    assert not any(
        subheader.value == "Sentiment model comparison" for subheader in app.subheader
    )
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'key="sentiment_model_tab"' in source
    assert 'on_change="rerun"' in source
    app.session_state["sentiment_model_tab"] = "Model Comparison"
    app.run()
    assert not app.exception
    assert any(
        subheader.value == "Sentiment model comparison" for subheader in app.subheader
    )
    metric_values = {metric.label: metric.value for metric in app.metric}
    assert metric_values["Sector-days compared"] == "9,832"
    assert metric_values["Pearson correlation"] == "0.312"
    assert metric_values["Same-label rate"] == "53.65%"
    metric_values = {metric.label: metric.value for metric in app.metric}
    assert metric_values["VADER weighted accuracy"] == "46.82%"
    assert metric_values["FinBERT weighted accuracy"] == "53.07%"
    assert metric_values["Exact McNemar p-value"] == "0.377"
    assert any("not evidence of return predictability" in caption.value for caption in app.caption)
    assert any(
        "only 32 paired headlines" in warning.value
        and "Statistical power is therefore limited" in warning.value
        for warning in app.warning
    )
    assert any("not an accuracy sample" in caption.value for caption in app.caption)
    assert app.selectbox[0].label == "Sector"
    assert app.date_input[0].label == "Date range"
    comparison_radio = next(
        item for item in app.radio if item.key == "finbert_comparison_series"
    )
    comparison_radio.set_value("Daily")
    app.session_state["sentiment_model_tab"] = "Model Comparison"
    app.run()
    assert not app.exception
    assert any("appear as gaps" in caption.value for caption in app.caption)

    source = APP_PATH.read_text(encoding="utf-8").lower()
    artifact_source = (PROJECT_ROOT / "src/app_artifacts.py").read_text(encoding="utf-8").lower()
    assert "finbert_title_scores.csv" not in source + artifact_source
    deployment = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for build_only in ("torch", "transformers", "huggingface-hub", "nltk"):
        assert build_only not in deployment
