"""Focused Phase 4 tests for student-labelled sentiment validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.app_artifacts import load_manual_validation_app_artifacts
from src.finbert_manual_validation import (
    DIAGNOSTIC_PURPOSE,
    REPRESENTATIVE_PURPOSE,
    ManualValidationError,
    evaluate_manual_review,
    import_student_review,
    join_review_to_audit,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLIND_PATH = PROJECT_ROOT / "results/tables/sentiment_manual_review_blind.csv"
AUDIT_PATH = PROJECT_ROOT / "results/tables/sentiment_manual_review_template.csv"
COMPLETED_PATH = PROJECT_ROOT / "results/tables/sentiment_manual_review_completed.csv"
METRICS_PATH = PROJECT_ROOT / "results/tables/sentiment_manual_validation_metrics.csv"
CONFUSION_PATH = PROJECT_ROOT / "results/tables/sentiment_manual_validation_confusion.csv"
JOINED_PATH = PROJECT_ROOT / "results/tables/sentiment_manual_validation_joined.csv"
ERROR_PATH = PROJECT_ROOT / "results/tables/sentiment_manual_error_analysis.csv"
METADATA_PATH = PROJECT_ROOT / "results/tables/sentiment_manual_validation_metadata.json"
FIGURE_PATH = PROJECT_ROOT / "results/figures/sentiment_manual_validation_comparison.png"


def _submitted_from_blind() -> pd.DataFrame:
    frame = pd.read_csv(BLIND_PATH, dtype=str, keep_default_na=False)
    labels = np.resize(np.array(["negative", "neutral", "positive"]), len(frame))
    frame["human_label"] = labels
    frame["human_confidence"] = "high"
    return frame


def test_import_restores_only_canonical_text_and_accepts_file_level_attestation() -> None:
    blind = pd.read_csv(BLIND_PATH, dtype=str, keep_default_na=False)
    submitted = _submitted_from_blind()
    submitted.loc[0, "text_raw"] = "spreadsheet-damaged text"
    imported = import_student_review(
        submitted,
        blind,
        single_student_reviewer_attested=True,
    )
    assert imported.restored_text_count == 1
    assert imported.normalised_confidence_whitespace_count == 0
    assert imported.completed["text_raw"].equals(blind["text_raw"])
    assert imported.completed["human_label"].equals(submitted["human_label"])
    assert imported.completed["human_confidence"].equals(submitted["human_confidence"])
    assert imported.completed["reviewed_by_student"].eq("").all()
    assert "sole authorship" in imported.reviewer_attestation


def test_import_rejects_invalid_student_fields_without_attestation() -> None:
    blind = pd.read_csv(BLIND_PATH, dtype=str, keep_default_na=False)
    submitted = _submitted_from_blind()
    submitted.loc[0, "human_label"] = "P"
    with pytest.raises(ManualValidationError, match="human_label"):
        import_student_review(
            submitted,
            blind,
            single_student_reviewer_attested=True,
        )
    submitted.loc[0, "human_label"] = "positive"
    with pytest.raises(ManualValidationError, match="attestation"):
        import_student_review(
            submitted,
            blind,
            single_student_reviewer_attested=False,
        )


def test_real_completed_review_joins_one_to_one_and_keeps_purposes_separate() -> None:
    if not COMPLETED_PATH.is_file():
        pytest.skip("Phase 4 completed review has not been imported yet")
    completed = pd.read_csv(COMPLETED_PATH, dtype=str, keep_default_na=False)
    audit = pd.read_csv(AUDIT_PATH, dtype=str, keep_default_na=False)
    joined = join_review_to_audit(completed, audit)
    assert len(joined) == 150
    assert joined["review_id"].is_unique
    assert joined["sample_purpose"].value_counts().to_dict() == {
        REPRESENTATIVE_PURPOSE: 100,
        DIAGNOSTIC_PURPOSE: 50,
    }
    assert joined["human_label"].notna().all()


def test_weighted_metrics_and_exact_paired_comparison_are_reproducible() -> None:
    if not COMPLETED_PATH.is_file():
        pytest.skip("Phase 4 completed review has not been imported yet")
    completed = pd.read_csv(COMPLETED_PATH, dtype=str, keep_default_na=False)
    audit = pd.read_csv(AUDIT_PATH, dtype=str, keep_default_na=False)
    joined = join_review_to_audit(completed, audit)
    first = evaluate_manual_review(joined)
    second = evaluate_manual_review(joined)
    assert first.metrics.equals(second.metrics)
    assert first.confusion.equals(second.confusion)
    assert len(first.error_analysis) == 50
    for model in ("VADER", "FinBERT"):
        selected = first.metrics.loc[first.metrics["model"].eq(model)]
        assert {
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        }.issubset(selected["metric"])
        accuracy = selected.loc[selected["metric"].eq("accuracy")].iloc[0]
        assert 0 <= accuracy["ci_low"] <= accuracy["value"] <= accuracy["ci_high"] <= 1
        confusion = first.confusion.loc[first.confusion["model"].eq(model)]
        assert len(confusion) == 9
        assert confusion["unweighted_count"].sum() == 100
    assert 0 <= first.paired_summary["mcnemar_exact_p_value"] <= 1
    assert set(first.error_analysis["error_category"]).issubset(
        {
            "uncategorised",
            "mixed positive and negative information",
            "entity/context ambiguity",
            "student comment supplied; category not assigned",
        }
    )


def test_published_phase4_artifacts_are_complete_and_app_readable() -> None:
    expected = (
        COMPLETED_PATH,
        METRICS_PATH,
        CONFUSION_PATH,
        JOINED_PATH,
        ERROR_PATH,
        METADATA_PATH,
        FIGURE_PATH,
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)
    joined = pd.read_csv(JOINED_PATH)
    errors = pd.read_csv(ERROR_PATH)
    assert len(joined) == 150
    assert len(errors) == 50
    assert errors["sample_purpose"].eq(DIAGNOSTIC_PURPOSE).all()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["phase_status"] == "PASS"
    assert metadata["restored_text_count"] == 2
    assert metadata["normalised_confidence_whitespace_count"] == 136
    assert (
        metadata["protected_artifact_hashes_before"]
        == metadata["protected_artifact_hashes_after"]
    )
    app_artifacts = load_manual_validation_app_artifacts(PROJECT_ROOT)
    assert app_artifacts is not None
    assert app_artifacts.metadata["representative_rows"] == 100
    assert set(app_artifacts.metrics["model"]) == {
        "VADER",
        "FinBERT",
        "Paired comparison",
    }
