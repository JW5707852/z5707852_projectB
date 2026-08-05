"""Student-labelled validation for the VADER--FinBERT robustness layer.

The functions in this module keep the representative evaluation sample
separate from the disagreement-enriched diagnostic sample.  Student labels are
never inferred or rewritten.  The only permitted text repair restores
``text_raw`` from the canonical blind-review file by ``review_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.finbert_review import BLIND_REVIEW_COLUMNS, audit_review_mapping

LABELS: Final[tuple[str, ...]] = ("negative", "neutral", "positive")
CONFIDENCE_LEVELS: Final[tuple[str, ...]] = ("low", "medium", "high")
REPRESENTATIVE_PURPOSE = "representative_evaluation"
DIAGNOSTIC_PURPOSE = "disagreement_enriched_diagnosis"
REVIEW_ROWS = 150
REPRESENTATIVE_ROWS = 100
DIAGNOSTIC_ROWS = 50
WILSON_Z_95 = 1.959963984540054


class ManualValidationError(RuntimeError):
    """Raised when student review evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class ReviewImport:
    """Canonical completed review plus non-judgemental import evidence."""

    completed: pd.DataFrame
    restored_text_count: int
    normalised_confidence_whitespace_count: int
    reviewer_attestation: str


@dataclass(frozen=True)
class ValidationResults:
    """All deterministic Phase 4 analytical outputs."""

    joined: pd.DataFrame
    metrics: pd.DataFrame
    confusion: pd.DataFrame
    error_analysis: pd.DataFrame
    paired_summary: dict[str, float | int]


def import_student_review(
    submitted: pd.DataFrame,
    canonical_blind: pd.DataFrame,
    *,
    single_student_reviewer_attested: bool,
) -> ReviewImport:
    """Validate labels and restore source text without changing student fields."""
    if list(submitted.columns) != list(BLIND_REVIEW_COLUMNS):
        raise ManualValidationError(
            "submitted review columns differ from the blind-review contract"
        )
    if list(canonical_blind.columns) != list(BLIND_REVIEW_COLUMNS):
        raise ManualValidationError("canonical blind-review columns are invalid")
    if len(submitted) != REVIEW_ROWS or len(canonical_blind) != REVIEW_ROWS:
        raise ManualValidationError("completed and canonical reviews must have 150 rows")
    for name, frame in (("submitted", submitted), ("canonical", canonical_blind)):
        if frame["review_id"].isna().any() or frame["review_id"].duplicated().any():
            raise ManualValidationError(f"{name} review IDs are missing or duplicated")
        if frame["text_raw"].isna().any():
            raise ManualValidationError(f"{name} review contains missing text_raw")
    if set(submitted["review_id"]) != set(canonical_blind["review_id"]):
        raise ManualValidationError("submitted review IDs differ from the blind file")

    student_fields = submitted[
        [
            "review_id",
            "human_label",
            "human_confidence",
            "reviewer_comment",
            "reviewed_by_student",
        ]
    ].copy()
    for column in ("human_label", "human_confidence"):
        if student_fields[column].isna().any():
            raise ManualValidationError(f"{column} contains missing values")
    if not student_fields["human_label"].eq(
        student_fields["human_label"].str.strip()
    ).all():
        raise ManualValidationError("human_label contains surrounding whitespace")
    confidence_stripped = student_fields["human_confidence"].str.strip()
    normalised_confidence_whitespace_count = int(
        student_fields["human_confidence"].ne(confidence_stripped).sum()
    )
    student_fields["human_confidence"] = confidence_stripped
    if set(student_fields["human_label"]) - set(LABELS):
        raise ManualValidationError("human_label must use positive, negative, or neutral")
    if set(student_fields["human_confidence"]) - set(CONFIDENCE_LEVELS):
        raise ManualValidationError("human_confidence must use high, medium, or low")
    if student_fields["human_label"].eq("").any():
        raise ManualValidationError("human_label contains blank values")
    if student_fields["human_confidence"].eq("").any():
        raise ManualValidationError("human_confidence contains blank values")

    reviewers = student_fields["reviewed_by_student"].fillna("").str.strip()
    if reviewers.eq("").any() and not single_student_reviewer_attested:
        raise ManualValidationError(
            "blank row-level reviewer values require single-student attestation"
        )
    reviewer_attestation = (
        "Single-student project: the student explicitly confirmed sole authorship "
        "of all manual labels; repeated row-level reviewer entries were intentionally "
        "left blank."
        if reviewers.eq("").all()
        else "Row-level reviewer identifiers supplied by the student."
    )

    canonical = canonical_blind[["review_id", "text_raw"]].copy()
    submitted_text = submitted[["review_id", "text_raw"]].rename(
        columns={"text_raw": "submitted_text_raw"}
    )
    text_check = canonical.merge(
        submitted_text,
        on="review_id",
        how="inner",
        validate="one_to_one",
    )
    restored_text_count = int(
        text_check["text_raw"].ne(text_check["submitted_text_raw"]).sum()
    )
    completed = canonical.merge(
        student_fields,
        on="review_id",
        how="inner",
        validate="one_to_one",
    )
    completed = completed[list(BLIND_REVIEW_COLUMNS)]
    if completed["human_label"].isna().any() or completed["human_confidence"].isna().any():
        raise ManualValidationError("canonical review import lost student fields")
    return ReviewImport(
        completed,
        restored_text_count,
        normalised_confidence_whitespace_count,
        reviewer_attestation,
    )


def join_review_to_audit(
    completed: pd.DataFrame,
    audit_template: pd.DataFrame,
) -> pd.DataFrame:
    """Join completed blind labels to the hidden audit design by review_id only."""
    mapping = audit_review_mapping(audit_template)
    canonical = mapping[["review_id", "text_raw"]]
    if not canonical.equals(completed[["review_id", "text_raw"]]):
        raise ManualValidationError("completed review no longer matches canonical text")

    placeholders = {
        "student_label",
        "student_confidence",
        "student_notes",
        "student_reviewed_by",
    }
    audit_columns = [column for column in mapping.columns if column not in placeholders]
    joined = mapping[audit_columns].merge(
        completed.drop(columns="text_raw"),
        on="review_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != REVIEW_ROWS:
        raise ManualValidationError("review-to-audit join did not produce 150 rows")
    purposes = joined["sample_purpose"].value_counts().to_dict()
    expected = {REPRESENTATIVE_PURPOSE: REPRESENTATIVE_ROWS, DIAGNOSTIC_PURPOSE: DIAGNOSTIC_ROWS}
    if purposes != expected:
        raise ManualValidationError(f"review purposes differ from {expected}: {purposes}")
    if joined["human_label"].isna().any():
        raise ManualValidationError("review-to-audit join lost student labels")
    return joined


def _weighted_confusion(
    truth: pd.Series,
    prediction: pd.Series,
    weights: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    unweighted = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    weighted = np.zeros((len(LABELS), len(LABELS)), dtype=float)
    label_index = {label: index for index, label in enumerate(LABELS)}
    for actual, predicted, weight in zip(truth, prediction, weights, strict=True):
        row = label_index[str(actual)]
        column = label_index[str(predicted)]
        unweighted[row, column] += 1
        weighted[row, column] += float(weight)
    return unweighted, weighted


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _wilson_interval(proportion: float, effective_n: float) -> tuple[float, float]:
    if effective_n <= 0:
        return np.nan, np.nan
    z = WILSON_Z_95
    denominator = 1.0 + z**2 / effective_n
    centre = (proportion + z**2 / (2.0 * effective_n)) / denominator
    radius = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / effective_n
            + z**2 / (4.0 * effective_n**2)
        )
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _model_results(
    representative: pd.DataFrame,
    *,
    model: str,
    prediction_column: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    truth = representative["human_label"].astype(str)
    prediction = representative[prediction_column].astype(str)
    weights = pd.to_numeric(representative["sampling_weight"], errors="raise")
    if (weights <= 0).any() or not np.isfinite(weights).all():
        raise ManualValidationError("representative sampling weights must be positive")
    if set(truth) - set(LABELS) or set(prediction) - set(LABELS):
        raise ManualValidationError("model or human labels are outside the approved classes")
    counts, weighted = _weighted_confusion(truth, prediction, weights)
    total_weight = float(weighted.sum())
    effective_n = float(weights.sum() ** 2 / np.square(weights).sum())
    accuracy = _divide(float(np.trace(weighted)), total_weight)

    recalls = np.array(
        [_divide(weighted[i, i], weighted[i, :].sum()) for i in range(len(LABELS))]
    )
    precisions = np.array(
        [_divide(weighted[i, i], weighted[:, i].sum()) for i in range(len(LABELS))]
    )
    f1 = np.array(
        [
            _divide(2.0 * precision * recall, precision + recall)
            for precision, recall in zip(precisions, recalls, strict=True)
        ]
    )
    predicted_rates = weighted.sum(axis=0) / total_weight
    ci_low, ci_high = _wilson_interval(accuracy, effective_n)

    values = {
        "accuracy": accuracy,
        "balanced_accuracy": float(recalls.mean()),
        "macro_precision": float(precisions.mean()),
        "macro_recall": float(recalls.mean()),
        "macro_f1": float(f1.mean()),
    }
    for index, label in enumerate(LABELS):
        values[f"precision_{label}"] = float(precisions[index])
        values[f"recall_{label}"] = float(recalls[index])
        values[f"predicted_{label}_rate"] = float(predicted_rates[index])

    metric_rows: list[dict[str, object]] = []
    for metric, value in values.items():
        is_accuracy = metric == "accuracy"
        metric_rows.append(
            {
                "model": model,
                "metric": metric,
                "value": value,
                "ci_low": ci_low if is_accuracy else np.nan,
                "ci_high": ci_high if is_accuracy else np.nan,
                "ci_method": (
                    "Approximate 95% Wilson interval using Kish effective sample size"
                    if is_accuracy
                    else "not reported"
                ),
                "sample_purpose": REPRESENTATIVE_PURPOSE,
                "sample_count": len(representative),
                "sampling_weight_sum": total_weight,
                "effective_sample_size": effective_n,
            }
        )

    confusion_rows: list[dict[str, object]] = []
    for row_index, actual in enumerate(LABELS):
        actual_weight = float(weighted[row_index, :].sum())
        for column_index, predicted in enumerate(LABELS):
            confusion_rows.append(
                {
                    "model": model,
                    "human_label": actual,
                    "predicted_label": predicted,
                    "unweighted_count": int(counts[row_index, column_index]),
                    "weighted_count": float(weighted[row_index, column_index]),
                    "weighted_sample_share": _divide(
                        float(weighted[row_index, column_index]), total_weight
                    ),
                    "weighted_within_human_label_share": _divide(
                        float(weighted[row_index, column_index]), actual_weight
                    ),
                    "sample_purpose": REPRESENTATIVE_PURPOSE,
                }
            )
    return metric_rows, confusion_rows, values


def _student_supported_category(comment: str) -> tuple[str, str]:
    value = str(comment).strip().lower()
    if not value:
        return "uncategorised", "No student comment; no error category inferred."
    if (
        "both side" in value
        or ("negative" in value and ("positive" in value or "postive" in value))
        or "worry" in value
    ):
        return "mixed positive and negative information", "Student comment"
    if "irrelevant" in value or "esg" in value:
        return "entity/context ambiguity", "Student comment"
    return "student comment supplied; category not assigned", "Student comment"


def _diagnostic_error_analysis(diagnostic: pd.DataFrame) -> pd.DataFrame:
    result = diagnostic.copy()
    result["vader_correct"] = result["vader_label"].eq(result["human_label"])
    result["finbert_correct"] = result["finbert_label"].eq(result["human_label"])
    result["model_outcome"] = np.select(
        [
            result["vader_correct"] & result["finbert_correct"],
            result["vader_correct"] & ~result["finbert_correct"],
            ~result["vader_correct"] & result["finbert_correct"],
        ],
        ["both_correct", "vader_only_correct", "finbert_only_correct"],
        default="both_wrong",
    )
    categories = result["reviewer_comment"].map(_student_supported_category)
    result["error_category"] = categories.map(lambda item: item[0])
    result["category_basis"] = categories.map(lambda item: item[1])
    columns = [
        "review_id",
        "text_raw",
        "human_label",
        "human_confidence",
        "reviewer_comment",
        "vader_label",
        "finbert_label",
        "vader_correct",
        "finbert_correct",
        "model_outcome",
        "error_category",
        "category_basis",
        "sector",
        "year",
        "sample_purpose",
    ]
    return result[columns].sort_values("review_id", kind="mergesort").reset_index(drop=True)


def evaluate_manual_review(joined: pd.DataFrame) -> ValidationResults:
    """Calculate design-aware representative metrics and separate diagnostics."""
    representative = joined.loc[
        joined["sample_purpose"].eq(REPRESENTATIVE_PURPOSE)
    ].copy()
    diagnostic = joined.loc[joined["sample_purpose"].eq(DIAGNOSTIC_PURPOSE)].copy()
    if len(representative) != REPRESENTATIVE_ROWS or len(diagnostic) != DIAGNOSTIC_ROWS:
        raise ManualValidationError("manual-review purpose separation is invalid")

    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, float]] = {}
    for model, prediction_column in (("VADER", "vader_label"), ("FinBERT", "finbert_label")):
        model_metrics, model_confusion, summary = _model_results(
            representative,
            model=model,
            prediction_column=prediction_column,
        )
        metric_rows.extend(model_metrics)
        confusion_rows.extend(model_confusion)
        summaries[model] = summary

    human = representative["human_label"]
    vader_correct = representative["vader_label"].eq(human)
    finbert_correct = representative["finbert_label"].eq(human)
    vader_only = int((vader_correct & ~finbert_correct).sum())
    finbert_only = int((~vader_correct & finbert_correct).sum())
    both_correct = int((vader_correct & finbert_correct).sum())
    both_wrong = int((~vader_correct & ~finbert_correct).sum())
    discordant = vader_only + finbert_only
    mcnemar_p = (
        float(binomtest(vader_only, discordant, 0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    weights = pd.to_numeric(representative["sampling_weight"], errors="raise")
    weighted_difference = summaries["FinBERT"]["accuracy"] - summaries["VADER"]["accuracy"]
    paired_summary: dict[str, float | int] = {
        "both_correct": both_correct,
        "vader_only_correct": vader_only,
        "finbert_only_correct": finbert_only,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "mcnemar_exact_p_value": mcnemar_p,
        "weighted_accuracy_difference_finbert_minus_vader": weighted_difference,
    }
    paired_metric_values = {
        **paired_summary,
        "representative_weight_sum": float(weights.sum()),
    }
    for metric, value in paired_metric_values.items():
        metric_rows.append(
            {
                "model": "Paired comparison",
                "metric": metric,
                "value": value,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "ci_method": "not reported",
                "sample_purpose": REPRESENTATIVE_PURPOSE,
                "sample_count": len(representative),
                "sampling_weight_sum": float(weights.sum()),
                "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
            }
        )
    metrics = pd.DataFrame.from_records(metric_rows)
    confusion = pd.DataFrame.from_records(confusion_rows)
    errors = _diagnostic_error_analysis(diagnostic)
    return ValidationResults(joined, metrics, confusion, errors, paired_summary)


__all__ = [
    "CONFIDENCE_LEVELS",
    "DIAGNOSTIC_PURPOSE",
    "LABELS",
    "REPRESENTATIVE_PURPOSE",
    "ManualValidationError",
    "ReviewImport",
    "ValidationResults",
    "evaluate_manual_review",
    "import_student_review",
    "join_review_to_audit",
]
