"""Deterministic blinding for the student-owned FinBERT manual review."""

from __future__ import annotations

import hashlib

import pandas as pd

BLIND_REVIEW_SEED = 5545
BLIND_REVIEW_COLUMNS = (
    "review_id",
    "text_raw",
    "human_label",
    "human_confidence",
    "reviewer_comment",
    "reviewed_by_student",
)
STUDENT_REVIEW_COLUMNS = (
    "human_label",
    "human_confidence",
    "reviewer_comment",
    "reviewed_by_student",
)


class BlindReviewError(RuntimeError):
    """Raised when the audit template cannot produce a genuine blind review."""


def _review_key(text: str, seed: int) -> str:
    payload = f"{seed}\0{text}".encode()
    return hashlib.sha256(payload).hexdigest()


def audit_review_mapping(
    audit_template: pd.DataFrame,
    *,
    seed: int = BLIND_REVIEW_SEED,
) -> pd.DataFrame:
    """Attach reproducible blind IDs and shuffle order to the audit template."""
    if "text_raw" not in audit_template:
        raise BlindReviewError("manual-review audit template is missing text_raw")
    mapping = audit_template.copy()
    if len(mapping) != 150:
        raise BlindReviewError(f"expected 150 review titles, found {len(mapping)}")
    if (
        mapping["text_raw"].isna().any()
        or not mapping["text_raw"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise BlindReviewError("manual-review titles must be non-empty strings")
    if mapping["text_raw"].duplicated().any():
        raise BlindReviewError("manual-review titles must be unique")
    mapping["_blind_order"] = mapping["text_raw"].map(lambda text: _review_key(text, seed))
    mapping = mapping.sort_values("_blind_order", kind="mergesort").reset_index(drop=True)
    mapping.insert(
        0,
        "review_id",
        [f"FINBERT-{position:03d}" for position in range(1, len(mapping) + 1)],
    )
    return mapping.drop(columns="_blind_order")


def build_blind_review(
    audit_template: pd.DataFrame,
    *,
    seed: int = BLIND_REVIEW_SEED,
) -> pd.DataFrame:
    """Return only blinded titles, stable IDs, and blank student-owned fields."""
    mapping = audit_review_mapping(audit_template, seed=seed)
    blind = mapping[["review_id", "text_raw"]].copy()
    for column in STUDENT_REVIEW_COLUMNS:
        blind[column] = ""
    return blind[list(BLIND_REVIEW_COLUMNS)]


__all__ = [
    "BLIND_REVIEW_COLUMNS",
    "BLIND_REVIEW_SEED",
    "STUDENT_REVIEW_COLUMNS",
    "BlindReviewError",
    "audit_review_mapping",
    "build_blind_review",
]
