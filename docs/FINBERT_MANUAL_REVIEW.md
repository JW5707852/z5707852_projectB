# FinBERT blind manual-review workflow

`results/tables/sentiment_manual_review_blind.csv` is the student-facing review
file. Its 150 unchanged titles are deterministically shuffled and expose no
model output, sampling purpose, sampling stratum, disagreement type, or sampling
weight. The student-owned fields must remain blank until the student performs
the review independently.

`results/tables/sentiment_manual_review_template.csv` remains the unblinded audit
and sampling record. It must not be replaced by the blind file. The pure
`audit_review_mapping()` function in `src/finbert_review.py` deterministically
recreates the same `review_id` values for the audit template. Completed blind
labels can therefore be joined back by `review_id` without matching on text or
revealing model outputs during review.

Only rows whose audit `sample_purpose` is `representative_evaluation` may later
support an overall performance estimate, using the recorded sampling design and
weights where appropriate. Rows marked `disagreement_enriched_diagnosis` remain
qualitative error-analysis evidence and must not be treated as a representative
accuracy sample.

Until completed student labels pass a separate validation workflow, accuracy,
F1, superiority, forecasting, and investment-suitability claims remain pending.
