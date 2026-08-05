# Student-labelled VADER--FinBERT validation

## Scope

Phase 4 validates headline classification only. It does not test return
prediction, portfolio performance, or investment suitability. The 100-row
representative sample supplies performance estimates; the separate 50-row
disagreement-enriched sample is excluded from those estimates.

## Student evidence and import policy

The student independently labelled all 150 blinded headlines as positive,
negative, or neutral and supplied high, medium, or low confidence. Because this
is a single-student project, the student explicitly chose file-level sole-reviewer
attestation instead of repeating the same identifier in 150 rows. The row-level
`reviewed_by_student` field therefore remains blank by documented student choice.

Excel changed two source strings and added trailing whitespace to 136 confidence
cells. The import restores `text_raw` from the canonical blind file by
`review_id` and removes confidence-field boundary whitespace. It does not infer,
replace, or complete any student label, confidence category, or comment.

## Metrics

Representative metrics use the recorded sector-year sampling weights. Reported
outputs include weighted accuracy, balanced accuracy, macro precision, macro
recall, macro F1, class-specific precision and recall, prediction rates, and
weighted confusion matrices. Weighted accuracy receives an approximate 95%
Wilson interval using Kish effective sample size.

The exact McNemar test uses the two models' paired correct/incorrect outcomes on
the same 100 representative headlines. It is unweighted and should be read with
the small-sample and sampling-design limitations.

## Observed results

- VADER weighted accuracy: 46.82%.
- FinBERT weighted accuracy: 53.07%.
- Weighted point-estimate difference: 6.25 percentage points in FinBERT's favour.
- Exact McNemar p-value: 0.377.
- VADER macro F1: 44.62%.
- FinBERT macro F1: 47.59%.

FinBERT has higher point estimates, but the paired test does not provide strong
evidence that the models have different classification error rates in this small
review sample. FinBERT also predicts neutral much more frequently and has high
neutral recall but low positive and negative recall. The result supports a model
risk and robustness interpretation, not an automatic superiority claim.

## Diagnostic limitation

Only four of 150 rows contain a student comment. Error categories are therefore
assigned only when the student's words support them; all other diagnostic rows
remain explicitly uncategorised. No AI-generated explanation is substituted for
missing student reasoning.
