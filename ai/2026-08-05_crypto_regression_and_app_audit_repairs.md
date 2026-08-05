# Prompt log - crypto regression and app-audit repairs

## What I wanted

Repair the audit findings introduced after the crypto-only fund family was
published, without changing the report or rebuilding fund results.

## Prompt(s)

"Please solve the identified project bugs and logic gaps; do not work on the
report."

## What the assistant produced

- Replaced stale fund-count, row-count, and 252-only test assumptions with the
  published fund contract and per-family annualisation.
- Added a FinBERT integrity-metadata refresh mode that first validates existing
  Phase 2 files against their manifest, then updates only protected-core hashes.
- Updated the sentiment tabs to render only the selected view and gave dynamic
  allocation headings stable anchors.

## What was wrong or risky

The crypto extension left historical test and FinBERT hash baselines stale. This
could make a valid artifact set fail CI, while also leaving an obsolete
"protected artifacts unchanged" claim in the FinBERT metadata. The full raw-data
rebuild remains unverified in this environment because both supplied data hosts
failed DNS resolution; changing the frozen data helper would not be a valid fix.

## What I changed and why

The automated repair refreshed only hash evidence after validating all existing
FinBERT derived files. It did not rerun inference or alter title scores, sector
scores, fund CSVs, weights, metrics, or report evidence. I still need to rerun
the complete build and full test suite with access to the supplied data bundle
before relying on this as final submission evidence.
