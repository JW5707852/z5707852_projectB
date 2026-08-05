# Prompt log - FinBERT Streamlit integration

## What I wanted

Integrate the validated Phase 2 VADER–FinBERT robustness evidence into the
existing PortFoYou sentiment journey without changing any fund, running either
sentiment model, or treating FinBERT as superior or predictive.

## Prompt(s)

The student approved Phase 3 and required an optional “Neural Sentiment
Robustness” experience inside the existing sentiment page. It had to load only
the precomputed sector index, comparison table, controlled disagreement sample,
and metadata; preserve the VADER and five-fund experience when any innovation
artifact is unavailable; provide model overview, sector time-series, sector
model-risk, disagreement, and pending-validation views; and create a genuinely
blinded 150-title review file.

## What the assistant produced

- Added an optional `FinBERTAppArtifacts` contract and project-relative loaders
  in `src/app_artifacts.py`.
- Validated schemas, dates, sectors, date-sector uniqueness, probability and score
  arithmetic, observation units, paired counts, disagreement fields, metadata,
  and exact coverage consistency with the existing VADER grid.
- Kept the four existing required app artifacts as the only app-wide dependency.
  Optional FinBERT loading occurs only on the sentiment page and is caught
  separately, so a failure leaves all funds and VADER usable.
- Added a reusable Plotly VADER–FinBERT overlay chart with stable model colours,
  `connectgaps=False`, a date range slider, a neutral baseline, and no file I/O.
- Added the “Neural Sentiment Robustness” tab with dynamically loaded headline
  and matched-sector metrics, sector/date/smoothing controls, a matched sector
  table, controlled disagreement filters, and a dynamic pending-review notice.
- Added deterministic blinding code, a standalone generator, the 150-row blind
  CSV, and documentation for recreating `review_id` on the unblinded audit file.
- Left the Phase 2 audit template unchanged. Its SHA-256 still matches the Phase
  2 manifest.

## What was wrong or risky

- Loading optional innovation files at app startup would have made every fund
  page depend on FinBERT. The loader is invoked only inside the existing
  sentiment navigation branch and its failure is isolated.
- Headline label agreement and matched sector-day correlations use different
  observation units. The interface separates their metric groups and the loader
  rejects unexpected observation-unit contracts.
- Rolling means can visually fill a no-news date even without imputing the raw
  value. The display mean is masked wherever the current raw observation is
  missing, and Plotly is instructed not to connect gaps.
- The controlled disagreement file contains URLs and sampling metadata that are
  not needed by investors. The explorer exposes only the approved headline,
  model, sector, date, ticker, and disagreement fields.
- A blind file that retained sample purpose, weights, or model outputs would bias
  review. The generated file has exactly six columns: review ID, unchanged text,
  and four blank student-owned fields.

## Verification

- Focused Ruff command over the Phase 3 source, app, scripts, and tests: exit 0,
  all checks passed.
- Focused app command over `test_app_artifacts.py`, `test_app_charts.py`, and
  `test_finbert_app.py`: exit 0, 18 tests passed.
- Focused Phase 2 artifact-protection tests: exit 0, 3 tests passed.
- Streamlit `AppTest` exercised the full existing investor journey and the new
  sentiment controls without runtime exceptions.
- The four protected hashes and all approved Phase 2 input hashes were identical
  before and after Phase 3.
- No Git, deployment, unified build, full-inference, or final hand-in command was
  run.

## What I changed and why

### Assistant implementation record

The assistant implemented the approved Phase 3 interface and focused tests. It
did not perform student manual labels, accuracy calculation, model selection,
investment conclusions, deployment, or repository publication.

### Student correction and confirmation — to be completed by the student

- [ ] I reviewed the distinction between headline and sector-day metrics.
- [ ] I reviewed the model-risk and non-predictive wording.
- [ ] I completed the blind labels independently, if applicable.
- [ ] Corrections I requested: `<student to complete>`
- [ ] My interpretation of the evidence: `<student to complete>`
