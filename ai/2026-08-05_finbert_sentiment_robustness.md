# Prompt log - VADER–FinBERT financial sentiment robustness layer

## Status

Phase 1 pilot, Phase 2 full-corpus, Phase 3 Streamlit, and Phase 4
student-labelled validation gates passed. The five funds and protected core
artifacts remain unchanged.

## Phase 4 student-labelled validation

The student supplied a completed 150-row blind review. The initial integrity
gate rejected abbreviated labels, missing repeated reviewer entries, and two
Excel-altered source strings. The student then supplied full positive, negative,
and neutral labels; high, medium, and low confidence; explicitly confirmed that
the project and all labels were their sole work; declined to repeat the same
reviewer identifier in every row; and authorised restoration of the two source
strings from the canonical blind file.

The deterministic import retained all student judgements. It restored two
`text_raw` values by `review_id` and removed trailing Excel whitespace from 136
confidence cells. It did not infer or replace a label, confidence category, or
comment. The completed review was joined one-to-one to the hidden audit design.

Only the 100 representative rows were used for weighted performance metrics.
The 50 disagreement-enriched rows remained qualitative diagnostics. VADER's
weighted accuracy was 46.82% and FinBERT's was 53.07%, a 6.25 percentage-point
point-estimate difference. The exact paired McNemar p-value was 0.377, so the
small review sample does not provide strong evidence of different model error
rates. The app presents the point estimates and paired-test limitation without
claiming return prediction or model superiority.

Only four student comments were supplied. Diagnostic categories are assigned
only when those comments support them; other cases remain uncategorised rather
than receiving AI-authored explanations.

## What I wanted

Add an independent and auditable pretrained FinBERT comparison to the existing
VADER headline sentiment index without changing the five funds, the VADER core
artifact, the fusion methods, or the deployed app's lightweight dependency set.
The work must distinguish descriptive agreement from accuracy, preserve student
ownership of human labels and conclusions, and proceed through smoke, pilot,
full-inference, and app gates.

## Prompt(s)

The student requested a pinned `ProsusAI/finbert` robustness layer using
unchanged `text_raw`, batched inference, exact probability outputs, the score
`probability_positive - probability_negative`, ticker-day then equal-weight
sector aggregation, comparison evidence, two-purpose manual-review sampling,
report-ready figures, precomputed Streamlit presentation, focused tests, and
protected-artifact regression checks.

Before the full run, the student added a mandatory pilot gate of approximately
512–1,000 deterministic distinct titles. The gate had to verify the model
revision, label IDs, probability constraints, score formula, device, throughput,
memory use, truncation, and revised full-run estimate. The student required a
checkpoint report before the complete 105,334-title inference.

## What the assistant produced in Phase 1

### Dependency and model contract

- Added build-only pins to `requirements-dev.txt`:
  - `transformers==5.14.1`
  - `huggingface-hub==1.26.0`
  - `torch==2.12.1`
- Left deployment `requirements.txt` unchanged.
- Pinned model: `ProsusAI/finbert`.
- Pinned and resolved revision:
  `4556d13015211d73dccd3fdd39d39232506f3e43`.
- Verified mapping: `0=positive`, `1=negative`, `2=neutral`.
- Model licence recorded as Apache-2.0.
- Score definition locked as positive probability minus negative probability.

### Implementation

- Added `src/finbert_innovation.py` with:
  - exact revision and label-map validation;
  - external-cache enforcement;
  - CUDA, MPS, then CPU automatic selection;
  - configurable batch size and truncation length;
  - unchanged-title validation and distinct-title scoring;
  - `model.eval()` and `torch.inference_mode()`;
  - finite/bounded/sum-to-one probability validation;
  - approved continuous score and argmax label calculation;
  - token-length and truncation measurement; and
  - throughput and process peak-RSS evidence.
- Added explicit and mutually exclusive `--pilot-only`, `--full-corpus`, and
  guarded `--postprocess-existing` modes to `scripts/run_finbert_innovation.py`.
  Full inference is never the default operation.
- Added pure tests and an opt-in real-model smoke test. No Streamlit file was
  modified.

### Verification and observed pilot evidence

1. Dependency installation:
   - Command: repository interpreter plus `-m pip install -r requirements.txt
     -r requirements-dev.txt`.
   - Exit 0; exact approved versions installed.
2. Pure tests:
   - Command: repository interpreter plus `-m pytest -q -p no:cacheprovider
     tests/test_finbert_innovation.py`.
   - Exit 0; 5 tests passed.
3. Focused Ruff:
   - Final command over the new source, script, and tests with `--no-cache`.
   - Exit 0; all checks passed after three mechanical sorting corrections.
4. Real pinned-model smoke test:
   - Exit 0; 1 test passed in 32.10 seconds.
   - One upstream tokenizer deprecation warning appeared; inference succeeded.
5. Deterministic pilot:
   - Command: `scripts/run_finbert_innovation.py --pilot-only --pilot-size 768
     --batch-size 32 --max-length 128 --device auto` with an external cache.
   - Exit 0; device selected: MPS.
   - Clean headline rows: 146,836.
   - Full distinct-title count: 105,334.
   - Pilot seed: 5,545.
   - Model load time from the populated external cache: 3.441 seconds.
   - Inference time: 4.046 seconds.
   - Throughput: 189.819 distinct titles per second.
   - Revised full-inference estimate: 554.919 seconds, or 9.249 minutes.
   - Peak RSS before model: 642.094 MiB.
   - Peak RSS after model load: 1,159.672 MiB.
   - Peak RSS after inference: 1,192.984 MiB.
   - Peak-RSS model increment: 517.578 MiB.
   - Maximum pilot token length: 49.
   - Truncated pilot titles: 0 of 768 (0%).
   - Probability-sum range: 0.999999873 to 1.000000121.
   - All 768 probability rows were finite.
   - Score arithmetic matched the approved formula.

The revised 9.25-minute estimate is within the original 5–20 minute accelerated
planning range and is not materially longer.

### Protected artifacts

Hashes were identical before and after the pilot:

- `fund_returns.csv`:
  `9688e72738949a93193c2e154bbf31a0088b6fe9d4aea7817d513084de5e2502`
- `fund_weights.csv`:
  `660e64e9173c147764f7bae523fcb9885f982722be88aae12a3832eb505a7865`
- `sector_sentiment_index.csv`:
  `4b809ee46ca83a697775869eb20de2a5920d98fb25d863e0fb2ec205e4ff421f`
- `performance_metrics.csv`:
  `8e35b069717df0d909b6678a05dd3f90dc8a81ca00e2e1bb3619a38728533910`

## What was wrong or risky

- Treating the model's label positions as remembered knowledge could reverse
  the continuous score. Both configuration and runtime mapping are required to
  equal the pinned contract before inference.
- An explicit unavailable device must raise rather than silently fall back. In
  automatic mode the observed process selected MPS successfully.
- Timing a small smoke example would exaggerate load overhead. The separate
  768-title benchmark times batched inference and records model-load time
  independently.
- A 768-title pilot cannot establish the full-corpus truncation rate. Its zero
  truncations are pilot evidence only; Phase 2 must count truncation across all
  105,334 distinct titles.
- Model agreement is not accuracy. Accuracy and superiority remain pending the
  student's labels on a representative evaluation sample.
- The Hugging Face client emitted an unauthenticated-request warning. The exact
  pinned public revision still loaded successfully; no token or secret is
  required for this build.

## What I changed and why

### Assistant implementation record

The assistant implemented and tested the approved Phase 1 and Phase 2 model,
inference, comparison, sampling, artifact, and figure infrastructure. It did not
edit Streamlit, add a fund, run the unified build, or perform Git/deployment
operations.

## Phase 2 full-corpus and artifact evidence

### Successful execution

- Command: repository interpreter plus
  `scripts/run_finbert_innovation.py --full-corpus --device mps --batch-size 32
  --max-length 128 --cache-dir /private/tmp/portfoyou-huggingface`.
- Successful run exit status: 0.
- Model load time: 3.260 seconds.
- Full inference time: 387.284 seconds.
- Full Phase 2 wall time: 400.898 seconds.
- Throughput: 271.982 distinct titles per second.
- Peak RSS: 1,220.016 MiB.
- Distinct titles scored exactly once: 105,334.
- Finite and score-arithmetic-validated probability rows: 105,334.
- Probability-sum range: 0.999999860 to 1.000000149.
- Actual maximum token length: 84.
- Truncated titles: 0 of 105,334 (0%).

### Join, aggregation, and comparison

- Clean headline rows before and after the score join: 146,836 and 146,836.
- Matched rows: 146,836; unmatched rows: 0; multiplied rows: 0.
- Six headlines remained outside the equity calendar, consistent with the
  existing treatment.
- Ticker-day rows after headline-first aggregation: 37,962.
- FinBERT sector grid: 10,060 date-sector rows; 9,832 observed news rows and 228
  missing-score rows.
- Coverage, ticker counts, headline counts, possible counts, and missingness
  matched the protected VADER artifact across all 10,060 grid rows.
- VADER–FinBERT correlations are reported only on matched observed date-sector
  rows, with 9,832 paired observations overall and a paired count on every
  correlation row.
- Headline label agreement remains a descriptive comparison. Headline-level
  correlations are suppressed to enforce the approved date-sector rule.

### Review samples and claims boundary

- The 150-row manual-review template contains 100 representative-evaluation
  titles selected by deterministic sector-year stratified random sampling and
  50 separate disagreement-enriched diagnostic titles.
- Diagnostic strata cover opposite-sign, neutral/non-neutral, negation,
  financial-term, numerical, and earnings-announcement cases.
- Representative rows retain inclusion probabilities and sampling weights;
  diagnostic rows are explicitly unweighted for qualitative diagnosis only.
- All student-owned label, confidence, note, and reviewer fields remain blank.
- Every review row remains `pending student review`. No accuracy, superiority,
  or predictive claim was made.

### Generated evidence and tests

- Generated the distinct-title score cache, the FinBERT sector index, comparison
  table, disagreement audit, manual-review template, metadata, completion
  manifest, figure-QA table, and three PNG/PDF/caption figure bundles.
- The three figures show distinct-title score distributions, matched sector
  time series, and matched sector-day correlations/agreement.
- Final focused Ruff result: all checks passed.
- Final focused test command covered Phase 1 logic, Phase 2 logic, and published
  artifact contracts; exit 0 with 13 tests passed.
- The figures were inspected after export. A first layout attempt had title and
  footer overlaps; it was rejected, margins were corrected, and the final PNGs
  were visually rechecked.

### Failure and maintainable correction record

The first full attempt completed all 105,334 neural inferences but stopped before
publishing outputs. A strict `DataFrame.equals` coverage check treated equivalent
CSV microsecond and in-memory second datetime storage resolutions as unequal.
A read-only diagnostic found zero substantive differences across every coverage
field. The correction normalises dates, compares keys/counts/booleans exactly,
and applies a `1e-15` tolerance only to the derived coverage ratio. A regression
test confirms that datetime resolution passes while a one-headline difference
still fails. No model or aggregation definition was changed.

A guarded postprocessing refresh was later used to correct figure layout and
suppress non-date-sector correlations without repeating inference. It accepts
only the validated 105,334-row cache plus PASS metadata, rebuilds derived outputs
in an external staging directory, and promotes them only after validation.

### Protected artifacts after Phase 2

All four hashes remained identical before and after Phase 2:

- `fund_returns.csv`:
  `9688e72738949a93193c2e154bbf31a0088b6fe9d4aea7817d513084de5e2502`
- `fund_weights.csv`:
  `660e64e9173c147764f7bae523fcb9885f982722be88aae12a3832eb505a7865`
- `sector_sentiment_index.csv`:
  `4b809ee46ca83a697775869eb20de2a5920d98fb25d863e0fb2ec205e4ff421f`
- `performance_metrics.csv`:
  `8e35b069717df0d909b6678a05dd3f90dc8a81ca00e2e1bb3619a38728533910`

### Student correction and confirmation — to be completed by the student

- [ ] I reviewed the pinned model revision and label mapping.
- [ ] I reviewed the observed MPS throughput and revised runtime estimate.
- [ ] I reviewed the external-cache and truncation policies.
- [ ] I authorise Phase 2 full-corpus inference after reviewing this checkpoint.
- [ ] Corrections I requested, if any: `<student to complete>`
- [ ] Limitations or concerns I want recorded in my own words:
      `<student to complete>`

The assistant has not completed any student-owned confirmation or judgement.
