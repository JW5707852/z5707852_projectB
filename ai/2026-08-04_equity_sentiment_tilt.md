# Prompt log - equity sentiment tilt

## What I wanted

Implement the locked `equity_sentiment_tilt` fund as a transparent,
look-ahead-safe extension of `equity_equal_weight`, and measure its matched
before-versus-after result without tuning on the final sample.

## Prompt(s)

"Implement equity_sentiment_tilt under the locked design. Sentiment may affect
equity weights only and must not affect crypto. Use equity_equal_weight as the
base fund and, at every monthly decision, use only sector sentiment that was
available at the time and lagged by at least one trading day.

For each sector, calculate a z-score from an expanding mean and standard
deviation with at least 60 prior trading days, then clip it to [-2, 2]. Multiply
stock weights by 1 + 0.10 × sector_z and renormalise; use a multiplier of 1 when
the signal is missing. Keep these parameters fixed and do not retune them using
final out-of-sample performance.

The base and enhanced funds must use exactly the same out-of-sample dates,
rebalance schedule, constraints, 0 bps costs, and metric definitions. Compare
annualised return, volatility, Sharpe, maximum drawdown, turnover, and sector
exposure. Retain negative results and do not imply that sentiment predicts
returns unless a look-ahead-safe test supports the claim.

Add tests/test_fusion.py covering signal lagging, past-only expanding
statistics, weight constraints, missing-signal treatment, the common sample,
and fair comparison. Run focused tests and the full pytest suite. Report
commands, exit codes, the actual before-versus-after comparison, and
limitations, and create an AI prompt log. Stop after reporting."

## What the assistant produced

- A fixed-rule fusion implementation in `src/fusion.py` that consumes the
  already-lagged sector signal, records source dates and applied multipliers,
  and derives fund returns from logged weights and the base fund's exact holding
  periods.
- Core-build orchestration that adds the fourth fund to the required fund CSVs.
- Matched fusion performance/turnover and sector-exposure artifacts.
- Executable tests for timing, future perturbation, formula reproduction,
  missing signals, equity-only scope, common samples, and fair conventions.
- Generated evidence on the locked 2021-01-04 to 2023-12-29 sample. Relative
  to `equity_equal_weight`, the tilt reduced annualised return by about 0.10
  percentage points and Sharpe by 0.0051, while reducing volatility by about
  0.007 percentage points and making maximum drawdown about 0.06 percentage
  points less negative. Average monthly target-weight turnover was 4.056% for
  the tilted fund versus 0% for the static equal-weight target.
- Focused validation: `7 passed in 10.26s`. Full suite: `55 passed in 15.26s`.

## What was wrong or risky

- The project is represented as an untracked directory inside the parent Git
  repository, so Git cannot provide granular tracked-file protection.
- A sentiment value must not be joined by sector alone; it must be selected by
  both decision date and sector, with its source date earlier than the decision.
- A missing score must remain identifiable as missing even though the applied
  operational multiplier is one.
- Turnover can be defined several ways. This implementation reports half the
  absolute change in monthly target weights, excludes initial funding, and does
  not deduct costs because the locked assumption is 0 bps.
- Any economic claim based on the final comparison would be in-sample model
  selection if used to retune the fixed 0.10 strength; no retuning is performed.
- The result is a negative-to-mixed fusion outcome, not evidence that headline
  sentiment predicts returns. It was retained rather than retuned or filtered.

## What I changed and why

The final economic interpretation and any claim about the usefulness of
sentiment remain for the student to confirm in their own words.

---

## Follow-up - persistent Checkpoint 8 regression audits

### What I wanted

Persist the official-data Checkpoint 8 placebo, manual reconstruction,
predeclared sensitivity grid, and expanding temporal-fold audits in
`tests/test_fusion.py` so the focused fusion test reruns them after future code
changes.

### Prompt(s)

The follow-up required four persistent official-data test categories while
allowing changes only to `tests/test_fusion.py` and the fusion AI log. It fixed
the three manual decision dates, the diagnostic strength grid
`[0.00, 0.05, 0.10, 0.15, 0.20]`, the three expanding fold boundaries, and a
`1e-12` tolerance. It prohibited production, result, dependency, context, app,
and data-access changes and prohibited full-project verification commands.

### What the assistant produced

- An official zero-sentiment placebo test using the normal production fusion
  path and the actual equity rebalance schedule.
- Three parameterised manual reconstruction cases that start from headline-
  level VADER scores and use direct Pandas/NumPy aggregation, expanding
  statistics, clipping, tilting, normalisation, and return calculations.
- A predeclared diagnostic-grid regression test using common dates, schedules,
  constraints, cost assumptions, and direct metric formulas.
- Three time-ordered expanding-fold regression cases with independently
  calculated performance and stock-to-sector exposure differences.
- Module-local helpers and reuse of the existing module-scoped official build,
  avoiding repeated hosted-data builds.

### What was wrong or risky

- Fixed-data numerical snapshots are reproducibility evidence for this supplied
  dataset only. They are not universal claims about sentiment or tilt strength.
- The final-sample diagnostic grid must not be used to select a preferred
  strength. The production value remains the predeclared `0.10`.
- Expected manual signals must not be copied from the production z-score field.
  The regression tests therefore reconstruct raw sector scores from
  headline-level VADER scores before calculating expanding statistics.
- Official-data tests are slower than synthetic unit tests and depend on the
  hosted-data loader, so they share one module-scoped build.
- Git still represents the project as an untracked directory inside the parent
  repository; explicit hashes are used to demonstrate that production and
  result files were unchanged.

### Verification

Commands and results are recorded after execution below:

- `../../.venv/bin/python -m ruff check --no-cache tests/test_fusion.py`
  initially found one pre-existing 102-character assertion line. After a
  formatting-only reflow in the permitted test file, the rerun exited `0` with
  `All checks passed!`.
- `../../.venv/bin/python -m pytest -q tests/test_fusion.py` exited `0` with
  `15 passed in 11.51s`, increasing the focused suite from 7 to 15 collected
  cases.
- No full pytest suite, unified build, Streamlit process, submission check, or
  Git mutation was run.

### What I changed and why

The student must confirm any interpretation of the fixed-data sensitivity and
fold results; these tests establish reproducibility, not predictive validity.
