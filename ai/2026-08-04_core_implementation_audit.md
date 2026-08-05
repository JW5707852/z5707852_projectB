# Prompt log - core implementation audit and repair

## What I wanted

Audit the existing core implementation against the project financial and timing
definitions, repair only confirmed defects, and independently reconcile sampled
calculations against the official hosted data and generated artifacts.

## Prompt(s)

> Audit and repair the current core implementation without adding new features.
> Read docs/ai/code-review.md and review financial calculations, time alignment,
> look-ahead prevention, optimiser validation, error handling, typing and
> documentation, relative paths, determinism, app-artifact schemas, and test
> coverage. Independently hand-recalculate at least one equity return, one crypto
> return, one sampled daily fund return, a short growth-of-$1 path, annualised
> return, Sharpe, and maximum drawdown. Sample-check non-trading-day headline
> mapping, the one-day signal lag, ticker-day aggregation, and sector ticker
> equal-weighting. Run the full tests, core build, and Ruff when configured. Fix
> root causes, add a test for each substantive fix, and report the evidence.

## What the assistant produced

- Corrected drawdown calculations so the running peak includes the initial $1.
- Rejected non-positive adjusted prices before calculating returns.
- Separated equal-weight covariance reporting from the minimum-variance
  optimiser's non-zero covariance requirement.
- Added independent optimiser objective and feasible-start validation.
- Rejected duplicate rebalance-calendar dates instead of silently de-duplicating
  them.
- Rejected fund returns whose decision date is on or after the return date,
  non-positive annualisation factors, and metric samples shorter than two rows.
- Added focused regression tests for each changed behaviour.
- Rebuilt and revalidated the core artifacts and performed direct NumPy/Pandas
  reconciliations of returns, metrics, timing, and sentiment aggregation.

Affected files:

- `src/features.py`
- `src/portfolios.py`
- `src/metrics.py`
- `src/evidence.py`
- `src/app_artifacts.py`
- `tests/test_etl.py`
- `tests/test_portfolios.py`
- `tests/test_backtest.py`
- `tests/test_metrics.py`
- `tests/test_evidence.py`
- `tests/test_app_artifacts.py`
- this AI log

## What was wrong or risky

The material financial defect was the drawdown baseline. Applying
`wealth / wealth.cummax() - 1` only to observed wealth makes a first-period loss
look like zero drawdown because the impaired wealth value becomes its own first
peak. The correct path includes the starting wealth of 1.0 in the running peak.
The current real funds reach their maximum drawdowns later in the sample, so the
reported real-fund maximum-drawdown values did not change, but the path formula
was nevertheless incorrect at this boundary.

The optimiser previously trusted a solver's success flag and returned weights
without reconciling the reported objective to those weights or checking that the
solution improved on the feasible equal-weight starting point. In addition, the
equal-weight benchmark reused a covariance-scaling helper that unnecessarily
rejected a valid zero-covariance sample.

The rebalance-schedule helper called `unique()` before testing for duplicate
dates, making its stated duplicate guard unreachable. The artifact metrics layer
also relied on downstream app validation to reject holding returns dated on or
before their decision dates. These issues did not alter the valid official
artifacts but weakened failure behaviour for invalid inputs.

The first full-suite run was executed in a network-restricted sandbox and exited
1 because the protected hosted-data loader could not resolve either official
data URL. It recorded 88 passed, 1 failed, and 26 errors. The identical command
was rerun with network access and exited 0 with 115 passed. The Streamlit
MemoryCacheStorageManager warnings outside Streamlit were informational.

The exact repository-wide Ruff command exited 1 with 378 findings. The findings
are dominated by the read-only `FINS5545/` teaching reference and also include
the protected `src/data_access.py` and unchanged starter files. No protected or
read-only file was edited to hide those findings. A focused Ruff run over every
changed production and test file exited 0.

## Verification evidence

- Focused regression tests: 16 passed, exit 0.
- Full suite with hosted-data access: 115 passed in 38.62 seconds, exit 0.
- `../../.venv/bin/python scripts/run_part_b.py`: exit 0; 3,765 fund-return
  rows, 9,720 weight rows, 5 metric rows, and 10,060 sector-index rows.
- The four core CSV SHA-256 values still exactly match the previously recorded
  deterministic-build hashes; valid official financial outputs were unchanged.
- Focused Ruff over changed files with `--no-cache`: all checks passed, exit 0.
- Project-code Ruff with the read-only `FINS5545/`, protected
  `src/data_access.py`, and unchanged starter check files explicitly excluded on
  the command line: all checks passed, exit 0. No repository configuration was
  changed to suppress findings.
- Exact `../../.venv/bin/python -m ruff check .`: 378 pre-existing or
  out-of-scope findings, exit 1.
- Independent audit script: exit 0 with tolerance `1e-12`.

Selected independent reconciliations:

- ABBV 2020-01-06 equity return:
  `68.581176757812 / 68.044166564941 - 1 = 0.007892082745382`.
- ADA-USD 2020-01-06 crypto return, using the native Sunday price:
  `0.037271998823 / 0.034720998257 - 1 = 0.073471406161165`.
- Equity sentiment tilt on 2022-07-01: the 50 logged weights summed to 1 and
  independently reproduced `0.010513330959588` (difference `3.47e-18`).
- For returns `[0.10, -0.20, 0.05]`, wealth was
  `[1.10, 0.88, 0.924]`, geometric annual return
  `-0.998692488303802`, annualised volatility `2.551470164434615`,
  Sharpe `-1.646109783506203`, and maximum drawdown `-0.20`.
- A 2020-01-05 ABBV weekend headline mapped forward to 2020-01-06.
- The first available Comm tradable signal used 2020-03-30 on 2020-03-31,
  exactly one equity trading day later.
- ABBV's 2020-01-02 ticker-day score was the mean of three headline scores,
  `0.1468`.
- The Comm 2020-01-02 sector score equal-weighted four ticker-day values to
  `0.108975`; the raw headline-weighted mean would have been
  `0.154745454545455`.

## What I changed and why

AI-generated changes and evidence above require student review. No student
interpretation, approval, or correction has been inferred.

- Student-confirmed correction: _To be completed by the student._
- Student review date: _To be completed by the student._
- Student decision on remaining repository-wide Ruff findings:
  _To be completed by the student._
