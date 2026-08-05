# Prompt log - Base funds and strict walk-forward backtest

## What I wanted

Implement the three locked non-sentiment PortFoYou funds:
`combined_equal_weight`, `combined_min_variance`, and
`equity_equal_weight`. The implementation had to estimate weights using only
past information, apply them to the following holding period, retain a complete
rebalance audit, fail rather than conceal optimiser problems, and reproduce
daily fund returns from the recorded target weights.

## Prompt(s)

The student requested reusable portfolio and backtest logic in `src/`, build
orchestration in `scripts/`, long-only fully invested minimum-variance weights,
transparent covariance scaling, solver and weight validation, non-trivial
weight variation checks, focused unit and integration tests, full-suite
verification, weight summaries, and a separate AI record.

## What the assistant produced

### Implementation

- Replaced the starter `src/portfolios.py` with:
  - deterministic equal-weight and constrained minimum-variance solvers;
  - positive-scalar covariance rescaling for numerical conditioning while
    reporting the unscaled daily-variance objective;
  - explicit optimiser failure, finite-weight, long-only, weight-sum, and
    objective checks;
  - a monthly, expanding-window schedule with a 252-trading-day initial window;
  - a strict boundary requiring `training_end_date < first_holding_date`;
  - holding-period returns calculated with the validated weights retained in
    the long-form weight log;
  - one-row-per-rebalance audit records, including JSON target weights; and
  - a validation gate that rejects a dynamic method that is unchanged across
    rebalances or indistinguishable from its 1/N benchmark.
- Updated `scripts/run_part_b.py` to load and transform official data through
  the protected data route, build the three funds on a common equity calendar,
  and expose in-memory outputs for later sentiment-fund integration.
- Added `tests/test_portfolios.py` and `tests/test_backtest.py` for optimiser
  constraints, failed-solver rejection, look-ahead boundaries, future-data
  perturbation, weight variation, metric definitions, hosted-data integration,
  and hand reproduction of sampled daily returns.
- Required result CSVs were not written because the fourth locked fund,
  `equity_sentiment_tilt`, has not yet been implemented. This avoids presenting
  a partial fund set as a complete submission artifact.

### Observed official-data build

- All three funds contain 753 out-of-sample equity trading days from
  `2021-01-04` through `2023-12-29`.
- Each fund has 36 monthly decisions from `2020-12-31` through `2023-11-30`.
- Expanding training windows contain 252 to 985 observations, and the first
  holding date is `2021-01-04`.
- `combined_equal_weight`: 60 assets, weights fixed at `0.0166667`.
- `equity_equal_weight`: 50 assets, weights fixed at `0.02`.
- `combined_min_variance`: 60 assets, observed weights from `0.0` to
  `0.231052`; raw daily-variance objectives from `0.000107` to `0.000233`;
  covariance scale factors from `146.890121` to `242.706735`.
- Every fund-date weight sum was exactly `1.0` at the reported precision.
- The minimum-variance method's maximum single-asset change between adjacent
  rebalances was `0.0305066391`; its maximum difference from combined 1/N was
  `0.2143856419`.
- All 108 rebalance audit rows report solver success. Equal-weight rows use the
  explicit `not_required` solver status; minimum-variance rows retain the SLSQP
  status and message.

### Verification commands and outcomes

1. `../../.venv/bin/python -m ruff check --no-cache src/portfolios.py scripts/run_part_b.py tests/test_portfolios.py tests/test_backtest.py`
   - Initial run: exit 1 for one unused `noqa` directive and one non-raw regular
     expression in a test.
   - After the mechanical corrections: exit 0; `All checks passed!`.
2. `../../.venv/bin/python -m pytest -q tests/test_portfolios.py`
   - Exit 0; `5 passed, 1 warning in 0.96s`.
   - The warning was a `PytestCacheWarning`: the restricted process could not
     write the parent repository's `.pytest_cache`. It did not affect test
     execution or project files.
3. `../../.venv/bin/python -m pytest -q tests/test_backtest.py`
   - Exit 0; `7 passed in 5.43s`.
4. `../../.venv/bin/python -m pytest -q`
   - Exit 0; `24 passed in 5.51s`.
5. `../../.venv/bin/python scripts/run_part_b.py`
   - Exit 0; built the three funds in memory and reported 753 dates, 36
     rebalances, and the two weight-variation statistics above.
   - Running the data helper outside a Streamlit runtime emitted benign
     `MemoryCacheStorageManager` warnings before completing successfully.
6. A read-only summary invocation of `build_base_funds()`:
   - Exit 0; produced the date, return, weight, objective, scale, window, and
     weight-sum summaries recorded above.

## What was wrong or risky

- Optimising directly against very small daily covariance values can make SLSQP
  appear successful without moving meaningfully from its starting point. The
  objective is multiplied by one positive scalar for solving, which preserves
  the optimum, while the unscaled variance is retained in the audit.
- Solver success alone is insufficient evidence. Returned weights are checked
  for length, finiteness, bounds, full investment, and a finite non-negative
  objective, and the completed minimum-variance history must vary through time
  and differ from 1/N.
- A month-end decision could accidentally be applied to that same date's return.
  The schedule makes the next equity trading date the first holding date, and
  tests verify that future-row perturbations cannot change past weights or
  returns.
- Silently substituting 1/N after solver failure would mislabel the fund. The
  failed-solver test confirms that an exception is raised with the original
  solver status/message and no fallback weights are accepted.
- Minimum variance remains an estimation-dependent method. Passing numerical
  and timing checks does not establish future economic superiority.

## What I changed and why

### Assistant implementation record

The assistant implemented the locked base-fund design and added executable
timing and portfolio-integrity checks. It did not add transaction costs,
leverage, volatility targeting, another optimiser, or other optional
innovation.

### Student correction and confirmation — to be completed by the student

- [ ] I reviewed the month-end decision and next-trading-day holding convention.
- [ ] I manually checked at least one logged-weight daily return.
- [ ] I reviewed the covariance-scaling explanation and raw objective values.
- [ ] I confirm the three fund definitions and identifiers are correct.
- [ ] Corrections I requested, if any: `<student to complete>`
- [ ] Why I accepted or changed the optimiser validation policy:
      `<student to complete>`
- [ ] Limitations or concerns I want recorded in my own words:
      `<student to complete>`

The assistant has not filled these student-owned confirmation fields.

## Remaining limitations

- This step does not implement `equity_sentiment_tilt`, sentiment scoring or
  fusion, final result CSVs, report figures, the app, or the Word report.
- The build currently assumes the locked 0 bps transaction cost and therefore
  reports gross returns; turnover is not deducted.
- Hosted integration tests and orchestration require access to the official
  course data host, or an authorised local bundle supplied through the protected
  data-access mechanism.
- Streamlit cache warnings appear when the hosted-data helper is called from a
  plain Python process; they are informational and the build completes.
