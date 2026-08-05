# Prompt log - Performance metrics and fund artifacts

## What I wanted

Apply one consistent fact-sheet calculation to every implemented PortFoYou fund,
retain the latest target holdings, and generate validated return, weight, and
performance CSVs for the app and report. The definitions had to use geometric
annual return, 252-day sample volatility, a zero-rate arithmetic-mean Sharpe,
and running-peak drawdown.

## Prompt(s)

The student requested growth of $1, geometric annualised return, annualised
volatility, Sharpe ratio, maximum drawdown, current target holdings, the three
exact required fund artifact filenames, clear schemas, consistent identifiers
and dates, independent metric and sampled-return verification, tests, output
evidence, and a separate AI log.

## What the assistant produced

### Implementation

- Added `src/metrics.py` as the reusable calculation and validation layer. It:
  - sorts and validates one return per fund-date;
  - recalculates growth of $1 as the cumulative product of `1 + daily_return`;
  - delegates the locked performance definitions to the existing common
    portfolio metric function;
  - records 252 periods per year, a 0% annual risk-free rate, and the geometric
    annual-return method explicitly;
  - validates long-only fully invested target weights;
  - adds a generic weight `date` alongside `decision_date`;
  - labels only the most recent rebalance for each fund with `is_current`;
  - retains all historical rebalance metadata; and
  - reconciles fund identifiers, metadata, performance sample dates, and return
    decision keys before permitting a write.
- Updated `scripts/run_part_b.py` to build the validated frames and write:
  - `results/data/fund_returns.csv`;
  - `results/data/fund_weights.csv`; and
  - `results/tables/performance_metrics.csv`.
- Added `tests/test_metrics.py` with independent formula calculations, latest-
  holdings checks, invalid-schema rejection, hosted artifact schemas, date and
  fund reconciliation, three sampled-return reproductions, and a CSV round-trip
  test outside the submission tree.

### Generated artifact evidence

- `fund_returns.csv`: 2,259 rows and seven columns; 753 observations for each
  of three funds from `2021-01-04` through `2023-12-29`.
- `fund_weights.csv`: 6,120 rows; 36 rebalance dates per fund; all audit fields
  retained; current holdings dated `2023-11-30` and summing to one.
- `performance_metrics.csv`: three rows, one per implemented fund, with sample,
  as-of, and current-holdings dates plus explicit formula assumptions.
- Fund identifiers reconcile across all three files, and every return
  `fund + decision_date` key has a matching weight record.

Observed metrics:

| Fund | Growth of $1 | Geometric annual return | Annual volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| `combined_equal_weight` | 1.517526 | 0.149792 | 0.212517 | 0.763279 | -0.287470 |
| `combined_min_variance` | 1.225270 | 0.070355 | 0.127873 | 0.595667 | -0.183037 |
| `equity_equal_weight` | 1.427260 | 0.126435 | 0.161662 | 0.817387 | -0.203219 |

The independent CSV reload produced formula differences no larger than about
`5.1e-15`, attributable to decimal round-trip precision. Three sampled combined
minimum-variance daily returns reproduced from the generated weight CSV and
underlying asset returns with absolute errors of `4.16e-17`, `8.50e-17`, and
`6.12e-17`.

### Verification commands and outcomes

1. `../../.venv/bin/python -m ruff check --no-cache src/metrics.py scripts/run_part_b.py tests/test_metrics.py src/portfolios.py`
   - Initial run: exit 1 because the new `__all__` list was not in Ruff's
     required order.
   - After the mechanical sort: exit 0; `All checks passed!`.
2. `../../.venv/bin/python -m pytest -q tests/test_metrics.py`
   - Exit 0; `7 passed in 5.31s`.
3. `../../.venv/bin/python -m pytest -q`
   - Exit 0; `31 passed in 7.50s`.
4. `../../.venv/bin/python scripts/run_part_b.py`
   - Exit 0; wrote the three requested files and reported 753 dates and 36
     rebalances.
   - Plain-Python calls to the hosted data helper emitted informational
     Streamlit memory-cache warnings but completed successfully.
5. Independent CSV reload and formula audit:
   - First attempt: exit 1 due solely to an inline audit-command quoting error;
     no project file was changed.
   - Corrected attempt: exit 0; schemas, identifiers, dates, current holdings,
     and formulas reconciled.
6. Independent three-date return reproduction using the generated CSV weights:
   - Exit 0; all three differences were below `1e-16`.

## What was wrong or risky

- Using arithmetic annual return for the fact sheet would conflict with the
  locked geometric definition. The method and periods per year are stored in
  the performance artifact so the interpretation is explicit.
- A Sharpe calculated from geometric return would not match the required
  arithmetic-mean daily excess-return definition. The table retains annualised
  mean excess return, annualised volatility, and their ratio for auditability.
- Selecting current holdings by file row order could return stale targets. The
  current flag is derived from the maximum decision date separately for each
  fund, and the selected holdings must still sum to one.
- Writing artifacts before validating identifiers and decision keys could leave
  internally inconsistent app inputs. All frames are reconciled in memory before
  the orchestration writes any CSV.
- The locked core design also names `equity_sentiment_tilt`, but that fund has
  not yet been implemented. No placeholder returns, fabricated holdings, or
  empty performance row was created. The generated artifacts cover the three
  currently implemented base funds and must be regenerated after sentiment
  fusion is added.

## What I changed and why

### Assistant implementation record

The assistant implemented the shared metrics and artifact contracts, generated
the current three-fund outputs, and retained full historical weights plus an
explicit latest-holdings flag. It did not implement sentiment or invent the
fourth fund to make the artifact appear complete.

### Student correction and confirmation — to be completed by the student

- [ ] I independently checked at least one growth-of-$1 path and fact-sheet row.
- [ ] I confirm that Sharpe uses arithmetic mean daily excess return and a 0%
      risk-free rate.
- [ ] I reviewed the latest target holdings dated `2023-11-30`.
- [ ] I understand that these artifacts must be regenerated after
      `equity_sentiment_tilt` is implemented.
- [ ] Corrections I requested, if any: `<student to complete>`
- [ ] Why I accepted or changed the metric definitions: `<student to complete>`
- [ ] Limitations or concerns I want recorded in my own words:
      `<student to complete>`

The assistant has not filled these student-owned confirmation fields.

## Main Step 6 behavioural coverage extension

Validation Checkpoint 6 initially identified gaps in executable test coverage.
The production calculation and generated artifacts were not changed. The
following test-only additions were made:

- `tests/test_metrics.py` now parameterises three independent golden series:
  - zero returns;
  - constant positive 1% returns; and
  - `[0.10, -0.20, 0.05]`, whose wealth path is
    `[1.10, 0.88, 0.924]` and maximum drawdown is `-0.20`.
- The golden tests independently calculate cumulative wealth, geometric annual
  return, sample annual volatility, zero-rate arithmetic-mean Sharpe, and
  running-peak maximum drawdown. Zero and constant-return cases explicitly
  require `NaN` Sharpe when volatility is zero.
- Added substantive `tests/test_artifacts.py`, which loads the actual three CSV
  artifacts and validates:
  - columns, parsed date/numeric/boolean types, and non-null identifiers;
  - unique keys and per-fund date ordering;
  - fund/family/method identity reconciliation;
  - the 252-day and 0% risk-free-rate conventions;
  - growth paths, sample dates, observations, and join row conservation;
  - finite long-only fully invested weights at every rebalance;
  - direct NumPy performance calculations for `combined_min_variance`;
  - three beginning/middle/end CSV returns from logged CSV weights and hosted
    asset returns; and
  - exact `is_current`, latest-decision, weight-sum, and fact-sheet date
    reconciliation.

Verification:

1. `../../.venv/bin/python -m ruff check --no-cache tests/test_metrics.py tests/test_artifacts.py`
   - Exit 0; `All checks passed!`.
2. `../../.venv/bin/python -m pytest -q tests/test_metrics.py tests/test_artifacts.py`
   - Exit 0; `17 passed in 6.38s`.

No full-suite test, unified build, Streamlit process, or hand-in check was run in
this extension. The student-owned confirmation fields above remain open.

## Remaining limitations

- The files currently contain the three implemented funds only. The locked
  `equity_sentiment_tilt` fund remains future work.
- `sector_sentiment_index.csv`, final figures, app fact sheets, allocation UI,
  and report content are not part of this step.
- Returns are gross under the locked 0 bps transaction-cost assumption.
- Hosted builds require network access or an authorised local official-data
  bundle through `src/data_access.py`.
