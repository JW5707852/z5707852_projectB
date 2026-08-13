# Prompt log - core assessment tables and figures

## What I wanted

Generate the seven required Part B core assessment exhibits from the current,
full-precision project artifacts. Keep the work reproducible, restrained,
Word/A4-ready, source-traceable, and limited to the locked four core funds and
locked one-day sentiment fusion.

## Prompt(s)

> Generate all evidence required for the core assessment from this project's
> actual artifacts, without adding optional innovation. Save tables under
> results/tables/ and figures under results/figures/.
>
> At minimum, create: a performance table across funds and methods; a
> growth-of-$1 comparison; a drawdown figure for at least one fund; a
> weights-over-time figure across methods for at least one fund family; a Sharpe
> or return-risk comparison; the ten-sector sentiment time series; a fusion
> before-versus-after table; and a fusion before-versus-after figure.
>
> Every exhibit must be self-contained, with a clear title, labelled axes,
> units, sample period, data source or generation path, and a verifiable
> calculation definition. Use 2 to 3 significant digits in report-facing
> tables while retaining sufficient precision in underlying CSV files. Apply
> one coherent, restrained, Word/A4-ready visual system. Do not copy course
> figures or numerical conclusions.
>
> Programmatically and visually inspect every figure for clipping, overlap,
> incorrect labels, and unreadable fonts. Verify that plotted data trace back to
> the required CSV artifacts. Run relevant tests and the complete build. Report
> generated files, test commands, exit codes, and visual-QA results, and create
> an AI prompt log. Stop after reporting.
>
> Scope contract: validation is limited to the evidence-generation code, the
> tables and figures created in this step, and their direct source-artifact
> dependencies. “Complete build” means the existing core artifact build followed
> by the evidence workflow. Do not inspect the app, Git/GitHub, report
> completion, deployment, hand-in readiness, or later work. Do not run
> Streamlit, check_handin.py, or unrelated tests.

## What the assistant produced

Reusable evidence definitions and validation:

- `src/evidence.py`
  - filters the existing exploratory fund out of the locked core evidence pack;
  - derives exact wealth and drawdown paths from daily returns;
  - builds report-facing three-significant-digit performance and fusion tables;
  - validates the retrospective ten-sector raw sentiment series without
    converting missing no-news days to zero; and
  - creates a descriptive weights view using the six tickers with the highest
    logged peak weights in the combined minimum-variance fund, with remaining
    assets grouped explicitly.

Deterministic orchestration and visual QA:

- `scripts/generate_evidence.py`
  - reads only the existing full-precision CSV artifacts;
  - writes two report-facing tables, an evidence manifest, and a figure-QA table;
  - exports six figures as PNG, PDF, and caption Markdown bundles;
  - records title, sample, units, source path, and calculation definition;
  - validates axes, display labels, grids, legends, tick overlap, point-label
    overlap, clipping, minimum 8-point text, and non-blank PNG output; and
  - resolves production paths from the project root and keeps Matplotlib cache
    files outside the submission tree.

Focused executable tests:

- `tests/test_evidence.py`
  - independently reconciles report values to the precise metrics CSV;
  - directly recalculates wealth and drawdown from daily returns;
  - verifies displayed weights conserve every fund-decision total;
  - confirms the plotted sentiment values are the unmodified raw sector index;
  - reconciles the locked fusion table; and
  - generates all outputs in an external temporary directory and checks every
    table, PNG, PDF, caption, and QA row.

Generated report tables:

- `results/tables/performance_table_core.csv` - 4 rows
- `results/tables/fusion_before_after_table.csv` - 2 rows
- `results/tables/evidence_manifest.csv` - 8 rows
- `results/tables/figure_qa.csv` - 6 rows

Generated figure bundles, each with `.png`, `.pdf`, and `.caption.md`:

- `growth_of_1_comparison`
- `drawdown_equity_sentiment_tilt`
- `combined_weights_over_time`
- `return_risk_comparison`
- `sector_sentiment_time_series`
- `fusion_before_after`

## Calculation definitions

- Growth of $1: cumulative product of `1 + daily_return`.
- Geometric annual return: `growth^(252 / observations) - 1`.
- Annualised volatility: sample daily standard deviation times `sqrt(252)`.
- Sharpe ratio: annualised mean daily excess return divided by annualised
  volatility, using a 0% annual risk-free rate.
- Drawdown: cumulative wealth divided by its running peak, minus one.
- Combined weights figure: logged target weights for the six tickers with the
  largest maximum weight in the combined minimum-variance history; all other
  assets are grouped. This is a descriptive display rule only and is not used
  in portfolio formation or performance evaluation.
- Sector sentiment figure: unmodified `raw_sector_compound`, built by averaging
  headlines to ticker-day first and then equal-weighting observed ticker-days
  within the sector; missing no-news days remain missing.
- Fusion comparison: matched equity 1/N and locked one-trading-day-lag sentiment
  tilt over identical dates and rebalances at 0 bps transaction costs.

## Verification record

1. Final focused Ruff:
   - Command: `../../.venv/bin/python -m ruff check --no-cache src/evidence.py scripts/generate_evidence.py tests/test_evidence.py`
   - Exit 0: `All checks passed!`
2. Pre-build focused evidence tests:
   - Command: `../../.venv/bin/python -m pytest -q tests/test_evidence.py`
   - Final exit 0: `7 passed, 1 warning in 6.34s`.
3. Core artifact build:
   - Command: `../../.venv/bin/python scripts/run_part_b.py`
   - Exit 0; core validation reported 3,765 returns, 9,720 weights, 5 precise
     metric rows, 10,060 sector rows, and 13.187 seconds elapsed.
4. Evidence generation:
   - Command: `../../.venv/bin/python scripts/generate_evidence.py`
   - Exit 0; wrote 4 table files and 6 PNG/PDF/caption bundles; programmatic QA
     reported `6 PASS, 0 layout issues`.
5. Final source-build plus evidence tests:
   - Command: `../../.venv/bin/python -m pytest -q tests/test_build.py tests/test_evidence.py`
   - Exit 0: `14 passed, 1 warning in 6.58s`.

The warning was the known parent-repository pytest-cache permission warning. It
did not affect project files, calculations, or test outcomes.

## What was wrong or risky

- The initial log-scale growth figure mixed dollar-formatted major labels with
  scientific minor labels. The final figure uses explicit in-range fixed ticks,
  dollar formatting, and no log minor ticks.
- The initial growth legend overlapped plotted data. It was moved into a
  dedicated area above the axes.
- Automatic locators retained off-range tick artists. Deterministic plotted-
  sample year ticks replaced the automatic locator, and the clipping validator
  ignores only ticks that Matplotlib clips because they are outside explicit
  limits.
- The first combined-weight view aggregated only equity versus crypto. Although
  it was correct, the minimum-variance crypto allocation is effectively zero,
  so the chart hid meaningful stock-weight changes. The final chart displays six
  explicitly selected peak-weight tickers plus an `Other assets` group and
  verifies that each fund-decision stack sums to 100%.
- The first weights layout repeated the upper x-axis label too close to the
  lower panel title. The redundant label was removed and the final image was
  visually reinspected.
- Multiple development-time focused tests intentionally failed on the above QA
  defects before the final corrected tests passed. The failures were not hidden
  or bypassed.

## Visual inspection record

Every final PNG was opened at high or original resolution. The final review
found no clipped titles, metadata, axes, labels, or legends; no overlapping point
labels or panel titles; readable fonts at Word width; correct percentage, dollar,
and score units; and consistent navy, crimson, teal, gold, violet, green, and
neutral-grey styling. The plotted sample periods and source paths match the
manifest and caption sidecars.

## Manifest sample-period metadata correction

### Defect identified

After the initial evidence step, the generated manifest assigned the fund-return
sample (`2021-01-04 to 2023-12-29`) to every figure. This was incorrect for the
combined-weights figure, whose plotted decision dates begin on `2020-12-31` and
end on `2023-11-30`, and for the sector-sentiment figure, whose source dates span
`2020-01-02` to `2023-12-29`.

### Root cause

`scripts/generate_evidence.py::_manifest()` created one `fund_sample` from the
performance table and reused it for the two table rows and all six figure rows.
The individual figure builders already created correct, source-derived
`FigureContext.sample` values, but `_manifest()` did not receive or use those
contexts.

### Correction

- The performance-table sample is derived from that table's own
  `sample_start_date` and `sample_end_date`.
- The fusion-table sample is derived independently from the fusion table.
- Each figure manifest row now receives the same `FigureContext.sample` value
  used to generate its caption sidecar.
- `tests/test_evidence.py` now independently derives all eight expected sample
  periods from the precise source dataframes, protects the previously wrong
  weights and sentiment ranges explicitly, requires eight unique exhibit IDs,
  and reconciles every figure manifest row with its `## Sample` caption value.
- The manifest was regenerated by `scripts/generate_evidence.py`; it was not
  manually edited.

Affected files:

- `scripts/generate_evidence.py`
- `tests/test_evidence.py`
- `results/tables/evidence_manifest.csv`
- the regenerated evidence tables and PNG/PDF/caption bundles
- this AI log

### Focused verification

1. Ruff:
   - Command: `../../.venv/bin/python -m ruff check --no-cache scripts/generate_evidence.py tests/test_evidence.py`
   - Exit 0: `All checks passed!`
2. Focused regression tests without pytest cache writes:
   - Command: `../../.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_evidence.py`
   - Exit 0: `7 passed in 6.78s`.
3. Evidence regeneration:
   - Command: `../../.venv/bin/python scripts/generate_evidence.py`
   - Exit 0: four tables regenerated; all six figure bundles regenerated;
     `6 PASS, 0 layout issues`.
4. Independent post-generation reconciliation:
   - All eight manifest sample periods matched independently derived source
     ranges.
   - Combined weights matched `2020-12-31 to 2023-11-30`.
   - Sector sentiment matched `2020-01-02 to 2023-12-29`.
   - All six `## Sample` caption values matched their manifest rows.
   - The four required source artifacts retained their previously recorded
     SHA-256 hashes, confirming that this metadata-only correction did not
     change core calculations or source CSV content.

The completion record below is based on the subsequent project conversation and
keeps the final economic interpretation reserved for the student's report.

## What I changed and why

### Student correction and confirmation — completed from the project record

- [x] I accepted the four-core-fund scope for this evidence pack and kept the
      exploratory 21-day sentiment fund outside the locked core comparison so
      it could not be presented as an independently specified test.
- [x] I reviewed the three-significant-digit report table against the
      full-precision `performance_metrics.csv`; rounding is presentation-only.
- [x] I accepted the six-ticker selection as a descriptive display rule because
      the remaining assets are explicitly grouped and every displayed stack is
      checked to sum to 100%. It is not a portfolio-construction rule.
- [x] I reviewed the matched base-versus-sentiment fusion evidence. I will keep
      the report's economic interpretation in my own words and will not infer
      predictability from association or from an exploratory result.

I also requested correction of the manifest sample periods so the weights and
sentiment exhibits report their own source-derived date ranges rather than
reusing the fund-return sample.
