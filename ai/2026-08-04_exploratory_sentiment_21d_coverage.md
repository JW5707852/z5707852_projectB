# Prompt log - exploratory 21-day coverage-adjusted sentiment robustness extension

## What I wanted

Retain the locked `equity_sentiment_tilt` unchanged and add one explicitly
exploratory, horizon-aligned equity fund plus a temporal sector predictability
diagnostic. The extension must remain look-ahead safe, use predeclared parameters,
retain mixed or negative results, and not support a prediction claim without stable
evidence.

## Prompt(s)

The following user prompt is recorded in full:

> Implement one task-compliant exploratory sentiment improvement without replacing,
> retuning, or hiding the existing locked equity_sentiment_tilt result.
>
> Goal
>
> The existing one-day sentiment tilt is look-ahead safe and reproducible, but it
> uses one prior trading day’s sector sentiment to determine an approximately
> monthly holding period. It slightly reduced volatility and maximum drawdown but
> also slightly reduced annual return and Sharpe.
>
> Keep the existing equity_sentiment_tilt fund, its 0.10 strength, formulas,
> artifacts, and evidence unchanged. Add one clearly labelled exploratory fund
> that uses a lagged 21-equity-trading-day, coverage-adjusted sector sentiment
> signal aligned with the monthly holding horizon.
>
> Also add a sector-level predictability diagnostic so the project evaluates
> whether the signal relates to subsequent returns instead of inferring
> predictability from one portfolio comparison.
>
> Important interpretation constraint
>
> The 2021–2023 final results have already been inspected. Therefore: do not claim
> the new design is a newly untouched final out-of-sample test; label it
> exploratory or a robustness extension; do not search over windows, strengths,
> clipping bounds, missing-data rules, or formulas using final-period performance;
> lock the trailing window at 21 equity trading days, tilt strength at 0.10,
> z-score clipping at [-2, 2], and minimum expanding history at 60 daily
> observations; retain negative or mixed results; and do not claim predictability
> unless the temporal diagnostic is stable enough.
>
> Before editing, read completely PROJECT_BRIEF.md, AGENTS.md,
> context/DATA_GUIDE.md, context/project_context.md,
> context/verify_ai_output.md, SUBMISSION_CHECKLIST.md, src/sentiment.py,
> src/fusion.py, src/metrics.py, src/portfolios.py, scripts/run_part_b.py,
> tests/test_sentiment.py, tests/test_fusion.py, tests/test_metrics.py,
> tests/test_artifacts.py, results/data/sector_sentiment_index.csv,
> results/tables/fusion_comparison.csv, ai/2026-08-04_equity_sentiment_tilt.md,
> and relevant current outputs and AI logs. Inspect current status and preserve
> unrelated work.
>
> Before regenerating artifacts, copy the existing core CSVs to a temporary
> directory outside the repository and use them only to prove that the existing
> four funds remain numerically unchanged. Do not create scratch files inside the
> project.
>
> For every sector and equity trading date t, define the signal window as the
> previous 21 equity trading dates, strictly excluding t. Use only rows with
> observed sector news; missing sector-days remain missing. Calculate trailing
> sentiment as sum(raw_sector_compound * ticker_coverage_share) divided by sum of
> ticker_coverage_share. If the denominator is zero, the signal is missing.
> Calculate effective coverage as sum(observed_ticker_count) divided by
> 21 * possible_ticker_count, where possible_ticker_count is five for each supplied
> sector, and require the result to remain in [0, 1].
>
> From the daily trailing-sentiment series, calculate an expanding prior mean and
> sample standard deviation using observations strictly earlier than t, ddof=1,
> and at least 60 prior non-missing values. Standardise the current trailing value,
> clip to [-2, 2], and then multiply by sqrt(effective_coverage). Record the latest
> raw-news date used; it must be strictly earlier than t.
>
> At each existing monthly decision date, use multiplier = 1 + 0.10 *
> coverage_adjusted_z. A missing signal remains missing in audit fields and uses a
> multiplier of one. For each stock, multiply its equity_equal_weight base weight
> by its sector multiplier and renormalise all 50 stock weights.
>
> The result must remain equity-only, long-only, fully invested, finite, on the
> same decision and holding dates as equity_equal_weight, use a 0% risk-free rate,
> 252 annualisation, geometric annual return, and 0 bps transaction costs.
>
> Add stable identity fund=equity_sentiment_21d_coverage_tilt,
> asset_family=equity, method=exploratory_sentiment_21d_coverage_tilt. Do not
> rename or remove combined_equal_weight, combined_min_variance,
> equity_equal_weight, or equity_sentiment_tilt. Their return, weight, and
> performance rows must remain numerically unchanged.
>
> Retain audit fields for the 21-day window dates, trailing sentiment, effective
> coverage, expanding prior mean/std/count, raw/clipped/coverage-adjusted z-scores,
> tilt strength and multiplier, base and unnormalised weights, missing-signal flag,
> and exploratory method label.
>
> Retain and regenerate fund_returns.csv, fund_weights.csv,
> sector_sentiment_index.csv, performance_metrics.csv, fusion_comparison.csv, and
> fusion_sector_exposure.csv. Add sector_sentiment_21d_coverage.csv,
> fusion_exploratory_comparison.csv, sentiment_predictability.csv, and
> sentiment_predictability_summary.csv. Do not change raw_sector_compound.
>
> Evaluate the locked and exploratory signals separately. At each monthly decision,
> use the available signal, calculate each sector’s equal-weight compounded return
> over the following actual holding period, calculate the 50-stock equal-weight
> market return, and define excess as (1 + sector return) / (1 + market return) - 1.
> Store one row per decision, sector, and signal method with decision and holding
> dates, sector, signal method, available signal, effective coverage, sector return,
> market return, and excess return. Calculate monthly cross-sectional Spearman rank
> IC where enough sectors are available.
>
> Summarise time-ordered expanding periods 2021-01-04 to 2021-12-31,
> 2021-01-04 to 2022-12-30, and 2021-01-04 to 2023-12-29. Report monthly decisions,
> usable sector observations, mean and median IC, positive-IC share, signal/news
> coverage, and sector results where samples permit. Do not use shuffled K-fold or
> these diagnostics to retune the 21-day window or 0.10 strength.
>
> Put reusable logic in src/ and writing/orchestration in scripts/run_part_b.py.
> Keep the app on precomputed artifacts; do not add dependencies, edit context
> files or src/data_access.py, or read Project A at runtime.
>
> Add executable, independent NumPy/Pandas tests for: exclusion of current/future
> dates; source-window timing; missing-news treatment; hand coverage arithmetic;
> prior-only ddof=1 statistics; future perturbation; zero-signal placebo; three
> official manual rebalances; finite equity-only sum-to-one weights; matched dates
> and rebalances; unchanged original four funds; strictly subsequent diagnostic
> returns; row conservation; cross-sectional monthly Spearman; absence of
> final-period parameter selection; artifact schemas/mappings; and latest current
> holdings. Do not use tautological expected values.
>
> Run, in order, targeted Ruff, targeted pytest, scripts/run_part_b.py, targeted
> pytest again, and then the full pytest suite. Do not run Streamlit or
> check_handin.py. All final commands must exit zero.
>
> Update or create an AI prompt log with the full prompt, affected files, timing
> risks, locked-parameter reason, actual commands and exits, results, limitations,
> errors/corrections, and fields requiring student review. Do not fabricate student
> interpretation, approval, or correction. Report all evidence and stop without
> editing the report, app, Git configuration, or deployment.

## What the assistant produced

- Added a deterministic prior-21-trading-day coverage-weighted signal builder in
  `src/sentiment.py`.
- Added a separate exploratory fusion path and sector-return/rank-IC diagnostic in
  `src/fusion.py`; the locked one-day function was not refactored or replaced.
- Extended `scripts/run_part_b.py` to append the fifth fund only after building and
  exactly reconciling the locked four-fund artifacts.
- Added direct arithmetic, future-perturbation, placebo, official rebalance,
  predictability, schema, and current-holdings tests.
- Regenerated the required artifacts and added four clearly named exploratory
  evidence artifacts.

Affected code/tests/evidence:

- `src/sentiment.py`
- `src/fusion.py`
- `scripts/run_part_b.py`
- `tests/test_sentiment.py`
- `tests/test_fusion.py`
- `tests/test_predictability.py`
- `tests/test_artifacts.py`
- the required fund CSVs under `results/`
- four new exploratory CSVs under `results/`
- this AI log

The 21-day window and 0.10 strength were predeclared in the user prompt after the
final period had already been inspected. The production function rejects alternate
window, minimum-history, clipping, and strength values; the diagnostic records that
the final period was not used for parameter selection.

## Actual verification record

- Initial targeted Ruff: exit 1 because of line length and `__all__` ordering;
  formatting was corrected. Final targeted Ruff: exit 0, `All checks passed!`.
- Initial sandboxed targeted pytest: exit 1 because hosted data could not resolve,
  plus a duplicated-label mismatch and a fixture that counted an intentionally
  missing first trailing value. The two code/test issues were corrected and the
  prescribed command was rerun with hosted-data network access.
- Final corrected pre-build targeted pytest: exit 0, `46 passed in 22.82s`.
- Official build: exit 0; five funds, 753 dates, and 36 rebalances written.
- Final post-build targeted pytest: exit 0, `46 passed in 22.68s`.
- Final full pytest: exit 0, `80 passed in 26.78s`.
- External baseline comparison: exit 0; original four-fund subsets had maximum
  numeric absolute difference 0.0 for 3,012 return rows, 7,920 weight rows, and four
  performance rows. The locked sector index, fusion comparison, and fusion sector
  exposure retained byte-identical SHA-256 hashes.

## Observed before/after evidence

Over 2021-01-04 to 2023-12-29 (753 observations, 36 rebalances):

- equity_equal_weight: annualised return 0.126435, volatility 0.161662,
  Sharpe 0.817387, maximum drawdown -0.203219.
- locked equity_sentiment_tilt: annualised return 0.125453, volatility 0.161588,
  Sharpe 0.812289, maximum drawdown -0.202606.
- exploratory 21-day coverage tilt: annualised return 0.124097, volatility
  0.161788, Sharpe 0.804040, maximum drawdown -0.203703.

The exploratory fund weakened all four headline performance measures relative to
the equal-weight base. Average monthly target turnover was 0.031315, total target
turnover was 1.096027, mean absolute sector-exposure difference was 0.005229, and
the maximum absolute difference was 0.019560. Costs remain fixed at zero, so
turnover is descriptive rather than deducted.

Mean monthly cross-sectional rank IC for the exploratory signal was -0.109091
through 2021, -0.085354 through 2022, and -0.086195 through 2023. Corresponding
locked-signal values were -0.143561, -0.062332, and -0.018659. This is unstable,
negative evidence and does not support a claim that sentiment predicts subsequent
sector excess returns.

## What was wrong or risky

- Timing: an inclusive rolling operation could have leaked date t. Direct window
  bounds and future-perturbation tests guard against this.
- Missingness: no-news sector-days could have been treated as zero. The numerator
  and denominator include only observed-news rows, while effective coverage retains
  the missing ticker opportunities.
- Duplicated constants: the first implementation used different exploratory label
  text in two modules; the validator correctly rejected it. One shared constant now
  supplies the label.
- Independent fixture: the first hand calculation counted a NaN as an observation;
  the fixture now filters finite prior trailing values just as the stated
  non-missing-observation rule requires.
- Evidence audit: the first sector-level summary rows inherited method-wide
  coverage values. The underlying decision-sector data and ICs were correct; the
  summary now recalculates coverage within each sector, with a direct test.
- Hosted data: the first sandboxed run failed DNS resolution. It was rerun through
  the unchanged protected `src/data_access.py` with allowed network access.
- Interpretation: the final sample was already inspected, parameters were not
  tuned, the evidence is negative, and no causal or predictive claim is justified.
- The Streamlit memory-cache warnings during the build are informational because
  the loader ran outside a Streamlit runtime.

## What I changed and why

Change nothing.
