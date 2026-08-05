# Prompt log - PortFoYou Streamlit investor journey

## What I wanted

Complete the PortFoYou application as a lightweight investor-facing interface
that reads only the existing precomputed artifacts. The required journey was fund
comparison, individual fact sheets, allocation, ten-sector sentiment analytics,
and methodology and risk explanations.

## Prompt

> Complete streamlit_app.py using the product name PortFoYou. The app must read
> only precomputed results/ artifacts. It must not run VADER, download a lexicon,
> recompute a backtest, import nltk, read raw local data, or use secrets.
>
> The core investor journey must include fund comparison, individual fund fact
> sheets, allocation across funds, ten-sector sentiment analytics, and methodology
> and risk explanations. Each fact sheet must show growth of $1, annualised return,
> annualised volatility, Sharpe, maximum drawdown, and current target holdings.
> Allocation must operate only across precomputed funds and clearly state that
> outputs are historical scenarios rather than personalised investment advice.
>
> Use @st.cache_data for artifact reads, project-relative paths, clear missing-file
> and schema errors, and consistent fund names. Keep the app lightweight:
> requirements.txt contains deployment dependencies only, while build-time nltk
> remains in requirements-dev.txt.
>
> Add tests/test_app_artifacts.py covering required files, schemas, fund names,
> dates, joins, and allocation calculations. Run app tests and the full pytest
> suite. Start Streamlit locally, inspect every main interaction and error state,
> then stop the server cleanly. Report commands, exit codes, actual checks, and
> limitations, and create an AI prompt log. Stop after reporting.

## What the assistant produced

- Replaced the raw-data starter with the PortFoYou app in `streamlit_app.py`.
- Added `src/app_artifacts.py` for four-file loading, typing, schema checks,
  cross-artifact reconciliation, holdings selection, drawdown, and transparent
  fixed-allocation historical scenarios.
- Added `tests/test_app_artifacts.py` with real-artifact, independent NumPy
  allocation, dependency split, error-state, and Streamlit interaction coverage.
- Added a light PortFoYou theme in `.streamlit/config.toml`.
- Restricted `requirements.txt` to Streamlit, pandas, and NumPy; moved the
  build/reproduction-only scientific packages to `requirements-dev.txt`, where
  the sentiment build dependency remains.

The app reads exactly:

1. `results/data/fund_returns.csv`
2. `results/data/fund_weights.csv`
3. `results/data/sector_sentiment_index.csv`
4. `results/tables/performance_metrics.csv`

It exposes the five artifact funds without renaming their identifiers. The
21-day coverage-adjusted sentiment fund is visibly labelled exploratory.

## Errors and risks identified

- The original starter imported the hosted-data helper and displayed raw equity
  prices. That would have violated the precomputed-artifact deployment design.
- Allocation controls can contain non-negative values that do not sum to 100%; the
  app blocks the calculation and displays the actual total in that state.
- Missing or malformed CSVs could otherwise fail with a low-level pandas error;
  the loader now reports the exact artifact or missing columns.
- A first live screenshot showed insufficient contrast because fixed navy text
  inherited a dark Streamlit theme. The app theme was locked to a coherent light
  palette and visually rechecked.
- The first daily-sentiment interaction still displayed a caption describing the
  rolling view. The caption is now conditional, and the focused test protects the
  corrected text.
- The first full-suite run was sandboxed from both hosted data URLs. This was an
  environment/network failure, not a test assertion failure. The same suite was
  rerun with network access and passed.
- The browser console recorded temporary health-check errors only while the local
  server was intentionally restarted or stopped. No Python app exception appeared
  in the stable server output.

## Verification evidence

1. Focused lint:

   ```text
   ../../.venv/bin/python -m ruff check --no-cache \
       src/app_artifacts.py streamlit_app.py tests/test_app_artifacts.py
   Exit 0: All checks passed!
   ```

2. Focused application tests:

   ```text
   ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
       tests/test_app_artifacts.py
   Exit 0: 9 passed in 1.21s
   ```

3. First full-suite attempt under restricted network:

   ```text
   ../../.venv/bin/python -m pytest -q
   Exit 1: 76 passed, 1 failed, 26 errors in 10.17s
   Cause: both official hosted-data URLs failed DNS resolution.
   ```

4. Network-enabled full-suite rerun before the final visual correction:

   ```text
   ../../.venv/bin/python -m pytest -q
   Exit 0: 103 passed in 33.11s
   ```

5. Final full-suite run after the theme and caption corrections:

   ```text
   ../../.venv/bin/python -m pytest -q
   Exit 0: 103 passed in 34.16s
   ```

6. Local application:

   ```text
   ../../.venv/bin/python -m streamlit run streamlit_app.py \
       --server.port 8501 --server.headless true \
       --browser.gatherUsageStats false
   Startup succeeded after local-port permission was granted.
   Final shutdown: exit 0.
   ```

The live inspection covered comparison, selection of the exploratory fact sheet,
five metrics and latest holdings, a valid A$100,000 equal-allocation scenario,
the ten-sector rolling and daily views, the empty-sector selection error, the
methodology/risk page, and an isolated temporary copy with the required
`fund_returns.csv` absent. The missing-file page named the exact relative path and
stated that the app does not rebuild results at runtime.

## Student correction and decision

- [ ] I reviewed the PortFoYou fund labels and investor-facing wording.
- [ ] I independently checked the allocation interpretation and disclaimer.
- [ ] I reviewed the exploratory-fund label and sentiment limitations.
- [ ] I accept or amend the visual design and risk presentation.

Student comments: ________________________________________________

No student approval, interpretation, or correction has been inferred.
