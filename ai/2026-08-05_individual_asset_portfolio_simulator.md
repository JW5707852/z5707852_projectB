# Prompt log - individual-asset portfolio simulator correction

## What I wanted

Correct PortFoYou's Portfolio Simulator so an investor can add specific stocks
or cryptoassets one row at a time, set target weights, and inspect the weighted
portfolio's historical performance and current asset-type mix.

## Prompt

> The Portfolio Simulator does not meet the requirement. I need to choose a
> specific stock rather than a broad financial asset, add rows with a plus
> button, select a stock or cryptoasset in each row, and enter its weight. The
> portfolio should then show the historical behaviour of the weighted portfolio
> as if I had built my own fund, plus the current proportions by asset type.

## What the assistant produced

- Preserved the required published-fund allocation journey as **Fund Allocation**.
- Rebuilt **Portfolio Simulator** around 2-15 individual holding rows. Each row
  selects Stock or Crypto, a concrete ticker, and a long-only target weight.
- Added `+ Add asset`, `Remove`, and `Use equal weights` controls with duplicate,
  positive-holding, weight-bound, and 100% total validation.
- Added `src/custom_portfolio.py` for deterministic artifact validation and
  fixed-weight portfolio arithmetic.
- Added `results/data/investable_asset_returns.csv`, containing 60 individual
  assets and 60,300 complete ticker-date return rows from 3 January 2020 to
  29 December 2023.
- Wired `scripts/run_part_b.py` to reproduce the new artifact from adjusted-price
  returns calculated on native equity and crypto calendars before alignment.
- Added historical portfolio value, annual geometric return, annual volatility,
  Sharpe ratio, maximum drawdown, a stock-versus-crypto donut, and a detailed
  holdings table.

## What was wrong or risky

- The previous Portfolio Simulator allocated across prebuilt funds, not specific
  securities, so it did not implement the requested self-built fund workflow.
- Calculating crypto returns after merging prices to equity dates would create
  false multi-day returns. The new artifact reuses the tested native-calendar
  return functions and aligns returns only afterward.
- Allowing duplicate tickers, totals other than 100%, negative weights, leverage,
  or only one positive position would make the displayed portfolio misleading.
- A weighted historical path can look like a recommendation. The interface
  explicitly labels it a historical constant-weight research simulation with
  0% risk-free rate and 0 bps assumed costs, not advice or optimisation.
- The restricted test environment initially blocked the official data URLs.
  The affected build and ETL checks were rerun with approved network access.

## Verification evidence

1. Reproducible targeted build to a temporary output:

   ```text
   rows=60300 tickers=60 sample=2020-01-03..2023-12-29
   path=/tmp/portfoyou_investable_asset_returns.csv
   Exit 0
   ```

2. Focused component and direct-regression tests:

   ```text
   ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
       tests/test_build.py tests/test_etl.py tests/test_custom_portfolio.py \
       tests/test_app_charts.py tests/test_app_artifacts.py
   41 passed in 5.85s
   Exit 0
   ```

3. Focused lint:

   ```text
   ../../.venv/bin/python -m ruff check --no-cache \
       streamlit_app.py src/custom_portfolio.py src/app_charts.py \
       scripts/run_part_b.py tests/test_custom_portfolio.py \
       tests/test_app_artifacts.py tests/test_app_charts.py
   All checks passed!
   Exit 0
   ```

4. Local Streamlit interaction and visual check:

   ```text
   ../../.venv/bin/python -m streamlit run streamlit_app.py \
       --server.port 8512 --server.headless true
   Startup succeeded; local UI inspection found no browser errors.
   The Portfolio Simulator showed NVDA and BTC-USD rows, + Add asset created a
   third ticker row, and the history, metrics, 50/50 asset-type mix and holdings
   table rendered correctly. Server shutdown: exit 0.
   ```

The app-facing asset CSV was reduced from 7.5 MB to 3.0 MB by keeping the five
non-redundant contract columns and documenting invariant provenance and calendar
rules in code and visible methodology copy rather than repeating them on every
row.

The tests independently cover native-calendar crypto returns, schema and common
dates, one hand-calculated portfolio, compounding, annualisation, volatility,
Sharpe, drawdown, invalid weights, row addition, manual edits below and above
100%, equal-weight recovery, the retained fund-allocation page, and the final
asset-type mix.

## Student correction and decision — completed from the project record

- [x] I requested the individual-security workflow after determining that the
      earlier simulator only allocated across prebuilt funds. I accepted the
      reproducible universe of 50 supplied US equities and 10 supplied
      cryptoassets, with NVDA and BTC-USD as an editable demonstration rather
      than a recommendation.
- [x] I reviewed the constant-mix interpretation: each daily portfolio return is
      the fixed target-weight sum of the selected assets' common equity-calendar
      returns, so the scenario implies continuous rebalancing back to target
      weights. It is long-only, fully invested, uses 252 annualisation periods,
      a 0% risk-free rate, and 0 bps assumed costs.
- [x] I accepted the row-based Stock/Crypto selector, add/remove controls,
      equal-weight shortcut, 100% validation, historical value chart, asset-type
      allocation, and holdings table as the clearest implementation of the
      requested self-built fund workflow.

My limitation is that this is a historical research simulation, not an
optimised portfolio, forecast, live trading tool, or personalised investment
recommendation. Taxes, fees, slippage, and real rebalancing costs are excluded.
