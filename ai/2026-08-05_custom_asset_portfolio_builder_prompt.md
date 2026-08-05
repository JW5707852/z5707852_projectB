# Implementation prompt — custom asset portfolio builder

## Goal

Add a new **Custom portfolio** journey to PortFoYou. It must let an investor
select individual US equities and cryptoassets from the project's existing
investable universe, assign manual long-only weights, and inspect a transparent
historical fixed-allocation scenario. The feature is an educational research
tool, not a suitability assessment, recommendation engine, or live trading tool.

## Context

- The entrypoint is `streamlit_app.py`.
- The deployed app currently reads validated, precomputed files under `results/`
  and performs only display and allocation arithmetic.
- The five published PortFoYou funds and their stable fund IDs must remain
  unchanged.
- The existing **Allocate across funds** page allocates across published funds;
  it does not provide asset-level selection.
- The build pipeline is `scripts/run_part_b.py`. Reusable financial logic belongs
  in `src/`; focused tests belong in `tests/`.
- App-readable artifacts must use project-relative paths and be reproducible from
  a clean checkout. The deployed app must not load raw hosted data, call
  `src/data_access.py`, rebuild a backtest, run VADER/FinBERT, import NLTK, call an
  external API, or require secrets.

## Constraints and implementation requirements

1. Add a build-time artifact such as
   `results/data/investable_asset_returns.csv` with at least `date`, `ticker`,
   `asset_group`, `sector`, and `daily_return`. Document and validate its schema,
   units, sample dates, uniqueness key, and provenance.
2. Build asset returns from `adjClose`. Calculate equity and crypto simple returns
   within ticker on their native calendars first; then left-join crypto returns to
   the equity trading calendar. Never calculate returns across a merged price
   panel. Cap crypto prices at 2023-12-31.
3. Add **Custom portfolio** to the investor navigation without changing the
   existing **Allocate across funds** behaviour.
4. Let the investor select 2–15 assets. Show ticker, asset group, and sector where
   available. Provide a one-click equal-weight starting point and editable manual
   weights.
5. Enforce finite weights from 0% to 100%, a total of exactly 100% within a clear
   numerical tolerance, long-only exposure, no leverage, and no short positions.
   Display the current total and block calculation when validation fails.
6. Keep the calculation a fixed-weight historical scenario: daily portfolio return
   is the selected asset-return matrix multiplied by the user weights, and wealth
   compounds from an investor-entered AUD amount. State the common equity-calendar
   sample, missing-return policy, 252-day annualisation, 0% risk-free rate, and
   0 bps transaction-cost research assumption beside the result.
7. Do not add automatic optimisation or language implying that the selected mix is
   suitable, recommended, optimal for the person, or expected to outperform.
8. Show at minimum: growth of A$1 or the entered amount, annualised geometric
   return, annualised volatility, Sharpe ratio, maximum drawdown, an allocation
   donut, and an asset breakdown table. Use the existing professional visual
   system and responsive external chart legends so full asset names remain visible
   on desktop and 390 px mobile screens.
9. Cache only artifact reads. Put schema validation and portfolio arithmetic in
   small deterministic functions that can be tested without Streamlit.
10. Update visible methodology/provenance copy so it accurately distinguishes the
    four required fund artifacts from the additional asset-return artifact used by
    the custom builder. Retain the historical-simulation and implementation-gap
    warnings.
11. Record this implementation in a separate `ai/` prompt log. Do not alter the
    provided `context/` files or `src/data_access.py`.

## Done when

- `scripts/run_part_b.py` reproducibly creates the new asset-return artifact.
- Tests independently verify adjusted-price returns, native-calendar calculation
  before alignment, schema and uniqueness, common sample dates, weight validation,
  one hand-calculated portfolio day, compounding, annualisation, Sharpe, and
  drawdown.
- Streamlit `AppTest` covers asset selection, equal-weight initialization, manual
  edits, totals below/above 100%, a valid scenario, and client-facing warnings.
- The existing fund comparison, fact sheets, fund allocation, sentiment, and
  methodology views still pass their focused tests.
- A local desktop and 390 px browser check confirms that controls, tables, chart
  labels, and legends are readable without horizontal clipping.
- No raw data, build-only dependency, secret, absolute local path, or runtime
  model/backtest code is introduced into the deployed app.

