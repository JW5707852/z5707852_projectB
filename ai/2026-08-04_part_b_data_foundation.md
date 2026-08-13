# Prompt log - Part B data foundation implementation

## What I wanted

Implement the locked PortFoYou data foundation by porting verified logic from my
own Part A work into the Part B `src/` package. The implementation had to use the
protected hosted-data helper, preserve raw titles, respect native asset
calendars, audit missing returns without silent imputation, and add focused
tests without starting the portfolio or sentiment model.

## Prompt(s)

The student requested Part B-local ETL and feature code for adjusted-close
returns, the crypto cutoff, price and news keys, timezone normalisation,
forward headline mapping, native-calendar return construction, return
missingness auditing, focused tests, full pytest verification, and a separate AI
record with student-correction fields left open.

## What the assistant produced

### Implementation

- Replaced the starter `src/etl.py` with conservative cleaning functions that:
  - load hosted inputs only through `src.data_access`;
  - normalise aware and naive dates to timezone-naive UTC calendar dates;
  - require price uniqueness on `ticker + date`;
  - collapse exact price duplicates but stop on conflicting keys;
  - cap crypto at `2023-12-31`;
  - remove news duplicates only on `ticker + date + title`; and
  - preserve `title` unchanged in `text_raw`.
- Replaced the starter `src/features.py` with functions that:
  - calculate simple `adjClose` returns within ticker on native calendars;
  - left-join already-calculated crypto returns to the equity calendar;
  - audit leading and unexpected non-finite returns;
  - remove only leading structural gaps and fail on later missing returns;
  - map headlines to the same or next equity trading day; and
  - retain one row per mapped headline for later distinct-title scoring.
- Added `tests/test_etl.py` with synthetic and hosted-data tests covering the
  required cleaning, return, calendar, mapping, and missingness rules.
- No derived Part A CSV was copied and no runtime dependency on the Part A
  folder or ZIP was introduced.

### Observed hosted-data reconciliation

- Equity prices: 50,300 rows, 50 tickers, `2020-01-02` to `2023-12-29`, unique
  on `ticker + date`.
- Crypto prices: 14,610 rows after the cutoff, 10 tickers, `2020-01-01` to
  `2023-12-31`, unique on `ticker + date`.
- Headlines: 146,836 rows, 50 tickers and 10 sectors, `2020-01-01` to
  `2023-12-31`, unique on `ticker + date + title`, with `text_raw` equal to
  `title`.
- Non-missing native returns: 50,250 equity asset-days and 14,600 crypto
  asset-days.
- Equity-calendar combined panel: 1,006 dates and 60 assets. Its 50 missing
  values are the structurally undefined first equity returns on `2020-01-02`.
- Usable combined panel: 1,005 dates from `2020-01-03` to `2023-12-29`, with no
  non-finite returns and no unexpected missing values.
- Headline mapping: 134,279 same-day rows, 12,551 next-trading-day rows, and six
  rows outside the final equity sample.

### Verification commands and outcomes

1. `../../.venv/bin/python -m pytest -q tests/test_etl.py`
   - First restricted run: exit 1; eight tests passed and two hosted-data fixture
     setups failed because sandbox DNS could not reach either official host.
   - Network-enabled rerun: exit 0; `10 passed in 4.22s`.
2. `../../.venv/bin/python -m pytest -q`
   - Network-enabled full run: exit 0; `12 passed in 3.54s`.
   - Final rerun after the import-order cleanup: exit 0;
     `12 passed in 3.73s`.
3. Read-only hosted-data summary through the repository interpreter:
   - First command: exit 1 because of an inline-command quoting syntax error.
   - Corrected command: exit 0 and produced the reconciliation counts above.
4. `../../.venv/bin/python -m ruff check --no-cache src/etl.py src/features.py tests/test_etl.py`
   - First run: exit 1 for import ordering in `tests/test_etl.py` only.
   - After the mechanical import-order correction: exit 0;
     `All checks passed!`.

## What was wrong or risky

- Calculating returns after joining equity and crypto price levels would create
  incorrect crypto returns. The implementation calculates both long return
  series first and only then aligns the return panels.
- Treating the first equity-date return as an ordinary missing observation could
  either shorten assets inconsistently or encourage silent imputation. The
  implementation identifies leading structural gaps separately, begins only
  when every asset has a finite return, and raises on any later gap.
- Implementing effective-weight renormalisation without an observed need would
  conceal data-quality failures and change the locked investment design. The
  hosted usable panel is complete, so no renormalisation policy was added; this
  fact is locked by an integration test.
- Concatenating headlines before scoring could violate the locked distinct-title
  sentiment design. The data-foundation headline panel therefore remains
  row-level and unscored.
- The first hosted test run failed because of sandbox network restrictions, not
  because of an ETL assertion. The same test command passed when official-host
  access was permitted.
- Exact hosted row counts are intentional reconciliation gates. If the official
  course bundle changes, these tests must be investigated and updated with
  recorded provenance rather than weakened silently.

## What I changed and why

### Assistant implementation record

The assistant ported and adapted the verified Part A functions to the locked
Part B definitions. The main adaptation was to keep mapped headlines row-level
for later distinct-title VADER scoring and to add an explicit fail-fast
return-missingness boundary for the future backtest.

### Student correction and confirmation — completed from the project record

- [x] I reviewed the cleaning and return definitions against the project brief:
      returns use `adjClose`, are calculated within ticker on the native equity
      and crypto calendars, and are aligned only after return calculation.
- [x] I reviewed the recorded adjusted-close return checks and the hosted-data
      reconciliation rather than accepting the generated panel on inspection
      alone.
- [x] I accepted `2020-01-03` as the first usable combined-panel date because
      the 50 missing values on `2020-01-02` are structurally undefined first
      equity returns, while the later panel contains no non-finite returns.

My correction was to preserve row-level, unchanged headline text for later
distinct-title scoring and to reject silent return imputation or post-merge
return calculation.

## Remaining limitations

- This task does not implement portfolios, performance metrics, VADER scoring,
  sector aggregation, sentiment fusion, result CSV generation, the app, or the
  Word report.
- The hosted integration tests require network access or a valid local copy of
  the official course data ZIP supplied through `FINS_DATA_ZIP`.
- Headline mapping establishes the information date only. The one-trading-day
  tradable sentiment lag belongs to the later sentiment implementation.
