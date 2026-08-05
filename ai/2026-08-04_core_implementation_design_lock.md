# Prompt log - PortFoYou core implementation design lock

## Status

Locked for the core implementation. No code, tests, results, app, or report files
were changed in this task. These decisions should change only if the student
explicitly revises them or a formal `PROJECT_BRIEF.md` requirement requires an
adjustment. Any such adjustment must be recorded with its reason.

## What I wanted

Lock a feasible, look-ahead-safe core design for PortFoYou before coding. The
product is for individual investors with basic financial literacy who want to
compare transparent, systematically managed funds. The scope is deliberately
limited to the assessed core journey: fund comparison, fund fact sheets,
allocation across funds, and sentiment analytics for the ten equity sectors.

## Prompt(s)

The student specified the product scope, four fund identifiers, walk-forward
backtest conventions, VADER sentiment method, missing-news treatment, expanding
standardisation, and sentiment-tilt formula. The student explicitly excluded
optional innovation such as AI chat, forecasting, KNN, logistic regression,
lexicon extensions, leverage, short selling, volatility targeting, and extra
asset families, and asked for a consistency and assessment-coverage review
without starting implementation.

## Confirmed core design

### Product and user journey

- Product name: **PortFoYou**.
- Target user: an individual investor with basic financial literacy who wants
  transparent comparison of systematically managed funds.
- The app will read precomputed Part B artifacts and support only:
  1. comparison of the four funds;
  2. one fact sheet for each fund;
  3. user allocation across the funds; and
  4. time-series sentiment analytics for all ten equity sectors.
- The app will not run the backtest or VADER and will not import `nltk`.
- No AI chat, price forecast, KNN, logistic regression, custom lexicon,
  leverage, short selling, volatility targeting, or additional asset family
  will be added to the core implementation.

### Fund universe and identifiers

The four investable funds are:

1. `combined_equal_weight` - equal weight across the combined equity and crypto
   universe;
2. `combined_min_variance` - long-only minimum-variance weights across the same
   combined universe;
3. `equity_equal_weight` - equal weight across the equity universe; and
4. `equity_sentiment_tilt` - the sentiment-tilted version of
   `equity_equal_weight`.

Every fund will have a separate fact sheet. The two combined funds satisfy the
formal minimum of one combined family with at least two methods. The two equity
funds provide the matched base-versus-fusion comparison required for sentiment
fusion. All four funds will use the same eligible out-of-sample equity-calendar
dates so their performance comparisons are aligned.

### Return construction and backtest timing

- Returns use `adjClose` and are calculated within ticker.
- Equity and crypto returns are calculated on their own native calendars first.
  Only the already-calculated crypto returns are left-joined to the equity
  trading calendar for combined funds. Price levels are never merged before
  return calculation.
- Equity and combined funds use 252 trading days for annualisation.
- Rebalancing is monthly. The decision date is the last equity trading day of
  each month and the target weights first apply to the next equity trading day.
- The estimation window is expanding and must contain at least 252 finite
  equity-calendar return observations before the first decision is eligible.
- The exact first live holding date will be produced by the deterministic
  calendar rule and recorded after the pipeline is built; it will not be
  hard-coded before verification.
- Information through the month-end decision date may be used because the first
  holding return is on the following trading day. The rebalance audit must show
  the decision date, training start and end, observation count, first holding
  date, solver status, and target weights.
- Constraints are long-only and fully invested: each weight is finite and
  non-negative, and weights sum to one within tolerance.
- Risk-free rate: 0% per year.
- Transaction costs: 0 basis points. Portfolio returns are therefore reported
  before transaction costs, with the zero-cost assumption stated. No turnover
  or cost model will be presented as an innovation.
- Missing-return policy: discard the unavoidable first return for each ticker
  before forming the usable panel. Once the live sample begins, require finite
  aligned returns for every asset used by a fund and stop the build with a clear
  audit failure if this condition is violated; do not silently impute a missing
  live return as zero.

### Portfolio methods

- Equal-weight target weights are `1 / N` for the eligible assets and are logged
  on the common monthly rebalance schedule.
- Minimum variance minimises `w' covariance w` using the expanding historical
  return window, subject to long-only and fully invested constraints.
- The covariance objective may be annualised or otherwise multiplied by a
  positive constant for numerical conditioning because this does not change the
  mathematical minimum. Solver success, objective value, finite weights,
  constraint compliance, and variation over rebalances must be audited.

### Performance definitions

- Growth of $1 is the cumulative product of `1 + daily fund return`.
- Annualised return is the geometric annual return:
  `(product(1 + r) ** (252 / number_of_returns)) - 1`.
- Annualised volatility is the sample standard deviation of daily fund returns
  multiplied by `sqrt(252)`.
- Sharpe ratio uses a 0% risk-free rate and is the daily arithmetic mean divided
  by daily sample volatility, multiplied by `sqrt(252)`.
- Maximum drawdown is the minimum of cumulative wealth divided by its running
  maximum minus one.
- Current holdings are the target weights from the most recent rebalance.

### Sentiment index

- Model: NLTK VADER `SentimentIntensityAnalyzer`, using the package's standard
  lexicon with no extension. The reported score is VADER `compound`.
- Input text is the unchanged `text_raw` title. Casing, punctuation, negation,
  stop words, and intensifiers are preserved for scoring.
- Each distinct title is scored once at build time and the score is joined back
  to the clean ticker rows. Join validation must prove that no clean rows were
  lost or multiplied.
- Headlines are first mapped to the same equity trading day or the next equity
  trading day. Headlines outside the final equity sample remain unmapped.
- Headline compound scores are averaged to one raw score per
  `trading_date + ticker + sector`.
- The sector index then takes an equal-weight arithmetic mean of the observed
  ticker-day scores in each `trading_date + sector`. It does not take a raw
  headline-weighted sector mean.
- Ticker-days without headlines remain missing. They are not filled with zero
  and are not carried forward. The output reports observed ticker count,
  possible ticker count, and coverage for each sector-day.
- A sector-day with no observed ticker score remains missing.
- The standalone raw sector index remains descriptive and unlagged for display.
  The tradable version is explicitly separated from it.

### Tradable sentiment timing and standardisation

For each sector and equity trading date `t`:

1. calculate the raw sector score for `t` from headlines mapped to `t`;
2. calculate its expanding z-score using only earlier raw sector observations,
   never the current or future observation, and require at least 60 earlier
   non-missing sector observations from prior equity trading dates;
3. clip the resulting z-score to `[-2, 2]`; and
4. shift that clipped value forward by one equity trading day before it can be
   used in a portfolio decision.

Equivalently, the signal available on decision date `t` can contain raw
sentiment information only from `t-1` or earlier. A Saturday or Monday headline
mapped to Monday is first usable on Tuesday. If the expanding history is shorter
than 60 observations, the tradable signal is missing.

### Equity sentiment fusion

- Baseline: `equity_equal_weight` on exactly the same dates and rebalance
  schedule.
- At each month-end decision, map every equity ticker to its sector and use the
  tradable sector z-score available on that decision date.
- For base weight `w_i` and available sector signal `z_s`, calculate the
  unnormalised tilted weight as `w_i * (1 + 0.10 * z_s)`.
- Because `z_s` is clipped to `[-2, 2]`, the multiplier lies in `[0.8, 1.2]` and
  cannot create a negative weight.
- If a ticker's sector signal is missing, its multiplier is one, meaning no
  tilt rather than assumed neutral sentiment.
- Renormalise all unnormalised weights to sum to one.
- Compare the base and tilted funds using the same sample, return inputs,
  rebalance decisions, holding dates, constraints, annualisation, risk-free
  rate, and zero-cost assumption.

### Required artifacts and evidence

The build must create these exact files from Part B-local code:

- `results/data/fund_returns.csv`;
- `results/data/fund_weights.csv`;
- `results/data/sector_sentiment_index.csv`; and
- `results/tables/performance_metrics.csv`.

At minimum, the sentiment artifact must distinguish the raw sector score from
the clipped, lagged tradable z-score and include coverage fields. Fund returns
and weights must retain stable fund identifiers and dates sufficient to
reproduce a sampled daily fund return by hand.

The assessment evidence remains limited to the required performance table,
growth-of-$1 comparison, drawdown figure, weights-over-time figure, risk-return
or Sharpe comparison, ten-sector sentiment series, and fusion before-versus-
after table and figure. These are core evidence, not optional innovation.

## Internal consistency and feasibility review

- **Consistent:** both combined methods use the same 60-asset universe and
  equity calendar, satisfying the required combined family comparison.
- **Consistent:** all funds share a monthly schedule and common OOS dates; the
  base-versus-tilt comparison is therefore matched.
- **Look-ahead safe by design:** estimation ends on the decision date, holdings
  begin next day, sentiment standardisation uses prior observations only, and
  the tradable signal is shifted one additional trading day.
- **Constraint safe by design:** the clipped tilt multiplier remains positive,
  and renormalisation preserves full investment.
- **Feasible:** approximately four years of supplied data provide enough history
  for a 252-observation initial portfolio window and 60-observation expanding
  sentiment standardisation, while monthly rebalancing keeps optimisation work
  modest.
- **Deployment feasible:** VADER runs only in the reproducible build; Streamlit
  reads the precomputed artifacts.
- **Scope trade-off:** excluding optional innovation is compatible with the
  formal minimum, although it intentionally does not pursue the higher-band
  innovation opportunities described in the brief.

## Assessment coverage review

- Combined equity-plus-crypto family with at least two methods: covered.
- Walk-forward OOS backtest with no look-ahead: covered by the timing rules.
- Fact sheet per fund: covered for all four identifiers.
- Standalone ten-sector sentiment index: covered.
- Look-ahead-safe sentiment fusion affecting equities only: covered.
- Required app journey: covered without extra features.
- Required metrics, figures, exact output filenames, reproducible build, Word
  report, public repository, live app, and AI workflow evidence: retained as
  acceptance requirements for later implementation.

## What was wrong or risky

- The original instruction did not explicitly repeat the formal requirement to
  score each distinct title once before aggregation. This has been added as a
  required implementation detail, not optional innovation.
- "Expanding z-score using at least 60 prior trading days" could otherwise be
  implemented using the current observation in its own mean and standard
  deviation. The locked definition uses strictly earlier non-missing sector
  observations, then shifts the resulting signal one trading day.
- A missing sector signal could be confused with neutral sentiment. The locked
  rule treats it as unavailable and applies no tilt while preserving missingness
  and coverage in the sentiment artifact.
- A missing-return policy was not included in the initial prompt, but the formal
  instructions require it to be stated. The locked policy is fail-fast for any
  missing live return rather than silent zero imputation.
- Part A descriptive return statistics cannot be reused as fund performance
  metrics. The locked definitions distinguish geometric annual return from the
  arithmetic-mean Sharpe calculation.
- The exact first live date must be generated from the clean return calendar; it
  has not been invented or hard-coded at the design stage.

## What I changed and why

The student's explicitly supplied product scope, fund list, timing conventions,
sentiment method, tilt strength, and exclusions were retained. The assistant
added only formal-requirement clarifications needed to make the design
implementable and auditable: title-level scoring, native-calendar return
construction, a precise no-look-ahead z-score order, a live-sample
missing-return policy, consistent performance formulas, matched OOS dates, and
required audit fields. No optional innovation was added.

This record describes the agreed design but does not claim that any component
has been implemented or tested. The student's later verification and personal
reflection must be recorded in their own words after implementation evidence is
available.
