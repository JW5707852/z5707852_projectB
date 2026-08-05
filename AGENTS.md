# FINS5545 Project B - AI Agent Instructions

## Authority and project scope

This is my individual FINS5545 Project B submission: Funds, Sentiment & App
(Data Factory Floor Stations 3-4). Before substantive work, read:

- `PROJECT_BRIEF.md`;
- `context/DATA_GUIDE.md`;
- `context/project_context.md`;
- `context/verify_ai_output.md`;
- `SUBMISSION_CHECKLIST.md`; and
- the relevant existing code, tests, outputs, report files, and AI logs.

Use this order of authority when sources differ:

1. `PROJECT_BRIEF.md` and the provided `context/` files;
2. this `AGENTS.md` and an explicit instruction from me;
3. my own Part A work in `FINS5545/z5707852_projectA.zip`;
4. the teaching examples under `FINS5545/`.

`FINS5545/` is read-only course reference material, not project source code or a
submission dependency. Learn from it, but do not copy its outputs, numerical
conclusions, or code without adapting and testing them. The formal project brief
wins over a lecture example. In particular:

- for this project, calculate equity and crypto returns on their own calendars
  first, then left-join crypto returns to the equity trading calendar; do not use
  the Week 5 teaching example that aligns price levels and differences afterward;
- build the sector sentiment index from ticker-day scores and equal-weight the
  tickers, not by taking a raw headline-weighted sector mean as in the Week 9
  demonstration.

The Part A ZIP is my own work and may be reused, but treat it as material to
inspect and verify, not as unquestioned truth. Port required logic into this Part
B folder; do not make the app or build pipeline depend on reading the ZIP at
runtime. Do not inspect or copy another student's project.

Work only inside this project folder unless I explicitly authorise another
location. Do not edit the provided `context/` files or `src/data_access.py`.

## Product objective and definition of done

Build a named FinTech investment product for a stated target user. The product
must let an investor compare systematically managed funds, read each fund's fact
sheet, allocate money across funds, and inspect equity-sector news sentiment.
The funds and their honest out-of-sample evidence are the product.

The minimum Part B implementation is:

- one combined equity-plus-crypto family with at least two portfolio methods;
- a walk-forward out-of-sample backtest with no look-ahead;
- one fact sheet per `(asset family, method)` fund;
- a standalone sentiment index across the ten equity sectors;
- a look-ahead-safe attempt to fuse sentiment into an equity fund;
- a Streamlit investor journey that reads precomputed artifacts; and
- a Word-authored report, reproducible code, required outputs, AI workflow pack,
  public GitHub repository, and live Streamlit URL.

For a higher band, prefer one well-motivated, evidenced innovation over several
shallow additions. Suitable directions include equity-only and crypto-only
funds, extra portfolio methods, transaction costs or turnover, a carefully
audited finance lexicon, robustness tests, a valuable app feature, or a coherent
original visual system. A negative result is acceptable when the method is sound
and the result is interpreted honestly.

## Working method and decision discipline

- Inspect existing work and `git status` before editing. Preserve unrelated or
  uncommitted work.
- For a material change, briefly state the objective, affected files or outputs,
  key risk, and verification method before implementation.
- Distinguish the product objective from a proposed implementation. If several
  methods are reasonable, compare their complexity, maintainability, financial
  validity, and assessment value before recommending one.
- Use the prompt structure taught in FINS5545: goal, context, constraints, and
  done-when criteria. Every code change needs an expected output, test, or audit.
- Make a reasonable stated assumption for a small, low-risk ambiguity. Stop and
  ask when a choice would materially change the investment design, reported
  conclusion, external state, or submission scope.
- Do not introduce a workaround only to silence an error. Explain the cause,
  implement a maintainable fix, and mark any unavoidable temporary compromise.
- Do not claim completion from code inspection alone. Run the relevant workflow
  and report the command, exit status, actual output, and remaining limitations.

## Numbered-step and checkpoint scope

- For a numbered implementation step or validation checkpoint, define completion
  only against that step's explicitly stated deliverables. A component `PASS`
  means that step is validated; it does not mean the whole project is complete or
  submission-ready.
- Validate only the files, artifacts, formulas, behaviours, and direct regressions
  created or affected by the current step. Do not expand a focused checkpoint into
  a project-readiness audit.
- Do not inspect or report unfinished later-step components as warnings, failures,
  limitations, blockers, recommendations, or evidence that the current step is
  incomplete. State only that later components were out of scope and not assessed.
- A pre-existing issue outside the current step may be reported only when it
  directly prevents the scoped workflow from running or invalidates an output
  created by that step. Explain that direct connection explicitly.
- Run the focused commands named for the current step. Do not append the full test
  suite, unified build, Streamlit, `check_handin.py`, deployment, or Git/GitHub
  readiness checks unless the user explicitly includes them in that step.
- During a component step, use `git status` only to preserve unrelated work. Do not
  audit, initialise, change, or report repository setup unless Git is the stated
  task or the existing state directly blocks the scoped work.
- Do not use a "remaining limitations" section to inventory unfinished later
  steps. Report only limitations inherent in the method or deliverables implemented
  in the current step.
- Apply the complete project-readiness gate under "Verification and acceptance
  gates" only when the user explicitly asks whether the whole project is ready, or
  when the numbered plan reaches its designated final readiness step.

## Folder, code, and dependency conventions

- Reusable analysis belongs in `src/`; orchestration in `scripts/`; tests in
  `tests/`; app-readable artifacts in `results/data/`; report tables in
  `results/tables/`; figures in `results/figures/`; writing in `report/`; and AI
  evidence in `ai/`.
- Keep functions small, typed where useful, deterministic, and documented when
  the financial calculation, timing rule, or date alignment is not obvious.
- Use relative paths resolved from the project root. Never hard-code a laptop or
  user-specific absolute path into code.
- Use Python 3.13 and the fins-agent repository interpreter, not system Python.
  From this folder on macOS/Linux it is normally `../../.venv/bin/python`; first
  inspect the environment if that path is unavailable.
- Run tools through the interpreter, for example
  `../../.venv/bin/python -m pytest -q` and
  `../../.venv/bin/python scripts/run_part_b.py`.
- If a package is needed, update `requirements.txt` for deployed app dependencies
  or `requirements-dev.txt` for build/test-only dependencies before installing it
  with `python -m pip`. Keep `nltk` and sentiment scoring dependencies out of the
  deployed app whenever the app only reads precomputed scores.
- Do not create preview, scratch, cache, or temporary artifacts in the submission
  tree. Use a temporary directory outside the repository and never create
  symlinks back into the project.

## Data foundation rules

- Load the hosted datasets only through `src/data_access.py`. Do not scrape, add
  API keys, repeatedly download the bundle, or commit raw source data.
- Use `adjClose` for returns. Cap crypto at `2023-12-31`; the supplied file has
  ten stray rows dated `2024-01-01`.
- Price rows must be unique on `ticker + date`. Multiple news rows on a
  ticker-date are normal; remove only exact headline duplicates on
  `ticker + date + title`.
- Calculate equity and crypto simple returns within ticker on their separate
  calendars. Then left-join crypto returns to the equity calendar for the
  combined fund. Do not calculate returns across a merge.
- Normalise the UTC-aware headline dates and timezone-naive price dates before
  mapping or merging. Map every headline to the same equity trading day if it is
  a trading day, otherwise to the next equity trading day.
- Preserve `title` unchanged as `text_raw`. Count-specific cleaning must use a
  separate column. Headlines are not full articles and remain a noisy proxy.
- Reuse Part A transformations only after rerunning their tests and reconciling
  row counts, date spans, schemas, and output definitions against the current
  data helper. Never copy a derived CSV without recording how it was produced.
- Record data provenance, transformation, frequency, coverage, and units for
  every app or report artifact.

## Portfolio and backtest rules

- Treat every `(asset family, method)` pair as a separate investable fund. Use an
  equal-weight 1/N benchmark and at least two methods for the combined family.
  Long-only, fully invested weights are the default unless leverage or shorting
  is explicitly motivated, constrained, and risk-tested.
- The backtest must walk forward. At each rebalance, the estimation window ends
  strictly before the first holding return; weights are estimated from past data
  only and applied to the following holding period.
- Rebalance monthly or less often. State the initial estimation length, expanding
  or rolling window, first live date, rebalance convention, constraints,
  risk-free rate, transaction-cost assumption, and missing-return policy.
- Use the actual evaluation calendar for annualisation: normally 252 for equity
  and equity-calendar combined funds, and 365 for a pure crypto daily fund.
  State the convention beside every metric.
- Validate every optimiser result: solver success, finite weights, sum of weights
  near one, constraint compliance, objective value, and non-trivial variation
  across methods and rebalances. Tiny daily covariances can cause silent solver
  stalls; scale or reformulate the problem rather than accepting identical or
  unchanged weights.
- Keep the 1/N result as an honest benchmark. FINS5545 lecture findings that 1/N
  was difficult to beat and tangency weights were unstable are hypotheses to
  test on this implementation, not results to copy into the report.
- Derive daily fund returns from the logged target weights and holding-period
  asset returns. Retain a rebalance audit with decision date, training endpoints,
  first holding date, window size, and solver status.
- Each fact sheet must show growth of $1, annualised return, annualised
  volatility, Sharpe ratio, maximum drawdown, and current target holdings from
  the most recent rebalance. Define arithmetic versus geometric annual return,
  risk-free rate, and drawdown formula consistently.
- If adding turnover, transaction costs, volatility targeting, or tuned
  parameters, compute all inputs from information available at that time and
  compare against an otherwise identical baseline.

## Sentiment index rules

- A sentiment score is `model(text)`. Record the exact model, package/version,
  lexicon or extension, text input, score definition, and transformation.
- Do not lowercase, strip punctuation, or remove stop words before VADER-style
  scoring. Capitalisation, punctuation, negation, and intensifiers are model
  inputs. Any count-specific text preparation stays separate from `text_raw`.
- A score of zero may mean the lexicon did not cover the finance term; do not
  automatically interpret zero as evidence of neutral information.
- Score each distinct title once for efficiency, cache build-time scores, and
  join them back to the clean ticker rows. Validate that the join neither loses
  nor multiplies clean rows unexpectedly.
- Aggregate in the required order: headline scores to ticker-day scores, then
  equal-weight ticker-day scores within each sector. Explicitly choose and
  justify how ticker-days with no headlines are treated: missing, carried
  forward, or neutral.
- Align news to the equity trading calendar, then lag the usable sentiment signal
  by at least one trading day. A Saturday or Monday headline mapped to Monday is
  first usable for Tuesday's decision.
- A full-sample standardisation may be used only for a clearly labelled
  retrospective descriptive chart. Any tradable or live signal must use rolling
  or expanding statistics based only on information available at that date.
- Validate sentiment rather than trusting one aggregate number: report coverage,
  neutral share, score distribution, sector/news-day coverage, model comparison
  or labelled/manual checks, and known error directions where feasible.
- If extending a finance lexicon, store candidate, rating evidence, student
  decision, and reason in an auditable file. AI rater agreement is not approval
  or accuracy evidence. Apply approved terms in memory on each build; do not edit
  installed packages. Test phrase rules explicitly because VADER special cases
  can fail silently.
- Do not claim sentiment predicts returns unless a look-ahead-safe test supports
  that claim. Greater vocabulary coverage is not the same as greater accuracy.

## Fusion rules

- Sentiment can affect equity weights only; crypto has no supplied news signal.
- Compare the base and sentiment-augmented fund over exactly the same out-of-
  sample dates, rebalance schedule, constraints, cost assumptions, and metric
  definitions.
- Lag the signal before forming weights. Any tilt strength, threshold, smoothing
  window, or hyperparameter selected from results must be chosen within the
  historical training process or labelled exploratory; never tune on the final
  test period and present it as out-of-sample.
- Report the before-versus-after effect on return, volatility, Sharpe, drawdown,
  turnover, and exposure where available. Explain a negative result honestly.

## Results, report, and app

`scripts/run_part_b.py` must reproduce all core artifacts from a clean checkout.
Create and retain these exact required files:

```text
results/data/fund_returns.csv
results/data/fund_weights.csv
results/data/sector_sentiment_index.csv
results/tables/performance_metrics.csv
```

The required Part B evidence also includes:

- a performance table across funds and methods;
- a growth-of-$1 comparison;
- a drawdown figure for at least one fund;
- a weights-over-time figure across methods for at least one fund;
- a Sharpe or return-versus-risk comparison;
- the sector sentiment time series; and
- a fusion before-versus-after table and figure.

Every exhibit must be self-contained: title/caption, labelled axes, units,
sample period, source or provenance, and a report interpretation. Do not copy
course figures or their numerical conclusions; use this project's generated
evidence and a coherent design system of my own.

The deployed `streamlit_app.py` must load precomputed `results/` artifacts and
remain light enough for Streamlit Community Cloud. It must not run VADER,
recompute a backtest, import `nltk`, or depend on raw local files. Use caching for
artifact reads, relative paths, clear missing-artifact errors, and no secrets.
The user journey must support fund comparison, individual fact sheets, allocation
across funds, and sentiment analytics.

Author the editable report in `report/report.docx` and submit
`report/report.pdf`. Keep written narrative within ten pages (about 5,000 words),
excluding appendix and references. Include the funds/backtest design, results and
fact sheets, sentiment index, fusion/innovation, app journey, critical reflection,
and three concrete real-world recommendations.

The final economic interpretation and conclusions must be my own words. The
agent may help outline, question, edit, and check my draft, but must not present
unreviewed AI prose as my reasoning.

## Verification and acceptance gates

Add focused tests before calling a substantive component complete. At minimum,
test:

- returns use adjusted prices within ticker and calendar alignment occurs after
  return calculation;
- training data ends before every holding period and no signal uses future data;
- weights are finite, satisfy constraints, sum to one, and vary where expected;
- logged weights reproduce sampled fund returns by hand;
- growth, annualisation, Sharpe, and drawdown match independent calculations;
- headline mapping moves non-trading-day news forward and the trading signal is
  lagged one trading day;
- the sector index equal-weights ticker-day scores rather than headlines; and
- the app artifact schemas, fund names, dates, and joins are internally
  consistent.

Before reporting the project ready, run in this order:

```bash
../../.venv/bin/python -m pytest -q
../../.venv/bin/python scripts/run_part_b.py
../../.venv/bin/python -m streamlit run streamlit_app.py
../../.venv/bin/python scripts/check_handin.py
git status --short
```

For the Streamlit command, verify the main pages and interactions locally, then
stop the server cleanly. Fix every `[FAIL]`; investigate every `[WARN]` instead
of silently ignoring it. A success claim without raw command output or an exit
status is unverified.

## Evidence, citations, and AI workflow

- Never invent a citation, statistic, method, data source, result, or test
  outcome. Trace every reported number to a formula, input, and reproducible code
  path.
- Verify every citation against a source I have opened. If it cannot be verified,
  label it unverified and omit it from final writing.
- Clearly separate observed results, modelling assumptions, limitations,
  exploratory findings, and recommendations.
- For every substantive AI-assisted task, create a separate record in `ai/`
  using `ai/prompt_log_template.md`: goal, prompts, AI output, errors or risks,
  and my correction with the reason. Do not fabricate a correction or personal
  decision on my behalf; leave it for my confirmation when necessary.
- Maintain `ai/AI_NOTES.md` in my own words with a candid account of where AI
  helped, where it was wrong, what I rejected, and how I verified the result.
  Treat every AI output as a draft to check, never as a fact to trust.

## Version control, safety, and submission hygiene

- Do not overwrite, delete, move, revert, or discard existing work without my
  explicit approval. Recommend a checkpoint before a high-risk change.
- Do not run `git commit`, `git push`, `git reset`, `git checkout --`, or any
  history-rewriting command without my explicit request.
- Never commit raw CSV/Parquet data outside `results/`, caches, `.DS_Store`,
  credentials, `.env`, or `.streamlit/secrets.toml`.
- The local `FINS5545/` reference folder contains raw teaching data and is not
  part of the Part B submission. Before hand-in, flag that it and OS-junk files
  must be excluded or moved outside the project; do not delete or move them
  without my approval.
- Commit the reproducible precomputed artifacts under `results/` because the
  deployed app reads them. Do not make generated files the only copy of logic.
- Keep this Part B folder as its own GitHub repository, separate from the parent
  fins-agent repository. Keep it private while developing, make it public at
  hand-in, verify the live app while logged out, and submit the ZIP, public repo
  URL, and live Streamlit URL.
