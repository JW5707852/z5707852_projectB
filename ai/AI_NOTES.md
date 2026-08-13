# AI use and verification notes

## How I used AI

I used Codex throughout Part B as a coding, testing, audit, and interface-design
assistant. I did not use one general prompt to generate the whole project. I
worked in smaller checkpoints: data preparation, walk-forward funds, performance
artifacts, VADER sentiment, sentiment fusion, reproducible outputs, evidence
figures, FinBERT robustness, and the Streamlit investor journey. I kept a
separate prompt log for each substantive task so that the request, output,
failure, correction, and verification could be traced.

I directed the work through `AGENTS.md`. The most important constraints were to
use adjusted prices, calculate equity and crypto returns on their native
calendars before alignment, prevent look-ahead, aggregate sentiment from
headline to ticker-day before equal-weighting tickers within sectors, and keep
the deployed app dependent only on precomputed artifacts. I also required
focused tests and actual command output before treating a component as complete.

AI was most useful for turning these constraints into reusable functions,
validation gates, regression tests, and repeatable scripts. It helped me inspect
schemas, trace dates through the rebalance process, reproduce fund returns from
logged weights, generate figures and tables, and find edge cases that would be
easy to overlook manually. It also helped me iterate quickly on the Streamlit
interface after I explained what was unclear from an investor's perspective.

## Where AI helped most

The strongest contribution was implementation discipline. For the backtest, AI
created a monthly expanding-window process with at least 252 prior equity
trading days, month-end decisions, and holdings beginning on the next trading
day. It retained decision dates, training endpoints, first holding dates,
weights, solver status, and objective values. This made look-ahead checks and
manual return reproduction possible instead of leaving the backtest as a single
opaque return series.

AI also helped convert the sentiment requirements into an auditable sequence.
Titles remained unchanged for VADER; each distinct title was scored once; scores
were joined back without losing or multiplying rows; headline scores were
averaged to ticker-day; and observed ticker-days were equal-weighted within each
sector. No-news sector-days remained missing rather than being turned into
neutral observations. The tradable signal used expanding prior history and was
lagged by one equity trading day.

For the FinBERT extension, AI implemented a mandatory pilot before full-corpus
inference, pinned the model revision and label map, measured throughput and
truncation, and protected the existing fund and VADER artifacts with hashes. I
then completed the 150-title blind review myself. Only the 100 representative
rows were used for weighted accuracy; the 50 disagreement-enriched rows were
kept as qualitative diagnostics. This separation was important because a
disagreement sample cannot be treated as representative accuracy evidence.

The app work benefited from repeated visual inspection. AI added fund
comparison, fact sheets, fund allocation, individual-asset simulation, sector
sentiment, model comparison, professional metric labels, contextual help, and a
single methodology control. It also implemented optional-artifact isolation so
that a failure in the individual-asset or FinBERT extension would not make the
core fund pages unavailable.

## Where AI was wrong or incomplete

AI output was not reliable enough to accept without execution and review. There
were several concrete failures.

First, an early sentiment implementation dropped the sector grouping column,
and the orchestration called a non-existent loader name. Focused sentiment tests
and the real build failed, which exposed both problems. The code was corrected
and rerun rather than weakening the tests.

Second, the initial drawdown function calculated the running peak only from the
observed wealth path. That makes a first-period loss appear as zero drawdown
because the reduced value becomes the first peak. The audit caught this with a
short hand-calculated series. The corrected calculation includes the starting
wealth of $1 in the running peak. This did not change the published maximum
drawdowns because their worst falls occurred later, but the formula was still
wrong and needed repair.

Third, the first evidence manifest reused the fund-return sample period for all
figures. This incorrectly described the weights and sentiment figures, which
have different source date ranges. The manifest was changed to derive each
sample from its own source, and tests now reconcile each manifest row with the
corresponding caption and data.

Fourth, the first FinBERT full run completed all neural inference but stopped
before publication because a strict dataframe equality check treated equivalent
datetime resolutions as different. A diagnostic showed zero substantive
coverage differences. The correction normalised dates and retained exact or
tightly tolerant checks for the actual values. A regression test confirms that
equivalent datetime storage passes but a genuine headline-count difference
still fails.

The first frontend versions were technically functional but not suitable for a
customer-facing finance product. They repeated the product name, displayed raw
artifact paths and sample information on multiple pages, used internal labels
such as `Combined 1/N`, clipped long legends, and looked more like presentation
slides than software. I explicitly rejected that presentation. I asked for
common investor terminology, compact horizontal navigation, fewer decorative
cards, one information control, and help icons for specialist measures. I also
corrected the meaning of Portfolio Simulator: the first version allocated only
across prebuilt funds, while my requirement was to choose individual stocks and
cryptoassets row by row.

The heatmap is another example of review changing the result. Its initial colour
range used the single most extreme value, so normal variation looked almost
white. I requested stronger changes. The revised chart uses a display-only 95th
percentile symmetric scale and a steeper red-white-green palette. The raw scores
remain unchanged in hover text, and rare extremes may saturate visually.

## What I rejected or constrained

I rejected silent fallbacks that could make an invalid model look successful.
The minimum-variance fund raises on solver failure instead of being silently
replaced by equal weight. I also rejected calculating crypto returns after
aligning price levels, filling absent news with zero, using current sentiment in
the same day's decision, and changing locked parameters after observing final
performance.

I did not treat VADER-FinBERT agreement as accuracy. My blind review produced a
weighted accuracy point estimate of 46.82% for VADER and 53.07% for FinBERT, but
the exact paired McNemar p-value was 0.377. I therefore treat the comparison as
evidence of model risk and vocabulary differences, not proof that FinBERT is
superior and not evidence that either model predicts returns.

I also kept build-time dependencies out of the deployed app. VADER, NLTK,
Transformers, Torch, model downloads, raw hosted data, and portfolio
optimisation are not run inside Streamlit. The app reads versioned result files
and performs only display or transparent allocation arithmetic.

## How I verified AI output

I used four types of checks rather than relying on code inspection alone:

1. **Independent calculations.** I checked sample equity and crypto returns,
   reconstructed sampled fund returns from logged weights, recalculated growth,
   volatility, Sharpe and drawdown, and checked headline-to-ticker and
   ticker-to-sector aggregation examples.
2. **Automated tests.** Focused tests cover native calendars, look-ahead
   boundaries, solver constraints, weight sums, metric formulas, signal lags,
   artifact schemas, optional-artifact failures, allocation arithmetic, and
   app interactions. When a test failed, I distinguished a code defect from a
   network, cache-permission, or local-environment issue before acting.
3. **Reproducibility checks.** The core build was run repeatedly and the required
   CSVs were compared by dataframe equality and SHA-256 hashes. FinBERT phases
   also recorded protected-artifact hashes so the robustness extension could not
   silently change the funds or VADER index.
4. **Visual checks.** Exported figures and the live Streamlit pages were opened
   and inspected for clipped labels, overlapping controls, misleading legends,
   mobile overflow, unclear terminology, and error states. Several visual
   defects were corrected only after seeing the rendered result.

I did not accept numbers simply because AI reported them. Material figures in
the prompt logs are tied to commands, generated artifacts, formulas, or direct
reconciliations. External references used for methods or software behaviour were
checked against the source rather than invented from memory.

## My final responsibility

AI accelerated implementation and helped expose defects, but it did not decide
whether the product was clear to investors or what the economic results mean.
Those decisions required my corrections. The clearest example was the frontend:
the early output satisfied many technical requirements but still used language
and layouts that I would not expect a normal investor to understand.

The final fund results remain historical simulations, not live records or
forecasts. Minimum-variance weights are estimation-dependent, sentiment is a
noisy headline proxy, and the research scenarios exclude real taxes, fees,
slippage, and transaction costs beyond the stated assumptions. I am responsible
for the final interpretation, recommendations, citations, and any claims used in
the report. The prompt logs record where AI assisted; they do not transfer that
responsibility to the tool.

If I repeated the project, I would define the customer-facing terminology and
screen hierarchy earlier, and I would add the independent formula tests at the
same time as the first implementation rather than during a later audit. That
would reduce rework and make it harder for a technically plausible but poorly
communicated result to survive until the product stage.
