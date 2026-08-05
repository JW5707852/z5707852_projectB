# Prompt log - Core VADER sector sentiment index

## What I wanted

Build the locked ten-sector sentiment index from clean equity headlines using
the standard NLTK VADER lexicon. Titles had to remain unchanged, each distinct
title had to be scored once, aggregation had to equal-weight observed
ticker-days rather than headlines, and the tradable signal had to use only
prior information with a one-equity-trading-day lag.

## Prompt(s)

The student requested build-only NLTK VADER scoring, exact model provenance,
safe score joins, ticker-day then sector-day aggregation, missing no-headline
ticker-days, ticker coverage fields, a descriptive raw index, a lagged tradable
signal, the exact `sector_sentiment_index.csv` artifact, focused tests, coverage
and distribution evidence, limitations, and a separate AI record.

## What the assistant produced

### Model and input provenance

- Package: NLTK `3.10.1`, installed from the existing
  `requirements-dev.txt` declaration only.
- Analyzer: `nltk.sentiment.SentimentIntensityAnalyzer`.
- Lexicon resource:
  `sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt`.
- Official lexicon ZIP SHA-256:
  `8adba4294eef3964d820bf655e37e61bdc3a341994356af59b74fb3b4a36ce5c`.
- Lexicon extension: none.
- Score: VADER `compound`, the normalized weighted valence sum
  `x / sqrt(x^2 + 15)`, bounded to `[-1, 1]`.
- Input: unchanged `text_raw`, which remains equal to `title`.
- Preprocessing: none; no lowercasing, punctuation removal, stop-word removal,
  negation removal, or intensifier removal.

The package and lexicon are build dependencies under the repository virtual
environment. `requirements.txt` was not changed, and the deployed app does not
import NLTK or score text.

### Implementation

- Replaced the starter `src/sentiment.py` with reusable functions that:
  - locate the prepared lexicon without downloading during scoring;
  - validate unchanged string inputs;
  - create one in-memory score-cache row per distinct title;
  - join cached scores back with a validated many-to-one merge and stable row
    identity;
  - aggregate headlines to unique ticker-day compound scores;
  - equal-weight observed ticker-day scores within sector-days;
  - retain no-news sector-days on the full equity calendar with a missing raw
    score and zero coverage;
  - report observed/possible ticker counts, ticker coverage, and headline count;
  - compute an expanding z-score from at least 60 earlier non-missing raw sector
    observations using sample standard deviation (`ddof=1`);
  - clip the raw-day z-score to `[-2, 2]`; and
  - shift that clipped value forward by one equity trading day, retaining its
    source date and history count.
- Extended `scripts/run_part_b.py` with separate sentiment build and write
  functions. The script writes the precomputed index for the app while keeping
  title scoring outside the deployed application.
- Added `tests/test_sentiment.py` covering distinct-title calls, score-join
  cardinality, unchanged VADER inputs, weekend-forward mapping, ticker-day
  averaging, sector ticker equal-weighting, coverage, prior-history z-scoring,
  one-day lagging, and rejection of a zero-day lag.
- Generated `results/data/sector_sentiment_index.csv`.

### Hosted-data evidence

- Clean mapped headlines: 146,836.
- Scored headline rows: 146,836; the score join lost and multiplied zero rows.
- Distinct titles scored once: 105,334.
- Observed ticker-days: 37,962.
- Sector artifact: 10,060 rows, comprising ten sectors across 1,006 equity
  trading dates.
- Mapping: 134,279 same-day headlines, 12,551 mapped to the next trading day,
  and six outside the final equity sample.
- Sector-news days: 9,832 of 10,060 (`97.73%`). A sector-day with no observed
  ticker remains missing for the raw score.
- Tradable signal values: 9,231. Every recorded source date is earlier than its
  availability date, all use at least 60 prior observations, and values remain
  in `[-2, 2]`.

Headline compound distribution:

| Statistic | Value |
|---|---:|
| Mean | 0.105614 |
| Standard deviation | 0.288148 |
| Minimum | -0.918600 |
| 5th percentile | -0.401900 |
| 25th percentile | 0.000000 |
| Median | 0.000000 |
| 75th percentile | 0.296000 |
| 95th percentile | 0.636900 |
| Maximum | 0.955200 |

- Zero-score headline rows: 71,724 of 146,836 (`48.85%`).
- Zero-score distinct titles: 50,656 of 105,334 (`48.09%`).
- Across sector-news days, mean observed-ticker coverage is `77.22%`, median is
  `80%`, and the range is `20%` to `100%`.
- Every sector contains five possible tickers. News-day coverage is highest for
  Tech (`100%` of equity dates) and lowest for Materials and RealEstate
  (`93.44%` of dates). Mean within-news-day ticker coverage is lowest for
  Materials (`56.85%`).

### Verification commands and outcomes

1. `../../.venv/bin/python -m pip install -r requirements-dev.txt`
   - Exit 0; installed NLTK `3.10.1` and its declared dependencies.
2. `../../.venv/bin/python -m nltk.downloader -d ../../.venv/nltk_data vader_lexicon`
   - Exit 1 because Python could not validate the downloader host's SSL chain;
     it then attempted an interactive retry in a non-interactive process.
   - The unchanged official NLTK data package was instead downloaded from the
     NLTK `nltk_data` repository with `curl`, copied into
     `.venv/nltk_data/sentiment/`, checksum-recorded, and successfully loaded by
     `SentimentIntensityAnalyzer`.
3. `../../.venv/bin/python -m ruff check --no-cache src/sentiment.py scripts/run_part_b.py tests/test_sentiment.py`
   - Exit 0; `All checks passed!`.
4. `../../.venv/bin/python -m pytest -q tests/test_sentiment.py`
   - First run: exit 1; four tests passed and three failed. One production defect
     dropped the sector grouping column. Two test-fixture expectations were
     corrected: required news URL/publisher fields and the hand-calculated
     clipped z-score.
   - Corrected run: exit 0; `7 passed, 1 warning in 0.93s`.
   - The warning was the known parent-repository `PytestCacheWarning`, not a
     sentiment failure.
   - Final rerun after making the recorded compound formula explicit: exit 0;
     `7 passed, 1 warning in 0.94s`.
5. `../../.venv/bin/python scripts/run_part_b.py`
   - First run: exit 1 before writing because orchestration called a non-existent
     `load_clean_headlines` name instead of `load_clean_news`.
   - Corrected run: exit 0; generated all current artifacts, including the
     10,060-row sentiment CSV.
6. Read-only hosted sentiment coverage/distribution/lag audit:
   - Exit 0; produced the evidence above and found zero lag violations.
7. `../../.venv/bin/python -m pytest -q`
   - Initial integrated run: exit 0; `48 passed in 8.75s`.
   - Final run after artifact regeneration: exit 0; `48 passed in 7.98s`.

## What was wrong or risky

- Scoring every headline independently would repeat model work for duplicate
  titles. The in-memory title cache reduces 146,836 scoring calls to 105,334.
- A raw headline-weighted sector mean would allow a heavily covered ticker to
  dominate. The implemented order is headline to ticker-day first, followed by
  an equal-weight mean across observed ticker-days.
- Filling missing ticker-days with zero would confuse absent news with a VADER
  zero and depress the index mechanically. Missing names remain absent from the
  mean, while counts and coverage make this selection explicit.
- Using the current raw score in a same-day decision would be look-ahead unsafe.
  The expanding statistics exclude the current observation, and the clipped
  raw-day z-score is shifted forward one equity trading day.
- The initial NLTK downloader route failed because of local SSL certificate
  verification. The fallback package came from NLTK's official data repository,
  was left unchanged, and was checksum-recorded rather than disabling SSL
  verification inside Python.

## What I changed and why

### Assistant implementation record

The assistant implemented the locked plain-VADER design, generated the raw and
tradable sector fields, and added validation at every aggregation and timing
boundary. It did not extend the lexicon, tune sentiment thresholds, implement
sentiment fusion, or add optional model innovation.

### Student correction and confirmation — to be completed by the student

- [ ] I reviewed the unchanged-title input and plain-VADER model choice.
- [ ] I checked at least one headline-to-ticker-day and ticker-to-sector example.
- [ ] I understand that a zero VADER score may be a false neutral.
- [ ] I reviewed the missing-news and observed-ticker coverage policy.
- [ ] I reviewed the expanding-history and one-trading-day lag convention.
- [ ] Corrections I requested, if any: `<student to complete>`
- [ ] Limitations or concerns I want recorded in my own words:
      `<student to complete>`

The assistant has not filled these student-owned confirmation fields.

## Remaining limitations

- Headlines are short, context-poor proxies rather than full articles.
- Plain VADER is general-purpose and assigns exactly zero to 48.85% of headline
  rows; zero can reflect vocabulary non-coverage rather than neutral information.
- Sector scores condition on observed ticker-days. Coverage varies across dates
  and sectors, so the index composition is not constant even though observed
  tickers are equal-weighted.
- No labelled finance-news validation sample or alternative model comparison is
  included in this core step.
- The lexicon is an external build resource and must be prepared once in a clean
  development environment; it is intentionally absent from app dependencies.
- The sentiment fusion fund and its before-versus-after performance evaluation
  remain future work.
