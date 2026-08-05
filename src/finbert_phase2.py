"""Auditable Phase 2 joins, aggregation, comparisons, and review sampling.

The functions in this module are model-independent once a validated FinBERT
title-score cache has been produced. They deliberately keep descriptive model
agreement separate from student-labelled accuracy evaluation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src import etl
from src.finbert_innovation import (
    EXPECTED_ID2LABEL,
    PROBABILITY_TOLERANCE,
    TEXT_INPUT_COLUMN,
    FinBERTValidationError,
    validate_probabilities,
)

SAMPLE_RANDOM_SEED = 5545
REPRESENTATIVE_SAMPLE_SIZE = 100
DIAGNOSTIC_SAMPLE_SIZE = 50
DISAGREEMENT_EXPORT_SIZE = 200
STUDENT_REVIEW_FIELDS = (
    "student_label",
    "student_confidence",
    "student_notes",
    "student_reviewed_by",
)
VADER_LABEL_RULE = "positive >= 0.05; negative <= -0.05; otherwise neutral"
FINBERT_LABEL_RULE = "argmax of positive, negative, and neutral probabilities"

_DIAGNOSTIC_RULES = (
    (
        "opposite_sign",
        re.compile(r"(?!)"),
        "VADER positive versus FinBERT negative, or VADER negative versus FinBERT positive",
    ),
    (
        "neutral_non_neutral",
        re.compile(r"(?!)"),
        "exactly one descriptive model label is neutral",
    ),
    (
        "negation",
        re.compile(
            r"\b(?:no|not|never|neither|nor|without|isn't|wasn't|won't|can't|cannot|hardly)\b", re.I
        ),
        "model disagreement and headline contains a negation cue",
    ),
    (
        "financial_term",
        re.compile(
            r"\b(?:upgrade|downgrade|bullish|bearish|dividend|buyback|default|"
            r"guidance|margin|liquidity|leverage|debt|cash flow|free cash flow)\b",
            re.I,
        ),
        "model disagreement and headline contains a finance-domain term",
    ),
    (
        "numerical",
        re.compile(r"(?:\d|%|\$|£|€)"),
        "model disagreement and headline contains a number, percentage, or currency symbol",
    ),
    (
        "earnings_announcement",
        re.compile(
            r"\b(?:earnings|quarterly results?|annual results?|profit|revenue|"
            r"sales|eps|forecast|outlook)\b",
            re.I,
        ),
        "model disagreement and headline concerns earnings, results, or outlook",
    ),
)


def validate_title_score_cache(title_scores: pd.DataFrame) -> pd.DataFrame:
    """Validate one complete probability row per unchanged distinct title."""
    required = {
        TEXT_INPUT_COLUMN,
        "probability_positive",
        "probability_negative",
        "probability_neutral",
        "finbert_score",
        "finbert_label",
    }
    etl.require_columns(title_scores, required, "FinBERT title score cache")
    cache = title_scores.copy().reset_index(drop=True)
    if (
        cache[TEXT_INPUT_COLUMN].isna().any()
        or not cache[TEXT_INPUT_COLUMN].map(lambda value: isinstance(value, str)).all()
    ):
        raise FinBERTValidationError("FinBERT title cache contains invalid text_raw")
    if cache.duplicated(TEXT_INPUT_COLUMN).any():
        raise FinBERTValidationError("FinBERT title cache is not unique by text_raw")
    probability_columns = [
        "probability_positive",
        "probability_negative",
        "probability_neutral",
    ]
    probabilities = (
        cache[probability_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    )
    validate_probabilities(probabilities, tolerance=PROBABILITY_TOLERANCE)
    expected_score = probabilities[:, 0] - probabilities[:, 1]
    observed_score = pd.to_numeric(cache["finbert_score"], errors="coerce").to_numpy(dtype=float)
    if not np.allclose(observed_score, expected_score, atol=1e-12, rtol=0.0):
        raise FinBERTValidationError("FinBERT title cache violates the score definition")
    labels = cache["finbert_label"].astype(str).str.lower()
    if not labels.isin(set(EXPECTED_ID2LABEL.values())).all():
        raise FinBERTValidationError("FinBERT title cache contains an invalid label")
    expected_labels = np.asarray(list(EXPECTED_ID2LABEL.values()))[probabilities.argmax(axis=1)]
    if not np.array_equal(labels.to_numpy(), expected_labels):
        raise FinBERTValidationError("FinBERT labels do not equal probability argmax")
    cache[probability_columns] = probabilities
    cache["finbert_score"] = observed_score
    cache["finbert_label"] = labels
    return cache


def join_title_scores(
    mapped_headlines: pd.DataFrame,
    title_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Join scores many-to-one without losing, multiplying, or reordering rows."""
    etl.require_columns(
        mapped_headlines,
        {"title", TEXT_INPUT_COLUMN, "ticker", "sector", "trading_date"},
        "mapped headline panel",
    )
    if not mapped_headlines["title"].equals(mapped_headlines[TEXT_INPUT_COLUMN]):
        raise FinBERTValidationError("mapped headline text_raw differs from title")
    cache = validate_title_score_cache(title_scores)
    original = mapped_headlines.copy().reset_index(drop=True)
    original["_clean_row_id"] = np.arange(len(original))
    scored = original.merge(
        cache,
        on=TEXT_INPUT_COLUMN,
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )
    if len(scored) != len(original):
        raise FinBERTValidationError("FinBERT join multiplied clean headline rows")
    unmatched = int(scored["_merge"].ne("both").sum())
    if unmatched:
        raise FinBERTValidationError(f"FinBERT join left {unmatched} rows unmatched")
    if scored["_clean_row_id"].duplicated().any():
        raise FinBERTValidationError("FinBERT join duplicated a clean headline row")
    scored = scored.sort_values("_clean_row_id", kind="mergesort")
    if not scored[TEXT_INPUT_COLUMN].reset_index(drop=True).equals(original[TEXT_INPUT_COLUMN]):
        raise FinBERTValidationError("FinBERT join changed text_raw or row order")
    audit = {
        "clean_rows_before_join": len(original),
        "rows_after_join": len(scored),
        "matched_rows": int(scored["_merge"].eq("both").sum()),
        "unmatched_rows": unmatched,
        "multiplied_rows": len(scored) - len(original),
        "outside_equity_sample_rows": int(scored["trading_date"].isna().sum()),
    }
    return scored.drop(columns=["_clean_row_id", "_merge"]).reset_index(drop=True), audit


def aggregate_ticker_days(headline_scores: pd.DataFrame) -> pd.DataFrame:
    """Average FinBERT headline scores to one observed ticker-day."""
    required = {
        "trading_date",
        "ticker",
        "sector",
        TEXT_INPUT_COLUMN,
        "finbert_score",
        "probability_positive",
        "probability_negative",
        "probability_neutral",
    }
    etl.require_columns(headline_scores, required, "FinBERT headline scores")
    scored = headline_scores.dropna(subset=["trading_date"]).copy()
    scored["trading_date"] = etl.normalise_date(scored["trading_date"])
    numeric = [
        "finbert_score",
        "probability_positive",
        "probability_negative",
        "probability_neutral",
    ]
    scored[numeric] = scored[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(scored[numeric].to_numpy(dtype=float)).all():
        raise FinBERTValidationError("FinBERT headline scores contain non-finite values")
    inconsistent = scored.groupby("ticker")["sector"].nunique(dropna=False)
    if (inconsistent != 1).any():
        raise FinBERTValidationError("a ticker maps to multiple sectors")
    result = (
        scored.groupby(["trading_date", "ticker", "sector"], as_index=False)
        .agg(
            ticker_day_finbert=("finbert_score", "mean"),
            ticker_day_probability_positive=("probability_positive", "mean"),
            ticker_day_probability_negative=("probability_negative", "mean"),
            ticker_day_probability_neutral=("probability_neutral", "mean"),
            headline_count=("finbert_score", "size"),
            distinct_title_count=(TEXT_INPUT_COLUMN, "nunique"),
        )
        .sort_values(["trading_date", "ticker"])
        .reset_index(drop=True)
    )
    if result.duplicated(["trading_date", "ticker"]).any():
        raise FinBERTValidationError("FinBERT ticker-day aggregation is not unique")
    return result


def sector_sentiment_index(
    ticker_day_scores: pd.DataFrame,
    equity_calendar: Iterable[object],
    sector_universe: pd.DataFrame,
) -> pd.DataFrame:
    """Equal-weight observed ticker-days within sector; no-news days stay missing."""
    required = {
        "trading_date",
        "ticker",
        "sector",
        "ticker_day_finbert",
        "ticker_day_probability_positive",
        "ticker_day_probability_negative",
        "ticker_day_probability_neutral",
        "headline_count",
    }
    etl.require_columns(ticker_day_scores, required, "FinBERT ticker-day scores")
    etl.require_columns(sector_universe, {"ticker", "sector"}, "sector universe")
    calendar = (
        pd.DatetimeIndex(etl.normalise_date(pd.Series(list(equity_calendar), dtype="object")))
        .sort_values()
        .unique()
    )
    if len(calendar) == 0:
        raise ValueError("equity calendar must not be empty")
    universe = sector_universe[["ticker", "sector"]].drop_duplicates().copy()
    if universe.groupby("ticker")["sector"].nunique().ne(1).any():
        raise FinBERTValidationError("sector universe maps a ticker more than once")
    checked = ticker_day_scores.copy()
    checked["trading_date"] = etl.normalise_date(checked["trading_date"])
    if checked.duplicated(["trading_date", "ticker"]).any():
        raise FinBERTValidationError("FinBERT ticker-day input contains duplicate keys")
    checked = checked.merge(
        universe.rename(columns={"sector": "universe_sector"}),
        on="ticker",
        how="left",
        validate="many_to_one",
    )
    if (
        checked["universe_sector"].isna().any()
        or not checked["sector"].eq(checked["universe_sector"]).all()
    ):
        raise FinBERTValidationError("ticker-day sector disagrees with universe")
    observed = (
        checked.groupby(["trading_date", "sector"], as_index=False)
        .agg(
            raw_sector_finbert=("ticker_day_finbert", "mean"),
            sector_probability_positive=("ticker_day_probability_positive", "mean"),
            sector_probability_negative=("ticker_day_probability_negative", "mean"),
            sector_probability_neutral=("ticker_day_probability_neutral", "mean"),
            observed_ticker_count=("ticker", "nunique"),
            headline_count=("headline_count", "sum"),
        )
        .rename(columns={"trading_date": "date"})
    )
    sectors = sorted(universe["sector"].unique())
    grid = pd.MultiIndex.from_product([calendar, sectors], names=["date", "sector"]).to_frame(
        index=False
    )
    panel = grid.merge(observed, on=["date", "sector"], how="left", validate="one_to_one")
    possible = universe.groupby("sector")["ticker"].nunique()
    panel["possible_ticker_count"] = panel["sector"].map(possible).astype(int)
    panel["observed_ticker_count"] = panel["observed_ticker_count"].fillna(0).astype(int)
    panel["headline_count"] = panel["headline_count"].fillna(0).astype(int)
    panel["ticker_coverage_share"] = panel["observed_ticker_count"] / panel["possible_ticker_count"]
    panel["has_observed_news"] = panel["observed_ticker_count"].gt(0)
    if not panel.loc[~panel["has_observed_news"], "raw_sector_finbert"].isna().all():
        raise FinBERTValidationError("no-news sector-days must remain missing")
    panel["sentiment_model"] = "ProsusAI/finbert"
    panel["score_name"] = "P(positive) - P(negative)"
    panel["text_input_column"] = TEXT_INPUT_COLUMN
    panel["text_preprocessing"] = "none"
    panel["raw_index_purpose"] = "retrospective descriptive robustness index"
    return panel.sort_values(["date", "sector"]).reset_index(drop=True)


def vader_labels(values: pd.Series) -> pd.Series:
    """Apply standard VADER descriptive thresholds."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise FinBERTValidationError("VADER comparison scores contain missing values")
    return pd.Series(
        np.select(
            [numeric.ge(0.05), numeric.le(-0.05)],
            ["positive", "negative"],
            default="neutral",
        ),
        index=values.index,
        dtype="object",
    )


def add_comparison_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Add explicitly descriptive VADER-threshold and FinBERT-argmax flags."""
    etl.require_columns(
        frame,
        {"vader_compound", "finbert_score", "finbert_label"},
        "model comparison frame",
    )
    result = frame.copy()
    result["vader_label"] = vader_labels(result["vader_compound"])
    result["finbert_label"] = result["finbert_label"].astype(str).str.lower()
    result["label_agreement"] = result["vader_label"].eq(result["finbert_label"])
    result["opposite_sign"] = (
        result["vader_label"].eq("positive") & result["finbert_label"].eq("negative")
    ) | (result["vader_label"].eq("negative") & result["finbert_label"].eq("positive"))
    result["neutral_non_neutral"] = result["vader_label"].eq("neutral") ^ result[
        "finbert_label"
    ].eq("neutral")
    return result


def _comparison_record(
    frame: pd.DataFrame,
    *,
    observation_unit: str,
    sector: str,
    include_correlations: bool,
) -> dict[str, object]:
    paired = frame.dropna(subset=["vader_compound", "finbert_score"]).copy()
    paired_count = len(paired)
    if not include_correlations or paired_count < 2:
        pearson = np.nan
        spearman = np.nan
    else:
        pearson = paired["vader_compound"].corr(paired["finbert_score"], method="pearson")
        spearman = paired["vader_compound"].corr(paired["finbert_score"], method="spearman")
    return {
        "observation_unit": observation_unit,
        "sector": sector,
        "paired_observation_count": paired_count,
        "vader_mean": paired["vader_compound"].mean(),
        "finbert_mean": paired["finbert_score"].mean(),
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "descriptive_label_agreement_rate": paired["label_agreement"].mean(),
        "opposite_sign_rate": paired["opposite_sign"].mean(),
        "neutral_non_neutral_rate": paired["neutral_non_neutral"].mean(),
        "vader_positive_rate": paired["vader_label"].eq("positive").mean(),
        "vader_negative_rate": paired["vader_label"].eq("negative").mean(),
        "vader_neutral_rate": paired["vader_label"].eq("neutral").mean(),
        "finbert_positive_rate": paired["finbert_label"].eq("positive").mean(),
        "finbert_negative_rate": paired["finbert_label"].eq("negative").mean(),
        "finbert_neutral_rate": paired["finbert_label"].eq("neutral").mean(),
        "vader_label_rule": VADER_LABEL_RULE,
        "finbert_label_rule": FINBERT_LABEL_RULE,
        "correlation_basis": (
            "matched observed date-sector pairs"
            if include_correlations
            else "not reported; correlations restricted to matched date-sector observations"
        ),
        "interpretation_status": "descriptive comparison; not accuracy or predictive evidence",
    }


def model_comparison_table(
    headline_comparison: pd.DataFrame,
    vader_sector_index: pd.DataFrame,
    finbert_sector_index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare matched headline rows and matched date-sector observations."""
    headlines = add_comparison_flags(headline_comparison)
    records = [
        _comparison_record(
            headlines,
            observation_unit="clean_headline_row",
            sector="All",
            include_correlations=False,
        )
    ]
    for sector, group in headlines.groupby("sector", sort=True):
        records.append(
            _comparison_record(
                group,
                observation_unit="clean_headline_row",
                sector=str(sector),
                include_correlations=False,
            )
        )

    vader_required = {"date", "sector", "raw_sector_compound"}
    finbert_required = {
        "date",
        "sector",
        "raw_sector_finbert",
        "sector_probability_positive",
        "sector_probability_negative",
        "sector_probability_neutral",
    }
    etl.require_columns(vader_sector_index, vader_required, "VADER sector index")
    etl.require_columns(finbert_sector_index, finbert_required, "FinBERT sector index")
    vader = vader_sector_index[list(vader_required)].copy()
    finbert = finbert_sector_index[list(finbert_required)].copy()
    vader["date"] = etl.normalise_date(vader["date"])
    finbert["date"] = etl.normalise_date(finbert["date"])
    matched = vader.merge(
        finbert,
        on=["date", "sector"],
        how="inner",
        validate="one_to_one",
    ).dropna(subset=["raw_sector_compound", "raw_sector_finbert"])
    matched = matched.rename(
        columns={
            "raw_sector_compound": "vader_compound",
            "raw_sector_finbert": "finbert_score",
        }
    )
    sector_probabilities = matched[
        [
            "sector_probability_positive",
            "sector_probability_negative",
            "sector_probability_neutral",
        ]
    ].to_numpy(dtype=float)
    matched["finbert_label"] = np.asarray(list(EXPECTED_ID2LABEL.values()))[
        sector_probabilities.argmax(axis=1)
    ]
    matched = add_comparison_flags(matched)
    records.append(
        _comparison_record(
            matched,
            observation_unit="matched_date_sector",
            sector="All",
            include_correlations=True,
        )
    )
    for sector, group in matched.groupby("sector", sort=True):
        records.append(
            _comparison_record(
                group,
                observation_unit="matched_date_sector",
                sector=str(sector),
                include_correlations=True,
            )
        )
    return pd.DataFrame.from_records(records), matched.sort_values(["date", "sector"])


def validate_coverage_reconciliation(
    vader_sector_index: pd.DataFrame,
    finbert_sector_index: pd.DataFrame,
    *,
    float_tolerance: float = 1e-15,
) -> dict[str, int]:
    """Require semantic coverage equality while ignoring datetime resolution.

    CSV parsing may produce microsecond datetimes while an in-memory calendar
    carries second or nanosecond resolution. That storage dtype is not a data
    difference, so dates are normalised before exact key/count comparisons.
    """
    columns = [
        "date",
        "sector",
        "observed_ticker_count",
        "headline_count",
        "possible_ticker_count",
        "ticker_coverage_share",
        "has_observed_news",
    ]
    etl.require_columns(vader_sector_index, set(columns), "VADER sector index")
    etl.require_columns(finbert_sector_index, set(columns), "FinBERT sector index")
    vader = vader_sector_index[columns].copy().reset_index(drop=True)
    finbert = finbert_sector_index[columns].copy().reset_index(drop=True)
    if len(vader) != len(finbert):
        raise FinBERTValidationError("VADER and FinBERT sector grids differ in length")
    vader["date"] = etl.normalise_date(vader["date"])
    finbert["date"] = etl.normalise_date(finbert["date"])
    exact_columns = [
        "date",
        "sector",
        "observed_ticker_count",
        "headline_count",
        "possible_ticker_count",
        "has_observed_news",
    ]
    mismatches = {column: int(vader[column].ne(finbert[column]).sum()) for column in exact_columns}
    coverage_difference = (
        pd.to_numeric(vader["ticker_coverage_share"], errors="coerce")
        - pd.to_numeric(finbert["ticker_coverage_share"], errors="coerce")
    ).abs()
    mismatches["ticker_coverage_share"] = int(
        coverage_difference.gt(float_tolerance).sum() + coverage_difference.isna().sum()
    )
    if any(mismatches.values()):
        raise FinBERTValidationError(
            f"FinBERT coverage or missingness differs from VADER: {mismatches}"
        )
    return {
        "sector_grid_rows_compared": len(vader),
        "coverage_mismatch_count": 0,
    }


def _largest_remainder_allocation(counts: pd.Series, sample_size: int) -> pd.Series:
    if sample_size > int(counts.sum()):
        raise ValueError("sample size exceeds population")
    exact = sample_size * counts / counts.sum()
    allocation = np.floor(exact).astype(int)
    eligible = counts.gt(0) & allocation.eq(0)
    allocation.loc[eligible] = 1
    while allocation.sum() > sample_size:
        removable = allocation[allocation.gt(1)]
        if removable.empty:
            raise FinBERTValidationError("cannot allocate representative strata")
        key = sorted(
            removable.index, key=lambda item: (exact.loc[item] - allocation.loc[item], str(item))
        )[0]
        allocation.loc[key] -= 1
    while allocation.sum() < sample_size:
        capacity = counts - allocation
        candidates = capacity[capacity.gt(0)]
        if candidates.empty:
            raise FinBERTValidationError("representative allocation lacks capacity")
        key = sorted(
            candidates.index,
            key=lambda item: (-(exact.loc[item] - allocation.loc[item]), str(item)),
        )[0]
        allocation.loc[key] += 1
    return allocation


def _title_level_population(headline_comparison: pd.DataFrame) -> pd.DataFrame:
    required = {
        TEXT_INPUT_COLUMN,
        "date",
        "ticker",
        "sector",
        "vader_compound",
        "finbert_score",
        "finbert_label",
        "probability_positive",
        "probability_negative",
        "probability_neutral",
    }
    etl.require_columns(headline_comparison, required, "headline comparison")
    population = headline_comparison.copy()
    population["date"] = etl.normalise_date(population["date"])
    population = population.sort_values(
        [TEXT_INPUT_COLUMN, "date", "ticker", "sector"], kind="mergesort"
    ).drop_duplicates(TEXT_INPUT_COLUMN, keep="first")
    population = add_comparison_flags(population)
    population["year"] = population["date"].dt.year.astype(int)
    return population.reset_index(drop=True)


def representative_sample(
    headline_comparison: pd.DataFrame,
    *,
    sample_size: int = REPRESENTATIVE_SAMPLE_SIZE,
    seed: int = SAMPLE_RANDOM_SEED,
) -> pd.DataFrame:
    """Deterministic sector-year stratified random evaluation sample."""
    population = _title_level_population(headline_comparison)
    population["sampling_stratum"] = (
        population["sector"].astype(str) + "|" + population["year"].astype(str)
    )
    counts = population.groupby("sampling_stratum", sort=True).size()
    allocation = _largest_remainder_allocation(counts, sample_size)
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    for stratum, group in population.groupby("sampling_stratum", sort=True):
        take = int(allocation.loc[stratum])
        positions = np.sort(rng.choice(len(group), size=take, replace=False))
        sample = group.iloc[positions].copy()
        sample["population_stratum_count"] = len(group)
        sample["stratum_sample_count"] = take
        sample["sampling_probability"] = take / len(group)
        sample["sampling_weight"] = len(group) / take
        selected.append(sample)
    result = pd.concat(selected, ignore_index=True)
    result["sample_purpose"] = "representative_evaluation"
    result["selection_rule"] = (
        f"seed={seed}; distinct-title sector-year stratified random sample; "
        "proportional largest-remainder allocation with at least one per nonempty stratum"
    )
    result["validation_status"] = "pending student review"
    for field in STUDENT_REVIEW_FIELDS:
        result[field] = ""
    return result.sort_values(["sampling_stratum", TEXT_INPUT_COLUMN]).reset_index(drop=True)


def _diagnostic_memberships(population: pd.DataFrame) -> dict[str, pd.Series]:
    text = population[TEXT_INPUT_COLUMN].astype(str)
    memberships: dict[str, pd.Series] = {}
    for name, pattern, _ in _DIAGNOSTIC_RULES:
        if name == "opposite_sign":
            memberships[name] = population["opposite_sign"]
        elif name == "neutral_non_neutral":
            memberships[name] = population["neutral_non_neutral"]
        else:
            memberships[name] = population["label_agreement"].eq(False) & text.str.contains(pattern)
    return memberships


def diagnostic_sample(
    headline_comparison: pd.DataFrame,
    *,
    excluded_titles: set[str] | None = None,
    sample_size: int = DIAGNOSTIC_SAMPLE_SIZE,
    seed: int = SAMPLE_RANDOM_SEED,
) -> pd.DataFrame:
    """Select disagreement-enriched cases for qualitative diagnosis only."""
    population = _title_level_population(headline_comparison)
    population = population.loc[population["label_agreement"].eq(False)].copy()
    if excluded_titles:
        population = population.loc[~population[TEXT_INPUT_COLUMN].isin(excluded_titles)]
    memberships = _diagnostic_memberships(population)
    targets = {
        "opposite_sign": 10,
        "neutral_non_neutral": 8,
        "negation": 8,
        "financial_term": 8,
        "numerical": 8,
        "earnings_announcement": 8,
    }
    rng = np.random.default_rng(seed + 1)
    chosen_titles: set[str] = set()
    pieces: list[pd.DataFrame] = []
    for name, _, rule in _DIAGNOSTIC_RULES:
        candidates = population.loc[
            memberships[name] & ~population[TEXT_INPUT_COLUMN].isin(chosen_titles)
        ]
        take = min(targets[name], len(candidates), sample_size - len(chosen_titles))
        if take:
            positions = np.sort(rng.choice(len(candidates), size=take, replace=False))
            selected = candidates.iloc[positions].copy()
            selected["sampling_stratum"] = name
            selected["selection_rule"] = (
                f"seed={seed + 1}; {rule}; disagreement-enriched, unweighted"
            )
            pieces.append(selected)
            chosen_titles.update(selected[TEXT_INPUT_COLUMN])
    if len(chosen_titles) < sample_size:
        remaining = population.loc[~population[TEXT_INPUT_COLUMN].isin(chosen_titles)]
        take = sample_size - len(chosen_titles)
        if len(remaining) < take:
            raise FinBERTValidationError("not enough model disagreements for diagnostic sample")
        positions = np.sort(rng.choice(len(remaining), size=take, replace=False))
        selected = remaining.iloc[positions].copy()
        selected["sampling_stratum"] = "other_disagreement"
        selected["selection_rule"] = f"seed={seed + 1}; remaining label disagreement; unweighted"
        pieces.append(selected)
    result = pd.concat(pieces, ignore_index=True)
    result["sample_purpose"] = "disagreement_enriched_diagnosis"
    result["population_stratum_count"] = pd.NA
    result["stratum_sample_count"] = result.groupby("sampling_stratum")[
        TEXT_INPUT_COLUMN
    ].transform("size")
    result["sampling_probability"] = pd.NA
    result["sampling_weight"] = pd.NA
    result["validation_status"] = "pending student review"
    for field in STUDENT_REVIEW_FIELDS:
        result[field] = ""
    return result.sort_values(["sampling_stratum", TEXT_INPUT_COLUMN]).reset_index(drop=True)


def manual_review_template(
    headline_comparison: pd.DataFrame,
    *,
    representative_size: int = REPRESENTATIVE_SAMPLE_SIZE,
    diagnostic_size: int = DIAGNOSTIC_SAMPLE_SIZE,
    seed: int = SAMPLE_RANDOM_SEED,
) -> pd.DataFrame:
    """Combine separately identified evaluation and diagnostic review records."""
    representative = representative_sample(
        headline_comparison, sample_size=representative_size, seed=seed
    )
    diagnostic = diagnostic_sample(
        headline_comparison,
        excluded_titles=set(representative[TEXT_INPUT_COLUMN]),
        sample_size=diagnostic_size,
        seed=seed,
    )
    result = pd.concat([representative, diagnostic], ignore_index=True, sort=False)
    if result.duplicated(TEXT_INPUT_COLUMN).any():
        raise FinBERTValidationError("manual review samples overlap by distinct title")
    return result


def disagreement_export(
    headline_comparison: pd.DataFrame,
    *,
    sample_size: int = DISAGREEMENT_EXPORT_SIZE,
    seed: int = SAMPLE_RANDOM_SEED,
) -> pd.DataFrame:
    """Return a deterministic controlled sample of descriptive disagreements."""
    population = _title_level_population(headline_comparison)
    disagreements = population.loc[population["label_agreement"].eq(False)].copy()
    if len(disagreements) < sample_size:
        raise FinBERTValidationError("not enough disagreements for controlled export")
    rng = np.random.default_rng(seed + 2)
    positions = np.sort(rng.choice(len(disagreements), size=sample_size, replace=False))
    result = disagreements.iloc[positions].copy()
    result["sample_purpose"] = "descriptive_disagreement_audit"
    result["sampling_stratum"] = np.select(
        [result["opposite_sign"], result["neutral_non_neutral"]],
        ["opposite_sign", "neutral_non_neutral"],
        default="other_label_disagreement",
    )
    result["selection_rule"] = (
        f"seed={seed + 2}; simple random sample from distinct-title label disagreements"
    )
    result["validation_status"] = "pending student review"
    for field in STUDENT_REVIEW_FIELDS:
        result[field] = ""
    return result.sort_values(["sampling_stratum", TEXT_INPUT_COLUMN]).reset_index(drop=True)


__all__ = [
    "DIAGNOSTIC_SAMPLE_SIZE",
    "DISAGREEMENT_EXPORT_SIZE",
    "FINBERT_LABEL_RULE",
    "REPRESENTATIVE_SAMPLE_SIZE",
    "SAMPLE_RANDOM_SEED",
    "STUDENT_REVIEW_FIELDS",
    "VADER_LABEL_RULE",
    "add_comparison_flags",
    "aggregate_ticker_days",
    "diagnostic_sample",
    "disagreement_export",
    "join_title_scores",
    "manual_review_template",
    "model_comparison_table",
    "representative_sample",
    "sector_sentiment_index",
    "vader_labels",
    "validate_coverage_reconciliation",
    "validate_title_score_cache",
]
