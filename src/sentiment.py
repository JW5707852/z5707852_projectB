"""VADER headline scoring and look-ahead-safe sector sentiment indices.

This module is build-only and belongs in ``requirements-dev.txt``.  The
deployed app reads the resulting CSV and must not import NLTK or score text.
"""
from __future__ import annotations

import importlib.metadata
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from src import etl, features

VADER_LEXICON_PACKAGE = "vader_lexicon"
VADER_LEXICON_ZIP = "sentiment/vader_lexicon.zip"
VADER_LEXICON_RESOURCE = (
    "sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt"
)
VADER_COMPOUND_DEFINITION = (
    "VADER normalized weighted valence sum x / sqrt(x^2 + 15), bounded to [-1, 1]"
)
TEXT_INPUT_COLUMN = "text_raw"
DEFAULT_MIN_HISTORY = 60
DEFAULT_ZSCORE_CLIP = 2.0
DEFAULT_SIGNAL_LAG = 1
EXPLORATORY_SIGNAL_WINDOW = 21
EXPLORATORY_METHOD_LABEL = "exploratory robustness extension (fixed 21-day coverage signal)"


class SentimentValidationError(RuntimeError):
    """Raised when scoring, aggregation, or signal timing is inconsistent."""


@dataclass(frozen=True)
class SentimentBuild:
    """Auditable intermediate tables and the app-readable sector index."""

    mapped_headlines: pd.DataFrame
    title_score_cache: pd.DataFrame
    headline_scores: pd.DataFrame
    ticker_day_scores: pd.DataFrame
    sector_index: pd.DataFrame
    metadata: dict[str, object]


def vader_metadata() -> dict[str, object]:
    """Return exact model provenance for artifacts and AI evidence."""
    return {
        "sentiment_model": "NLTK SentimentIntensityAnalyzer (VADER)",
        "nltk_version": importlib.metadata.version("nltk"),
        "vader_lexicon": VADER_LEXICON_RESOURCE,
        "lexicon_extension": "none",
        "score_name": "compound",
        "compound_score_definition": VADER_COMPOUND_DEFINITION,
        "text_input_column": TEXT_INPUT_COLUMN,
        "text_preprocessing": "none",
    }


def get_vader_analyzer(
    *,
    nltk_data_dir: Path | None = None,
) -> SentimentIntensityAnalyzer:
    """Load the prepared standard lexicon without downloading at score time."""
    data_dir = Path(sys.prefix) / "nltk_data" if nltk_data_dir is None else nltk_data_dir
    if data_dir.exists() and str(data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(data_dir))
    try:
        nltk.data.find(VADER_LEXICON_ZIP)
    except LookupError as error:
        raise SentimentValidationError(
            "VADER lexicon is not prepared. Install requirements-dev.txt and run "
            "the one-time build command: python -m nltk.downloader "
            f"-d {data_dir} {VADER_LEXICON_PACKAGE}"
        ) from error
    return SentimentIntensityAnalyzer(lexicon_file=VADER_LEXICON_RESOURCE)


def _validate_text_input(panel: pd.DataFrame) -> None:
    etl.require_columns(panel, {TEXT_INPUT_COLUMN}, "headline panel")
    if panel[TEXT_INPUT_COLUMN].isna().any():
        raise SentimentValidationError("text_raw contains missing titles")
    if not panel[TEXT_INPUT_COLUMN].map(lambda value: isinstance(value, str)).all():
        raise SentimentValidationError("text_raw must contain unchanged strings")
    if "title" in panel.columns and not panel["title"].equals(panel[TEXT_INPUT_COLUMN]):
        raise SentimentValidationError("text_raw no longer matches the original title")


def score_distinct_titles(
    panel: pd.DataFrame,
    *,
    analyzer: Any | None = None,
) -> pd.DataFrame:
    """Score each distinct unchanged title exactly once into an in-memory cache."""
    _validate_text_input(panel)
    model = get_vader_analyzer() if analyzer is None else analyzer
    titles = panel[[TEXT_INPUT_COLUMN]].drop_duplicates(keep="first").reset_index(drop=True)
    compounds = [
        float(model.polarity_scores(text)["compound"])
        for text in titles[TEXT_INPUT_COLUMN]
    ]
    values = np.asarray(compounds, dtype=float)
    if not np.isfinite(values).all() or (values < -1).any() or (values > 1).any():
        raise SentimentValidationError("VADER compound scores must be finite in [-1, 1]")
    titles["vader_compound"] = values
    if titles.duplicated([TEXT_INPUT_COLUMN]).any():
        raise SentimentValidationError("title score cache is not unique")
    return titles


def score_headlines(
    panel: pd.DataFrame,
    *,
    analyzer: Any | None = None,
    title_score_cache: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join one cached VADER compound score back to every clean headline row."""
    _validate_text_input(panel)
    cache = (
        score_distinct_titles(panel, analyzer=analyzer)
        if title_score_cache is None
        else title_score_cache.copy()
    )
    etl.require_columns(cache, {TEXT_INPUT_COLUMN, "vader_compound"}, "title cache")
    if cache.duplicated([TEXT_INPUT_COLUMN]).any():
        raise SentimentValidationError("title score cache must be unique by text_raw")

    original = panel.copy().reset_index(drop=True)
    original["_headline_row"] = np.arange(len(original))
    scored = original.merge(
        cache[[TEXT_INPUT_COLUMN, "vader_compound"]],
        on=TEXT_INPUT_COLUMN,
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )
    if len(scored) != len(original) or not scored["_merge"].eq("both").all():
        raise SentimentValidationError("score join lost or multiplied headline rows")
    if scored["_headline_row"].duplicated().any():
        raise SentimentValidationError("score join duplicated a clean headline row")
    scored = scored.sort_values("_headline_row").drop(
        columns=["_headline_row", "_merge"]
    )
    if not scored[TEXT_INPUT_COLUMN].equals(original[TEXT_INPUT_COLUMN]):
        raise SentimentValidationError("score join changed text_raw")
    return scored.reset_index(drop=True)


def aggregate_ticker_days(headline_scores: pd.DataFrame) -> pd.DataFrame:
    """Average headline scores to one observed ticker-day score."""
    required = {
        "trading_date",
        "ticker",
        "sector",
        TEXT_INPUT_COLUMN,
        "vader_compound",
    }
    etl.require_columns(headline_scores, required, "headline_scores")
    scored = headline_scores.dropna(subset=["trading_date"]).copy()
    scored["trading_date"] = etl.normalise_date(scored["trading_date"])
    scored["vader_compound"] = pd.to_numeric(
        scored["vader_compound"],
        errors="coerce",
    )
    if not np.isfinite(scored["vader_compound"]).all():
        raise SentimentValidationError("headline scores contain non-finite values")
    inconsistent = scored.groupby("ticker")["sector"].nunique(dropna=False)
    if (inconsistent != 1).any():
        raise SentimentValidationError("a ticker maps to multiple sectors")

    ticker_days = (
        scored.groupby(["trading_date", "ticker", "sector"], as_index=False)
        .agg(
            ticker_day_compound=("vader_compound", "mean"),
            headline_count=("vader_compound", "size"),
            distinct_title_count=(TEXT_INPUT_COLUMN, "nunique"),
        )
        .sort_values(["trading_date", "ticker"])
        .reset_index(drop=True)
    )
    if ticker_days.duplicated(["trading_date", "ticker"]).any():
        raise SentimentValidationError("ticker-day aggregation is not unique")
    return ticker_days


def _normalise_calendar(equity_calendar: Iterable[object]) -> pd.DatetimeIndex:
    calendar_values = list(equity_calendar)
    if not calendar_values:
        raise ValueError("equity_calendar must not be empty")
    calendar = pd.DatetimeIndex(
        etl.normalise_date(pd.Series(calendar_values, dtype="object"))
    )
    if calendar.has_duplicates:
        calendar = calendar.unique()
    return calendar.sort_values()


def _validate_sector_universe(sector_universe: pd.DataFrame) -> pd.DataFrame:
    etl.require_columns(sector_universe, {"ticker", "sector"}, "sector_universe")
    universe = sector_universe[["ticker", "sector"]].drop_duplicates().copy()
    if universe[["ticker", "sector"]].isna().any().any():
        raise SentimentValidationError("sector universe contains missing identifiers")
    ticker_sector_counts = universe.groupby("ticker")["sector"].nunique()
    if (ticker_sector_counts != 1).any():
        raise SentimentValidationError("sector universe maps a ticker more than once")
    return universe.sort_values(["sector", "ticker"]).reset_index(drop=True)


def _add_tradable_signal(
    sector_panel: pd.DataFrame,
    *,
    min_history: int,
    zscore_clip: float,
    signal_lag: int,
) -> pd.DataFrame:
    if min_history < 2:
        raise ValueError("min_history must be at least two observations")
    if zscore_clip <= 0:
        raise ValueError("zscore_clip must be positive")
    if signal_lag < 1:
        raise ValueError("signal_lag must be at least one trading day")

    def transform_sector(group: pd.DataFrame) -> pd.DataFrame:
        ordered = group.sort_values("date").copy()
        raw = ordered["raw_sector_compound"]
        prior_count = raw.expanding().count().shift(1).fillna(0).astype(int)
        prior_mean = raw.expanding(min_periods=min_history).mean().shift(1)
        prior_std = raw.expanding(min_periods=min_history).std(ddof=1).shift(1)
        raw_zscore = (raw - prior_mean) / prior_std
        raw_zscore = raw_zscore.where(prior_std.gt(0))
        clipped = raw_zscore.clip(-zscore_clip, zscore_clip)

        ordered["prior_observations_for_raw_z"] = prior_count
        ordered["raw_expanding_zscore"] = raw_zscore
        ordered["raw_zscore_clipped"] = clipped
        ordered["tradable_sector_zscore"] = clipped.shift(signal_lag)
        source_dates = ordered["date"].shift(signal_lag)
        ordered["tradable_signal_source_date"] = source_dates.where(
            ordered["tradable_sector_zscore"].notna()
        )
        ordered["signal_prior_observations"] = prior_count.shift(signal_lag).astype(
            "Int64"
        )
        return ordered

    transformed = pd.concat(
        [
            transform_sector(group)
            for _, group in sector_panel.groupby("sector", sort=False)
        ],
        ignore_index=True,
    )
    if transformed["tradable_signal_source_date"].notna().any():
        available = transformed.dropna(subset=["tradable_signal_source_date"])
        if not (available["tradable_signal_source_date"] < available["date"]).all():
            raise SentimentValidationError("tradable sentiment signal is not lagged")
    return transformed.sort_values(["date", "sector"]).reset_index(drop=True)


def build_coverage_adjusted_trailing_signal(
    sector_index: pd.DataFrame,
    *,
    signal_window: int = EXPLORATORY_SIGNAL_WINDOW,
    min_history: int = DEFAULT_MIN_HISTORY,
    zscore_clip: float = DEFAULT_ZSCORE_CLIP,
) -> pd.DataFrame:
    """Build the fixed exploratory 21-day signal without using date ``t``.

    For each sector-date, the source window contains the previous 21 equity
    trading dates only.  Observed sector news is coverage weighted inside the
    window; a missing sector-day is omitted rather than interpreted as neutral.
    The trailing series is then standardised against its own strictly prior
    observations before the clipped z-score is shrunk by effective coverage.

    The parameter values are intentionally locked.  They are exposed only to
    make the design explicit and are rejected if changed, preventing accidental
    final-period parameter search through the production function.
    """
    if signal_window != EXPLORATORY_SIGNAL_WINDOW:
        raise ValueError(
            f"signal_window is locked at {EXPLORATORY_SIGNAL_WINDOW} trading days"
        )
    if min_history != DEFAULT_MIN_HISTORY:
        raise ValueError(f"min_history is locked at {DEFAULT_MIN_HISTORY}")
    if zscore_clip != DEFAULT_ZSCORE_CLIP:
        raise ValueError(f"zscore_clip is locked at {DEFAULT_ZSCORE_CLIP}")

    required = {
        "date",
        "sector",
        "raw_sector_compound",
        "observed_ticker_count",
        "possible_ticker_count",
        "ticker_coverage_share",
        "has_observed_news",
    }
    etl.require_columns(sector_index, required, "sector_index")
    panel = sector_index[list(required)].copy()
    panel["date"] = etl.normalise_date(panel["date"])
    if panel.duplicated(["date", "sector"]).any():
        raise SentimentValidationError("sector_index contains duplicate date-sector keys")

    numeric_columns = [
        "raw_sector_compound",
        "observed_ticker_count",
        "possible_ticker_count",
        "ticker_coverage_share",
    ]
    for column in numeric_columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    coverage_columns = [
        "observed_ticker_count",
        "possible_ticker_count",
        "ticker_coverage_share",
    ]
    if panel[coverage_columns].isna().any().any():
        raise SentimentValidationError("sector coverage inputs contain missing values")
    if (panel["possible_ticker_count"] <= 0).any():
        raise SentimentValidationError("possible_ticker_count must be positive")
    if (panel["observed_ticker_count"] < 0).any() or (
        panel["observed_ticker_count"] > panel["possible_ticker_count"]
    ).any():
        raise SentimentValidationError("observed ticker counts are outside their bounds")
    expected_coverage = (
        panel["observed_ticker_count"] / panel["possible_ticker_count"]
    )
    if not np.allclose(
        panel["ticker_coverage_share"], expected_coverage, atol=1e-12, rtol=0.0
    ):
        raise SentimentValidationError("ticker coverage shares disagree with ticker counts")
    has_news = panel["has_observed_news"].astype(bool)
    if panel.loc[~has_news, "raw_sector_compound"].notna().any():
        raise SentimentValidationError("no-news sector-days must remain missing")
    if panel.loc[has_news, "raw_sector_compound"].isna().any():
        raise SentimentValidationError("observed-news sector-days require a score")

    sector_outputs: list[pd.DataFrame] = []
    for sector, group in panel.groupby("sector", sort=True):
        ordered = group.sort_values("date").reset_index(drop=True).copy()
        possible_counts = ordered["possible_ticker_count"].drop_duplicates()
        if len(possible_counts) != 1:
            raise SentimentValidationError(
                f"possible_ticker_count changes through time for sector {sector}"
            )
        possible_count = float(possible_counts.iloc[0])

        window_start_dates: list[pd.Timestamp | pd.NaT] = []
        window_end_dates: list[pd.Timestamp | pd.NaT] = []
        latest_news_dates: list[pd.Timestamp | pd.NaT] = []
        trailing_values: list[float] = []
        effective_coverages: list[float] = []
        actual_window_sizes: list[int] = []

        for position in range(len(ordered)):
            start = max(0, position - signal_window)
            prior_window = ordered.iloc[start:position]
            actual_window_sizes.append(len(prior_window))
            if prior_window.empty:
                window_start_dates.append(pd.NaT)
                window_end_dates.append(pd.NaT)
                latest_news_dates.append(pd.NaT)
                trailing_values.append(np.nan)
                effective_coverages.append(0.0)
                continue

            window_start_dates.append(pd.Timestamp(prior_window["date"].iloc[0]))
            window_end_dates.append(pd.Timestamp(prior_window["date"].iloc[-1]))
            observed = prior_window[
                prior_window["has_observed_news"].astype(bool)
                & prior_window["raw_sector_compound"].notna()
            ]
            denominator = float(observed["ticker_coverage_share"].sum())
            if denominator > 0.0:
                numerator = float(
                    (
                        observed["raw_sector_compound"]
                        * observed["ticker_coverage_share"]
                    ).sum()
                )
                trailing_values.append(numerator / denominator)
                latest_news_dates.append(pd.Timestamp(observed["date"].max()))
            else:
                trailing_values.append(np.nan)
                latest_news_dates.append(pd.NaT)
            effective_coverages.append(
                float(prior_window["observed_ticker_count"].sum())
                / (signal_window * possible_count)
            )

        ordered["signal_window_trading_days"] = signal_window
        ordered["signal_window_actual_trading_days"] = actual_window_sizes
        ordered["signal_window_start_date"] = window_start_dates
        ordered["signal_window_end_date"] = window_end_dates
        ordered["latest_raw_news_date_used"] = latest_news_dates
        ordered["trailing_coverage_weighted_sentiment"] = trailing_values
        ordered["effective_coverage"] = effective_coverages

        trailing = ordered["trailing_coverage_weighted_sentiment"]
        prior_count = trailing.expanding().count().shift(1).fillna(0).astype(int)
        prior_mean = trailing.expanding(min_periods=min_history).mean().shift(1)
        prior_std = trailing.expanding(min_periods=min_history).std(ddof=1).shift(1)
        raw_zscore = ((trailing - prior_mean) / prior_std).where(prior_std.gt(0))
        clipped_zscore = raw_zscore.clip(-zscore_clip, zscore_clip)
        coverage_adjusted = clipped_zscore * np.sqrt(ordered["effective_coverage"])

        ordered["expanding_prior_mean"] = prior_mean
        ordered["expanding_prior_std"] = prior_std
        ordered["expanding_prior_observations"] = prior_count
        ordered["raw_trailing_zscore"] = raw_zscore
        ordered["clipped_trailing_zscore"] = clipped_zscore
        ordered["coverage_adjusted_zscore"] = coverage_adjusted
        ordered["min_history_observations"] = min_history
        ordered["zscore_clip_bound"] = zscore_clip
        ordered["zscore_std_ddof"] = 1
        ordered["exploratory_method_label"] = EXPLORATORY_METHOD_LABEL
        sector_outputs.append(ordered)

    result = (
        pd.concat(sector_outputs, ignore_index=True)
        .sort_values(["date", "sector"])
        .reset_index(drop=True)
    )
    if not result["effective_coverage"].between(0.0, 1.0).all():
        raise SentimentValidationError("effective coverage must remain in [0, 1]")
    dated_windows = result.dropna(subset=["signal_window_end_date"])
    if not (dated_windows["signal_window_end_date"] < dated_windows["date"]).all():
        raise SentimentValidationError("the 21-day window includes its target date")
    used_news = result.dropna(subset=["latest_raw_news_date_used"])
    if not (used_news["latest_raw_news_date_used"] < used_news["date"]).all():
        raise SentimentValidationError("exploratory signal uses current or future news")
    finite_signal = result["coverage_adjusted_zscore"].dropna()
    if not np.isfinite(finite_signal).all():
        raise SentimentValidationError("exploratory signal contains non-finite values")
    return result


def sector_sentiment_index(
    ticker_day_scores: pd.DataFrame,
    equity_calendar: Iterable[object],
    sector_universe: pd.DataFrame,
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    zscore_clip: float = DEFAULT_ZSCORE_CLIP,
    signal_lag: int = DEFAULT_SIGNAL_LAG,
) -> pd.DataFrame:
    """Equal-weight observed ticker-day scores within each sector and date."""
    required = {
        "trading_date",
        "ticker",
        "sector",
        "ticker_day_compound",
        "headline_count",
    }
    etl.require_columns(ticker_day_scores, required, "ticker_day_scores")
    calendar = _normalise_calendar(equity_calendar)
    universe = _validate_sector_universe(sector_universe)
    ticker_days = ticker_day_scores.copy()
    ticker_days["trading_date"] = etl.normalise_date(ticker_days["trading_date"])
    if ticker_days.duplicated(["trading_date", "ticker"]).any():
        raise SentimentValidationError("ticker_day_scores contains duplicate keys")
    if not ticker_days["trading_date"].isin(calendar).all():
        raise SentimentValidationError("ticker-day scores fall outside the calendar")

    checked = ticker_days.merge(
        universe.rename(columns={"sector": "universe_sector"}),
        on="ticker",
        how="left",
        validate="many_to_one",
    )
    if checked["universe_sector"].isna().any() or not checked["sector"].eq(
        checked["universe_sector"]
    ).all():
        raise SentimentValidationError("ticker-day sector mapping disagrees with universe")

    observed = (
        checked.groupby(["trading_date", "sector"], as_index=False)
        .agg(
            raw_sector_compound=("ticker_day_compound", "mean"),
            observed_ticker_count=("ticker", "nunique"),
            headline_count=("headline_count", "sum"),
        )
        .rename(columns={"trading_date": "date"})
    )
    sectors = sorted(universe["sector"].unique())
    grid = pd.MultiIndex.from_product(
        [calendar, sectors],
        names=["date", "sector"],
    ).to_frame(index=False)
    panel = grid.merge(observed, on=["date", "sector"], how="left", validate="one_to_one")
    possible = universe.groupby("sector")["ticker"].nunique()
    panel["possible_ticker_count"] = panel["sector"].map(possible).astype(int)
    panel["observed_ticker_count"] = (
        panel["observed_ticker_count"].fillna(0).astype(int)
    )
    panel["headline_count"] = panel["headline_count"].fillna(0).astype(int)
    panel["ticker_coverage_share"] = (
        panel["observed_ticker_count"] / panel["possible_ticker_count"]
    )
    panel["has_observed_news"] = panel["observed_ticker_count"].gt(0)
    if not panel.loc[~panel["has_observed_news"], "raw_sector_compound"].isna().all():
        raise SentimentValidationError("no-news sector-days must remain missing")

    panel = _add_tradable_signal(
        panel,
        min_history=min_history,
        zscore_clip=zscore_clip,
        signal_lag=signal_lag,
    )
    metadata = vader_metadata()
    for name, value in metadata.items():
        panel[name] = value
    panel["raw_index_purpose"] = "retrospective descriptive index"
    panel["tradable_signal_purpose"] = "prior-day expanding z-score"
    panel["min_history_observations"] = min_history
    panel["zscore_clip_bound"] = zscore_clip
    panel["signal_lag_trading_days"] = signal_lag
    panel["zscore_std_ddof"] = 1
    return panel


def build_sentiment_index(
    headlines: pd.DataFrame,
    equity_prices: pd.DataFrame,
    *,
    analyzer: Any | None = None,
    min_history: int = DEFAULT_MIN_HISTORY,
    zscore_clip: float = DEFAULT_ZSCORE_CLIP,
    signal_lag: int = DEFAULT_SIGNAL_LAG,
    expected_sector_count: int | None = None,
) -> SentimentBuild:
    """Build all audited sentiment stages from clean project inputs."""
    etl.require_columns(equity_prices, {"date", "ticker", "sector"}, "equity_prices")
    calendar = etl.normalise_date(equity_prices["date"]).drop_duplicates().sort_values()
    universe = equity_prices[["ticker", "sector"]].drop_duplicates()
    if expected_sector_count is not None and universe["sector"].nunique() != expected_sector_count:
        raise SentimentValidationError(
            f"expected {expected_sector_count} sectors, found {universe['sector'].nunique()}"
        )

    mapped = features.assemble_headline_panel(headlines, calendar)
    cache = score_distinct_titles(mapped, analyzer=analyzer)
    scored = score_headlines(mapped, title_score_cache=cache)
    if len(scored) != len(mapped):
        raise SentimentValidationError("headline scoring changed clean row count")
    ticker_days = aggregate_ticker_days(scored)
    sector_index = sector_sentiment_index(
        ticker_days,
        calendar,
        universe,
        min_history=min_history,
        zscore_clip=zscore_clip,
        signal_lag=signal_lag,
    )
    return SentimentBuild(
        mapped_headlines=mapped,
        title_score_cache=cache,
        headline_scores=scored,
        ticker_day_scores=ticker_days,
        sector_index=sector_index,
        metadata=vader_metadata(),
    )


__all__ = [
    "DEFAULT_MIN_HISTORY",
    "DEFAULT_SIGNAL_LAG",
    "DEFAULT_ZSCORE_CLIP",
    "EXPLORATORY_METHOD_LABEL",
    "EXPLORATORY_SIGNAL_WINDOW",
    "SentimentBuild",
    "SentimentValidationError",
    "aggregate_ticker_days",
    "build_coverage_adjusted_trailing_signal",
    "build_sentiment_index",
    "get_vader_analyzer",
    "score_distinct_titles",
    "score_headlines",
    "sector_sentiment_index",
    "vader_metadata",
]
