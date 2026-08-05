"""Focused regression checks for the published Phase 2 FinBERT artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scripts import run_finbert_innovation
from src import finbert_innovation, finbert_phase2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_STEMS = (
    "sentiment_model_score_distribution",
    "sentiment_model_sector_timeseries",
    "sentiment_model_sector_agreement",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_full_title_cache_and_sector_index_contracts() -> None:
    titles = pd.read_csv(PROJECT_ROOT / "results/data/finbert_title_scores.csv")
    checked = finbert_phase2.validate_title_score_cache(titles)
    assert len(checked) == 105_334
    assert checked["text_raw"].nunique() == 105_334
    assert np.isfinite(
        checked[
            [
                "probability_positive",
                "probability_negative",
                "probability_neutral",
                "finbert_score",
            ]
        ].to_numpy(dtype=float)
    ).all()

    sector = pd.read_csv(PROJECT_ROOT / "results/data/sector_sentiment_finbert.csv")
    assert len(sector) == 10_060
    assert sector["sector"].nunique() == 10
    assert sector["date"].nunique() == 1_006
    assert not sector.duplicated(["date", "sector"]).any()
    assert int(sector["has_observed_news"].sum()) == 9_832
    assert sector.loc[~sector["has_observed_news"], "raw_sector_finbert"].isna().all()


def test_comparison_correlations_are_matched_and_review_samples_are_separate() -> None:
    comparison = pd.read_csv(PROJECT_ROOT / "results/tables/sentiment_model_comparison.csv")
    headline_rows = comparison["observation_unit"].eq("clean_headline_row")
    matched_rows = comparison["observation_unit"].eq("matched_date_sector")
    assert comparison.loc[headline_rows, "pearson_correlation"].isna().all()
    assert comparison.loc[headline_rows, "spearman_correlation"].isna().all()
    assert comparison.loc[matched_rows, "pearson_correlation"].notna().all()
    assert comparison.loc[matched_rows, "spearman_correlation"].notna().all()
    assert comparison.loc[matched_rows, "paired_observation_count"].gt(0).all()
    overall = comparison.loc[matched_rows & comparison["sector"].eq("All")].iloc[0]
    assert overall["paired_observation_count"] == 9_832
    assert "not accuracy" in overall["interpretation_status"]

    review = pd.read_csv(
        PROJECT_ROOT / "results/tables/sentiment_manual_review_template.csv",
        keep_default_na=False,
    )
    assert review["sample_purpose"].value_counts().to_dict() == {
        "representative_evaluation": 100,
        "disagreement_enriched_diagnosis": 50,
    }
    assert not review["text_raw"].duplicated().any()
    assert review[list(finbert_phase2.STUDENT_REVIEW_FIELDS)].eq("").all().all()
    assert review["validation_status"].eq("pending student review").all()
    representative = review["sample_purpose"].eq("representative_evaluation")
    diagnostic = review["sample_purpose"].eq("disagreement_enriched_diagnosis")
    assert review.loc[representative, "sampling_weight"].ne("").all()
    assert review.loc[diagnostic, "sampling_weight"].eq("").all()
    expected_diagnostic_strata = {
        "opposite_sign",
        "neutral_non_neutral",
        "negation",
        "financial_term",
        "numerical",
        "earnings_announcement",
    }
    assert set(review.loc[diagnostic, "sampling_stratum"]) == expected_diagnostic_strata


def test_metadata_figures_manifest_and_protected_hashes() -> None:
    protected_hashes = run_finbert_innovation.protected_hashes()
    metadata = json.loads(
        (PROJECT_ROOT / "results/tables/finbert_run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["phase_status"] == "PASS"
    assert metadata["model_name"] == finbert_innovation.MODEL_NAME
    assert metadata["pinned_revision"] == finbert_innovation.MODEL_REVISION
    assert metadata["resolved_revision"] == finbert_innovation.MODEL_REVISION
    assert metadata["device"] == "mps"
    assert metadata["batch_size"] == 32
    assert metadata["max_length"] == 128
    assert metadata["truncated_title_count"] == 0
    assert metadata["truncated_title_percentage"] == 0.0
    assert metadata["correlation_policy"].startswith("reported only for matched")
    assert metadata["protected_artifact_hashes_before"] == protected_hashes
    assert metadata["protected_artifact_hashes_after"] == protected_hashes

    qa = pd.read_csv(PROJECT_ROOT / "results/tables/finbert_figure_qa.csv")
    assert set(qa["figure"]) == set(FIGURE_STEMS)
    assert qa["qa_status"].eq("PASS").all()
    for stem in FIGURE_STEMS:
        png = PROJECT_ROOT / f"results/figures/{stem}.png"
        pdf = PROJECT_ROOT / f"results/figures/{stem}.pdf"
        caption = PROJECT_ROOT / f"results/figures/{stem}.caption.md"
        assert png.is_file() and pdf.is_file() and caption.is_file()
        with Image.open(png) as image:
            assert image.width >= 1_800
            assert image.height >= 1_100
        assert caption.read_text(encoding="utf-8").strip()

    manifest = json.loads(
        (PROJECT_ROOT / "results/tables/finbert_phase2_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["completion_status"] == "PASS"
    for relative, expected_hash in manifest["files"].items():
        assert _sha256(PROJECT_ROOT / relative) == expected_hash
    for relative, expected_hash in protected_hashes.items():
        assert _sha256(PROJECT_ROOT / relative) == expected_hash
