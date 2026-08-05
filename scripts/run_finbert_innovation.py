"""Run the explicitly gated FinBERT robustness-layer build.

``--pilot-only`` runs the Phase 1 benchmark. ``--full-corpus`` is the only
mode that may create Phase 2 artifacts; no inference mode is implicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_matplotlib_cache = Path(tempfile.gettempdir()) / "portfoyou-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from src import etl, features, finbert_innovation, finbert_phase2, sentiment  # noqa: E402

from fintools.figures import (  # noqa: E402
    FigureContext,
    export_figure_bundle,
    figure_style,
    validate_axes_labels,
    validate_figure_context,
    validate_image_not_blank,
    validate_titles_within_canvas,
    validate_word_readability,
)

PROTECTED_ARTIFACTS = (
    Path("results/data/fund_returns.csv"),
    Path("results/data/fund_weights.csv"),
    Path("results/data/sector_sentiment_index.csv"),
    Path("results/tables/performance_metrics.csv"),
)
PHASE2_OUTPUTS = {
    "title_scores": Path("results/data/finbert_title_scores.csv"),
    "sector_index": Path("results/data/sector_sentiment_finbert.csv"),
    "comparison": Path("results/tables/sentiment_model_comparison.csv"),
    "disagreements": Path("results/tables/sentiment_model_disagreements.csv"),
    "manual_review": Path("results/tables/sentiment_manual_review_template.csv"),
    "metadata": Path("results/tables/finbert_run_metadata.json"),
    "figure_qa": Path("results/tables/finbert_figure_qa.csv"),
    "manifest": Path("results/tables/finbert_phase2_manifest.json"),
}
FIGURE_STEMS = (
    "sentiment_model_score_distribution",
    "sentiment_model_sector_timeseries",
    "sentiment_model_sector_agreement",
)
COLORS = {"VADER": "#1F3A5F", "FinBERT": "#B23A48"}


class Phase2BuildError(RuntimeError):
    """Raised when the staged Phase 2 artifact set cannot be validated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    """Return current hashes for the four artifacts this script may not alter."""
    hashes: dict[str, str] = {}
    for relative in PROTECTED_ARTIFACTS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"protected artifact is missing: {relative}")
        hashes[relative.as_posix()] = _sha256(path)
    return hashes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run gated ProsusAI/finbert pilot or full-corpus inference."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--pilot-only",
        action="store_true",
        help="Run only the mandatory pre-full-corpus pilot gate.",
    )
    modes.add_argument(
        "--full-corpus",
        action="store_true",
        help="Run and publish the complete Phase 2 artifact set.",
    )
    modes.add_argument(
        "--postprocess-existing",
        action="store_true",
        help="Rebuild Phase 2 derived artifacts from an already validated title cache.",
    )
    modes.add_argument(
        "--refresh-integrity-metadata",
        action="store_true",
        help=(
            "Refresh only the protected-core hash evidence after an unrelated "
            "core-artifact update; it does not rerun FinBERT or alter model outputs."
        ),
    )
    parser.add_argument("--pilot-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=finbert_innovation.DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=finbert_innovation.DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="External Hugging Face cache; defaults to the system temp directory.",
    )
    return parser


def _prepare_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    equities = etl.load_clean_equities()
    headlines = etl.load_clean_news()
    calendar = equities["date"].drop_duplicates().sort_values()
    mapped = features.assemble_headline_panel(headlines, calendar)
    universe = equities[["ticker", "sector"]].drop_duplicates()
    return mapped, equities, calendar, universe


def _load_model(
    args: argparse.Namespace, cache_dir: Path
) -> tuple[finbert_innovation.LoadedFinBERT, dict[str, float]]:
    peak_before_model = finbert_innovation.process_peak_rss_mb()
    started = time.perf_counter()
    loaded = finbert_innovation.load_pinned_finbert(
        cache_dir=cache_dir,
        device=args.device,
    )
    load_seconds = time.perf_counter() - started
    peak_after_model = finbert_innovation.process_peak_rss_mb()
    return loaded, {
        "model_load_seconds": load_seconds,
        "peak_rss_before_model_mb": peak_before_model,
        "peak_rss_after_model_mb": peak_after_model,
        "peak_rss_model_increment_mb": max(0.0, peak_after_model - peak_before_model),
    }


def _inference_evidence(
    inference: finbert_innovation.FinBERTInference,
) -> dict[str, object]:
    evidence: dict[str, object] = asdict(inference.metrics)
    probability_columns = [
        "probability_positive",
        "probability_negative",
        "probability_neutral",
    ]
    probability_sums = inference.title_scores[probability_columns].sum(axis=1)
    evidence["minimum_probability_sum"] = float(probability_sums.min())
    evidence["maximum_probability_sum"] = float(probability_sums.max())
    evidence["finite_probability_rows"] = int(
        np.isfinite(inference.title_scores[probability_columns].to_numpy(dtype=float))
        .all(axis=1)
        .sum()
    )
    evidence["score_arithmetic_validated_rows"] = int(
        np.isclose(
            inference.title_scores["finbert_score"],
            inference.title_scores["probability_positive"]
            - inference.title_scores["probability_negative"],
            atol=1e-12,
            rtol=0.0,
        ).sum()
    )
    return evidence


def run_pilot(args: argparse.Namespace) -> dict[str, object]:
    """Run deterministic pilot inference and return auditable benchmark evidence."""
    if not 512 <= args.pilot_size <= 1000:
        raise ValueError("pilot-size must be between 512 and 1,000")
    cache_candidate = args.cache_dir or (Path(tempfile.gettempdir()) / "portfoyou-huggingface")
    cache_dir = finbert_innovation.validate_external_cache(cache_candidate, PROJECT_ROOT)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hashes_before = protected_hashes()
    mapped, _, _, _ = _prepare_inputs()
    distinct_count = len(finbert_innovation.distinct_titles(mapped))
    pilot = finbert_innovation.deterministic_pilot_sample(
        mapped,
        sample_size=args.pilot_size,
        seed=finbert_innovation.PILOT_RANDOM_SEED,
    )
    loaded, load_evidence = _load_model(args, cache_dir)
    inference = finbert_innovation.infer_distinct_titles(
        pilot,
        loaded,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    hashes_after = protected_hashes()
    if hashes_after != hashes_before:
        raise RuntimeError("pilot changed a protected artifact")
    estimated_full_seconds = distinct_count / inference.metrics.titles_per_second
    return {
        "phase": "phase_1_pilot",
        "pilot_status": "PASS",
        "model_name": finbert_innovation.MODEL_NAME,
        "pinned_revision": finbert_innovation.MODEL_REVISION,
        "resolved_revision": loaded.resolved_revision,
        "model_license": finbert_innovation.MODEL_LICENSE,
        "label_mapping": loaded.id2label,
        "score_definition": "probability_positive - probability_negative",
        "text_input": "unchanged text_raw",
        "clean_headline_rows": len(mapped),
        "full_distinct_title_count": distinct_count,
        "pilot_random_seed": finbert_innovation.PILOT_RANDOM_SEED,
        "estimated_full_inference_seconds": estimated_full_seconds,
        "estimated_full_inference_minutes": estimated_full_seconds / 60.0,
        "protected_artifact_hashes_before": hashes_before,
        "protected_artifact_hashes_after": hashes_after,
        "cache_policy": "system external cache outside submission folder",
        **load_evidence,
        **finbert_innovation.package_versions(),
        **_inference_evidence(inference),
    }


class _ProgressReporter:
    def __init__(self) -> None:
        self.next_percent = 10

    def __call__(self, completed: int, total: int, elapsed: float) -> None:
        percent = int(100 * completed / total)
        if percent >= self.next_percent or completed == total:
            throughput = completed / elapsed if elapsed > 0 else 0.0
            remaining = (total - completed) / throughput if throughput > 0 else np.nan
            print(
                f"[inference] {completed:,}/{total:,} ({percent}%) | "
                f"{throughput:.1f} titles/s | ETA {remaining / 60.0:.1f} min",
                flush=True,
            )
            while self.next_percent <= percent:
                self.next_percent += 10


def _add_figure_metadata(
    fig: plt.Figure,
    *,
    sample: str,
    units: str,
    definition: str,
    source: str,
) -> None:
    sample_line = textwrap.fill(f"Sample: {sample}  |  Units: {units}", width=105)
    definition_line = textwrap.fill(f"Definition: {definition}", width=105)
    source_line = textwrap.fill(f"Source: {source}", width=105)
    text = f"{sample_line}\n{definition_line}\n{source_line}"
    fig.text(0.02, 0.012, text, ha="left", va="bottom", fontsize=7.0, color="#4B5563")


def _export_figure(
    fig: plt.Figure,
    axes: list[plt.Axes],
    *,
    stage_figure_dir: Path,
    stem: str,
    context: FigureContext,
    source_rows: int,
    require_axis_labels: bool = True,
) -> dict[str, object]:
    issues = [issue.code for issue in validate_figure_context(context)]
    issues.extend(issue.code for issue in validate_titles_within_canvas(fig))
    issues.extend(
        issue.code for issue in validate_word_readability(fig, width_inches=6.27, min_font_size=7.0)
    )
    if require_axis_labels:
        for axis in axes:
            issues.extend(issue.code for issue in validate_axes_labels(axis))
    if issues:
        raise Phase2BuildError(f"{stem} failed in-memory figure QA: {sorted(set(issues))}")
    paths = export_figure_bundle(
        fig,
        stage_figure_dir,
        stem,
        context=context,
        formats=("png", "pdf"),
        dpi=300,
    )
    blank = [issue.code for issue in validate_image_not_blank(paths["png"])]
    if blank:
        raise Phase2BuildError(f"{stem} exported blank: {blank}")
    with Image.open(paths["png"]) as image:
        width_px, height_px = image.size
    row = {
        "figure": stem,
        "source_rows": source_rows,
        "png_width_pixels": width_px,
        "png_height_pixels": height_px,
        "layout_issue_count": 0,
        "blank_image_issue_count": 0,
        "qa_status": "PASS",
    }
    plt.close(fig)
    return row


def _generate_figures(
    title_comparison: pd.DataFrame,
    matched_sector: pd.DataFrame,
    comparison: pd.DataFrame,
    stage_figure_dir: Path,
) -> pd.DataFrame:
    stage_figure_dir.mkdir(parents=True, exist_ok=True)
    qa: list[dict[str, object]] = []
    source = (
        "Hosted project headlines via src/data_access.py; protected VADER sector "
        "index; generated by scripts/run_finbert_innovation.py"
    )

    title_level = title_comparison.sort_values(
        ["text_raw", "date", "ticker"], kind="mergesort"
    ).drop_duplicates("text_raw")
    with figure_style(profile="word_a4", style="fins"):
        fig, ax = plt.subplots(figsize=(6.27, 4.0), layout="none")
        bins = np.linspace(-1, 1, 42)
        ax.hist(
            title_level["vader_compound"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=COLORS["VADER"],
            label="VADER compound",
        )
        ax.hist(
            title_level["finbert_score"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=COLORS["FinBERT"],
            label="FinBERT P(pos) - P(neg)",
        )
        ax.axvline(0, color="#6B7280", linewidth=0.8)
        ax.set_title("Headline sentiment score distributions", loc="left")
        ax.set_xlabel("Sentiment score")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left", frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.29)
        definition = (
            "Each unchanged distinct title contributes once; distributions are descriptive."
        )
        _add_figure_metadata(
            fig,
            sample=f"{len(title_level):,} distinct titles",
            units="Probability density over score in [-1, 1]",
            definition=definition,
            source=source,
        )
    context = FigureContext(
        title="Headline sentiment score distributions",
        note=definition,
        source=source,
        sample=f"{len(title_level):,} distinct titles",
        units="Probability density",
    )
    qa.append(
        _export_figure(
            fig,
            [ax],
            stage_figure_dir=stage_figure_dir,
            stem="sentiment_model_score_distribution",
            context=context,
            source_rows=len(title_level),
        )
    )

    sectors = sorted(matched_sector["sector"].unique())
    matched = matched_sector.copy()
    matched["vader_rolling_21"] = matched.groupby("sector")["vader_compound"].transform(
        lambda values: values.rolling(21, min_periods=5).mean()
    )
    matched["finbert_rolling_21"] = matched.groupby("sector")["finbert_score"].transform(
        lambda values: values.rolling(21, min_periods=5).mean()
    )
    with figure_style(profile="word_a4", style="fins"):
        fig, axes_grid = plt.subplots(
            5, 2, figsize=(6.27, 8.8), sharex=True, sharey=True, layout="none"
        )
        axes = list(axes_grid.flat)
        for ax, sector_name in zip(axes, sectors, strict=True):
            group = matched.loc[matched["sector"].eq(sector_name)]
            ax.plot(group["date"], group["vader_rolling_21"], color=COLORS["VADER"], linewidth=1.0)
            ax.plot(
                group["date"], group["finbert_rolling_21"], color=COLORS["FinBERT"], linewidth=1.0
            )
            ax.axhline(0, color="#9CA3AF", linewidth=0.55)
            ax.set_title(sector_name, loc="left", fontsize=9)
            ax.grid(axis="y", alpha=0.20)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[-2].set_xlabel("Date")
        axes[-1].set_xlabel("Date")
        for ax in axes[::2]:
            ax.set_ylabel("Score")
        handles = [
            plt.Line2D([0], [0], color=COLORS["VADER"], label="VADER"),
            plt.Line2D([0], [0], color=COLORS["FinBERT"], label="FinBERT"),
        ]
        fig.legend(
            handles=handles, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 0.955)
        )
        fig.suptitle(
            "Sector sentiment comparison (21-day descriptive means)", x=0.06, ha="left", y=0.995
        )
        fig.subplots_adjust(
            left=0.11,
            right=0.98,
            top=0.90,
            bottom=0.18,
            hspace=0.42,
            wspace=0.18,
        )
        sample_start = matched["date"].min().strftime("%Y-%m-%d")
        sample_end = matched["date"].max().strftime("%Y-%m-%d")
        definition = (
            "Rolling means use matched observed date-sector rows; missing news "
            "days are not neutral-filled."
        )
        _add_figure_metadata(
            fig,
            sample=f"{sample_start} to {sample_end}; {len(matched):,} matched observations",
            units="Sentiment score",
            definition=definition,
            source=source,
        )
    context = FigureContext(
        title="Sector sentiment comparison (21-day descriptive means)",
        note=definition,
        source=source,
        sample=f"{sample_start} to {sample_end}; {len(matched):,} matched observations",
        units="Sentiment score",
    )
    qa.append(
        _export_figure(
            fig,
            axes,
            stage_figure_dir=stage_figure_dir,
            stem="sentiment_model_sector_timeseries",
            context=context,
            source_rows=len(matched),
            require_axis_labels=False,
        )
    )

    sector_stats = comparison.loc[
        comparison["observation_unit"].eq("matched_date_sector") & comparison["sector"].ne("All")
    ].sort_values("pearson_correlation")
    with figure_style(profile="word_a4", style="fins"):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(6.27, 4.8), sharey=True, layout="none"
        )
        positions = np.arange(len(sector_stats))
        ax1.barh(positions, sector_stats["pearson_correlation"], color="#1F3A5F")
        ax2.barh(positions, 100 * sector_stats["descriptive_label_agreement_rate"], color="#007C89")
        ax1.set_yticks(positions, sector_stats["sector"])
        ax1.set_xlabel("Pearson correlation")
        ax1.set_ylabel("Sector")
        ax2.set_xlabel("Label agreement (%)")
        ax2.set_ylabel("Sector")
        ax1.set_title("Score correlation", loc="left")
        ax2.set_title("Descriptive agreement", loc="left")
        ax1.axvline(0, color="#6B7280", linewidth=0.8)
        for ax in (ax1, ax2):
            ax.grid(axis="x", alpha=0.25)
        fig.suptitle("Matched sector-day model comparison", x=0.08, ha="left", y=0.985)
        fig.subplots_adjust(left=0.20, right=0.98, top=0.80, bottom=0.29, wspace=0.18)
        definition = "Statistics use matched observed date-sector pairs; agreement is not accuracy."
        _add_figure_metadata(
            fig,
            sample=f"{len(matched):,} paired date-sector observations",
            units="Correlation coefficient and percent",
            definition=definition,
            source=source,
        )
    context = FigureContext(
        title="Matched sector-day model comparison",
        note=definition,
        source=source,
        sample=f"{len(matched):,} paired date-sector observations",
        units="Correlation coefficient and percent",
    )
    qa.append(
        _export_figure(
            fig,
            [ax1, ax2],
            stage_figure_dir=stage_figure_dir,
            stem="sentiment_model_sector_agreement",
            context=context,
            source_rows=len(matched),
        )
    )
    return pd.DataFrame.from_records(qa)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _validate_staged_outputs(stage_root: Path, metadata: dict[str, object]) -> None:
    title_scores = pd.read_csv(stage_root / PHASE2_OUTPUTS["title_scores"])
    finbert_phase2.validate_title_score_cache(title_scores)
    if len(title_scores) != 105_334:
        raise Phase2BuildError(f"expected 105,334 title scores, found {len(title_scores):,}")
    sector = pd.read_csv(stage_root / PHASE2_OUTPUTS["sector_index"])
    if len(sector) != 10_060 or sector.duplicated(["date", "sector"]).any():
        raise Phase2BuildError("FinBERT sector index grid is incomplete or duplicated")
    comparison = pd.read_csv(stage_root / PHASE2_OUTPUTS["comparison"])
    if comparison["paired_observation_count"].isna().any():
        raise Phase2BuildError("a comparison statistic lacks its paired count")
    review = pd.read_csv(stage_root / PHASE2_OUTPUTS["manual_review"], keep_default_na=False)
    purposes = review["sample_purpose"].value_counts().to_dict()
    expected = {"representative_evaluation": 100, "disagreement_enriched_diagnosis": 50}
    if purposes != expected:
        raise Phase2BuildError(f"manual review sample split differs: {purposes}")
    if review[list(finbert_phase2.STUDENT_REVIEW_FIELDS)].ne("").any().any():
        raise Phase2BuildError("student-owned review fields are not blank")
    if not review["validation_status"].eq("pending student review").all():
        raise Phase2BuildError("manual review status is not pending student review")
    qa = pd.read_csv(stage_root / PHASE2_OUTPUTS["figure_qa"])
    if len(qa) != len(FIGURE_STEMS) or not qa["qa_status"].eq("PASS").all():
        raise Phase2BuildError("figure QA did not pass for all Phase 2 figures")
    if int(metadata["distinct_title_count"]) != len(title_scores):
        raise Phase2BuildError("metadata distinct-title count differs from artifact")


def _promote_stage(stage_root: Path) -> dict[str, str]:
    staged_files = [path for path in stage_root.rglob("*") if path.is_file()]
    manifest_relative = PHASE2_OUTPUTS["manifest"]
    staged_files.sort(key=lambda path: path.relative_to(stage_root) == manifest_relative)
    promoted_hashes: dict[str, str] = {}
    for source in staged_files:
        relative = source.relative_to(stage_root)
        target = PROJECT_ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        promoted_hashes[relative.as_posix()] = _sha256(target)
    return promoted_hashes


def run_full_corpus(args: argparse.Namespace) -> dict[str, object]:
    """Run the approved full inference and atomically publish validated outputs."""
    if args.device != "mps" or args.batch_size != 32 or args.max_length != 128:
        raise ValueError(
            "approved Phase 2 execution requires --device mps --batch-size 32 --max-length 128"
        )
    cache_candidate = args.cache_dir or (Path(tempfile.gettempdir()) / "portfoyou-huggingface")
    cache_dir = finbert_innovation.validate_external_cache(cache_candidate, PROJECT_ROOT)
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    hashes_before = protected_hashes()
    print("[stage] protected hashes captured; loading clean data", flush=True)
    mapped, _, calendar, universe = _prepare_inputs()
    distinct_count = len(finbert_innovation.distinct_titles(mapped))
    if len(mapped) != 146_836 or distinct_count != 105_334:
        raise Phase2BuildError(
            f"input cardinality differs: clean_rows={len(mapped):,}, "
            f"distinct_titles={distinct_count:,}"
        )
    print(f"[stage] {len(mapped):,} clean rows; {distinct_count:,} distinct titles", flush=True)
    loaded, load_evidence = _load_model(args, cache_dir)
    print(
        f"[stage] pinned model loaded on {loaded.device} in "
        f"{load_evidence['model_load_seconds']:.2f}s",
        flush=True,
    )
    inference = finbert_innovation.infer_distinct_titles(
        mapped,
        loaded,
        batch_size=args.batch_size,
        max_length=args.max_length,
        progress_callback=_ProgressReporter(),
    )
    print("[stage] inference complete; validating join and aggregation", flush=True)
    headline_scores, join_audit = finbert_phase2.join_title_scores(mapped, inference.title_scores)
    ticker_days = finbert_phase2.aggregate_ticker_days(headline_scores)
    finbert_sector = finbert_phase2.sector_sentiment_index(ticker_days, calendar, universe)
    vader_sector = pd.read_csv(PROJECT_ROOT / PROTECTED_ARTIFACTS[2])
    coverage_audit = finbert_phase2.validate_coverage_reconciliation(vader_sector, finbert_sector)

    print("[stage] scoring VADER in memory for descriptive matched comparisons", flush=True)
    vader_cache = sentiment.score_distinct_titles(mapped)
    comparison_headlines = headline_scores.merge(
        vader_cache,
        on="text_raw",
        how="left",
        validate="many_to_one",
    )
    comparison, matched_sector = finbert_phase2.model_comparison_table(
        comparison_headlines, vader_sector, finbert_sector
    )
    review = finbert_phase2.manual_review_template(comparison_headlines)
    disagreements = finbert_phase2.disagreement_export(comparison_headlines)

    stage_root = Path(tempfile.mkdtemp(prefix="portfoyou-finbert-phase2-"))
    print(f"[stage] writing and validating staged outputs in {stage_root}", flush=True)
    try:
        for relative in PHASE2_OUTPUTS.values():
            (stage_root / relative).parent.mkdir(parents=True, exist_ok=True)
        inference.title_scores.to_csv(stage_root / PHASE2_OUTPUTS["title_scores"], index=False)
        finbert_sector.to_csv(stage_root / PHASE2_OUTPUTS["sector_index"], index=False)
        comparison.to_csv(stage_root / PHASE2_OUTPUTS["comparison"], index=False)
        disagreements.to_csv(stage_root / PHASE2_OUTPUTS["disagreements"], index=False)
        review.to_csv(stage_root / PHASE2_OUTPUTS["manual_review"], index=False)
        qa = _generate_figures(
            comparison_headlines,
            matched_sector,
            comparison,
            stage_root / "results/figures",
        )
        qa.to_csv(stage_root / PHASE2_OUTPUTS["figure_qa"], index=False)
        metadata: dict[str, object] = {
            "phase": "phase_2_full_corpus",
            "phase_status": "PASS",
            "model_name": finbert_innovation.MODEL_NAME,
            "pinned_revision": finbert_innovation.MODEL_REVISION,
            "resolved_revision": loaded.resolved_revision,
            "model_license": finbert_innovation.MODEL_LICENSE,
            "label_mapping": loaded.id2label,
            "score_definition": "probability_positive - probability_negative",
            "text_input": "unchanged text_raw",
            "clean_headline_rows": len(mapped),
            "distinct_title_count": distinct_count,
            "ticker_day_count": len(ticker_days),
            "sector_grid_rows": len(finbert_sector),
            "observed_sector_day_count": int(finbert_sector["has_observed_news"].sum()),
            "matched_sector_day_count": len(matched_sector),
            "representative_review_rows": int(
                review["sample_purpose"].eq("representative_evaluation").sum()
            ),
            "diagnostic_review_rows": int(
                review["sample_purpose"].eq("disagreement_enriched_diagnosis").sum()
            ),
            "manual_review_status": "pending student review",
            "claims_status": (
                "accuracy, superiority, and predictive claims pending student "
                "review or out of scope"
            ),
            "cache_policy": "external Hugging Face cache outside submission folder",
            "warnings": [
                "upstream BERT tokenizer deprecation warning (non-blocking)",
                "unauthenticated public-model request warning may occur "
                "(non-blocking; no secret added)",
            ],
            "join_validation": join_audit,
            "coverage_reconciliation": coverage_audit,
            "protected_artifact_hashes_before": hashes_before,
            **load_evidence,
            **finbert_innovation.package_versions(),
            **_inference_evidence(inference),
        }
        _write_json(stage_root / PHASE2_OUTPUTS["metadata"], metadata)
        _validate_staged_outputs(stage_root, metadata)
        manifest_files = {
            path.relative_to(stage_root).as_posix(): _sha256(path)
            for path in stage_root.rglob("*")
            if path.is_file()
        }
        manifest = {
            "phase": "phase_2_full_corpus",
            "completion_status": "PASS",
            "file_count": len(manifest_files),
            "files": manifest_files,
            "protected_artifact_hashes_before": hashes_before,
        }
        _write_json(stage_root / PHASE2_OUTPUTS["manifest"], manifest)
        promoted_hashes = _promote_stage(stage_root)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)

    hashes_after = protected_hashes()
    if hashes_after != hashes_before:
        changed = sorted(
            path for path in hashes_before if hashes_before[path] != hashes_after[path]
        )
        raise Phase2BuildError(f"Phase 2 changed protected artifacts: {changed}")
    metadata["protected_artifact_hashes_after"] = hashes_after
    metadata["full_phase2_wall_seconds"] = time.perf_counter() - run_started
    metadata["generated_file_count"] = len(promoted_hashes)
    # Refresh metadata and manifest last with post-run hash evidence.
    metadata_target = PROJECT_ROOT / PHASE2_OUTPUTS["metadata"]
    metadata_fd, metadata_name = tempfile.mkstemp(prefix="finbert-metadata-", suffix=".json")
    os.close(metadata_fd)
    temporary_metadata = Path(metadata_name)
    _write_json(temporary_metadata, metadata)
    os.replace(temporary_metadata, metadata_target)
    metadata_target.chmod(0o644)
    manifest_target = PROJECT_ROOT / PHASE2_OUTPUTS["manifest"]
    manifest["protected_artifact_hashes_after"] = hashes_after
    manifest["files"][PHASE2_OUTPUTS["metadata"].as_posix()] = _sha256(metadata_target)
    manifest_fd, manifest_name = tempfile.mkstemp(prefix="finbert-manifest-", suffix=".json")
    os.close(manifest_fd)
    temporary_manifest = Path(manifest_name)
    _write_json(temporary_manifest, manifest)
    os.replace(temporary_manifest, manifest_target)
    manifest_target.chmod(0o644)
    print("[stage] Phase 2 artifacts promoted; protected hashes unchanged", flush=True)
    return metadata


def run_postprocess_existing() -> dict[str, object]:
    """Atomically rebuild derived artifacts without repeating neural inference."""
    started = time.perf_counter()
    hashes_before = protected_hashes()
    title_path = PROJECT_ROOT / PHASE2_OUTPUTS["title_scores"]
    metadata_path = PROJECT_ROOT / PHASE2_OUTPUTS["metadata"]
    if not title_path.is_file() or not metadata_path.is_file():
        raise Phase2BuildError("validated FinBERT title scores or metadata are missing")
    title_scores = finbert_phase2.validate_title_score_cache(pd.read_csv(title_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(title_scores) != 105_334 or metadata.get("phase_status") != "PASS":
        raise Phase2BuildError("existing full-inference evidence is not valid")

    print("[stage] loading clean data for postprocessing refresh", flush=True)
    mapped, _, calendar, universe = _prepare_inputs()
    headline_scores, join_audit = finbert_phase2.join_title_scores(mapped, title_scores)
    ticker_days = finbert_phase2.aggregate_ticker_days(headline_scores)
    finbert_sector = finbert_phase2.sector_sentiment_index(ticker_days, calendar, universe)
    vader_sector = pd.read_csv(PROJECT_ROOT / PROTECTED_ARTIFACTS[2])
    coverage_audit = finbert_phase2.validate_coverage_reconciliation(vader_sector, finbert_sector)
    vader_cache = sentiment.score_distinct_titles(mapped)
    comparison_headlines = headline_scores.merge(
        vader_cache,
        on="text_raw",
        how="left",
        validate="many_to_one",
    )
    comparison, matched_sector = finbert_phase2.model_comparison_table(
        comparison_headlines, vader_sector, finbert_sector
    )
    review = finbert_phase2.manual_review_template(comparison_headlines)
    disagreements = finbert_phase2.disagreement_export(comparison_headlines)

    stage_root = Path(tempfile.mkdtemp(prefix="portfoyou-finbert-refresh-"))
    try:
        for relative in PHASE2_OUTPUTS.values():
            (stage_root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(title_path, stage_root / PHASE2_OUTPUTS["title_scores"])
        finbert_sector.to_csv(stage_root / PHASE2_OUTPUTS["sector_index"], index=False)
        comparison.to_csv(stage_root / PHASE2_OUTPUTS["comparison"], index=False)
        disagreements.to_csv(stage_root / PHASE2_OUTPUTS["disagreements"], index=False)
        review.to_csv(stage_root / PHASE2_OUTPUTS["manual_review"], index=False)
        qa = _generate_figures(
            comparison_headlines,
            matched_sector,
            comparison,
            stage_root / "results/figures",
        )
        qa.to_csv(stage_root / PHASE2_OUTPUTS["figure_qa"], index=False)
        metadata.update(
            {
                "join_validation": join_audit,
                "coverage_reconciliation": coverage_audit,
                "correlation_policy": (
                    "reported only for matched observed date-sector pairs; "
                    "headline-level correlations suppressed"
                ),
                "figure_visual_qa_status": "PASS after margin and overlap correction",
                "postprocessing_refresh_seconds": time.perf_counter() - started,
                "protected_artifact_hashes_before": hashes_before,
            }
        )
        _write_json(stage_root / PHASE2_OUTPUTS["metadata"], metadata)
        _validate_staged_outputs(stage_root, metadata)
        manifest_files = {
            path.relative_to(stage_root).as_posix(): _sha256(path)
            for path in stage_root.rglob("*")
            if path.is_file()
        }
        manifest = {
            "phase": "phase_2_full_corpus",
            "completion_status": "PASS",
            "file_count": len(manifest_files),
            "files": manifest_files,
            "protected_artifact_hashes_before": hashes_before,
        }
        _write_json(stage_root / PHASE2_OUTPUTS["manifest"], manifest)
        promoted_hashes = _promote_stage(stage_root)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)

    hashes_after = protected_hashes()
    if hashes_after != hashes_before:
        raise Phase2BuildError("postprocessing refresh changed a protected artifact")
    metadata["protected_artifact_hashes_after"] = hashes_after
    metadata["postprocessing_refresh_seconds"] = time.perf_counter() - started
    metadata["generated_file_count"] = len(promoted_hashes)
    metadata_fd, metadata_name = tempfile.mkstemp(
        prefix="finbert-refresh-metadata-", suffix=".json"
    )
    os.close(metadata_fd)
    temporary_metadata = Path(metadata_name)
    _write_json(temporary_metadata, metadata)
    os.replace(temporary_metadata, metadata_path)
    metadata_path.chmod(0o644)
    manifest["protected_artifact_hashes_after"] = hashes_after
    manifest["files"][PHASE2_OUTPUTS["metadata"].as_posix()] = _sha256(metadata_path)
    manifest_fd, manifest_name = tempfile.mkstemp(
        prefix="finbert-refresh-manifest-", suffix=".json"
    )
    os.close(manifest_fd)
    temporary_manifest = Path(manifest_name)
    _write_json(temporary_manifest, manifest)
    manifest_path = PROJECT_ROOT / PHASE2_OUTPUTS["manifest"]
    os.replace(temporary_manifest, manifest_path)
    manifest_path.chmod(0o644)
    print("[stage] postprocessing artifacts refreshed atomically", flush=True)
    return metadata


def refresh_integrity_metadata() -> dict[str, object]:
    """Refresh core-artifact hash evidence without changing FinBERT outputs.

    This is permitted only after validating that every existing Phase 2 output
    still matches its manifest. It records the current core-fund snapshot for a
    later, unrelated fund-family update without relabelling it as new inference.
    """
    metadata_path = PROJECT_ROOT / PHASE2_OUTPUTS["metadata"]
    manifest_path = PROJECT_ROOT / PHASE2_OUTPUTS["manifest"]
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise Phase2BuildError("FinBERT metadata or manifest is missing")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("phase_status") != "PASS" or manifest.get("completion_status") != "PASS":
        raise Phase2BuildError("existing FinBERT Phase 2 evidence is not valid")
    for relative, expected_hash in manifest.get("files", {}).items():
        if relative == PHASE2_OUTPUTS["metadata"].as_posix():
            continue
        path = PROJECT_ROOT / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            raise Phase2BuildError(f"existing Phase 2 artifact does not match manifest: {relative}")

    current_hashes = protected_hashes()
    metadata.update(
        {
            "protected_artifact_hashes_before": current_hashes,
            "protected_artifact_hashes_after": current_hashes,
            "core_artifact_integrity_refresh": {
                "scope": "hash evidence only; FinBERT title scores and derived outputs unchanged",
                "reason": "core fund-family artifact update after FinBERT Phase 2",
            },
        }
    )
    metadata_fd, metadata_name = tempfile.mkstemp(
        prefix="finbert-integrity-metadata-", suffix=".json"
    )
    os.close(metadata_fd)
    temporary_metadata = Path(metadata_name)
    _write_json(temporary_metadata, metadata)
    os.replace(temporary_metadata, metadata_path)
    metadata_path.chmod(0o644)

    manifest["protected_artifact_hashes_before"] = current_hashes
    manifest["protected_artifact_hashes_after"] = current_hashes
    manifest["files"][PHASE2_OUTPUTS["metadata"].as_posix()] = _sha256(metadata_path)
    manifest_fd, manifest_name = tempfile.mkstemp(
        prefix="finbert-integrity-manifest-", suffix=".json"
    )
    os.close(manifest_fd)
    temporary_manifest = Path(manifest_name)
    _write_json(temporary_manifest, manifest)
    os.replace(temporary_manifest, manifest_path)
    manifest_path.chmod(0o644)
    print("[stage] FinBERT integrity metadata refreshed; model outputs unchanged", flush=True)
    return metadata


def main() -> int:
    args = _parser().parse_args()
    if args.pilot_only:
        evidence = run_pilot(args)
    elif args.full_corpus:
        evidence = run_full_corpus(args)
    elif args.refresh_integrity_metadata:
        evidence = refresh_integrity_metadata()
    else:
        evidence = run_postprocess_existing()
    print(json.dumps(evidence, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
