"""Build student-labelled VADER--FinBERT validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
_MPL_CACHE = Path(tempfile.gettempdir()) / "portfoyou-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src import finbert_manual_validation  # noqa: E402

from fintools.figures import FigureContext, export_figure_bundle, figure_style  # noqa: E402

PROTECTED_ARTIFACTS = (
    Path("results/data/fund_returns.csv"),
    Path("results/data/fund_weights.csv"),
    Path("results/data/sector_sentiment_index.csv"),
    Path("results/tables/performance_metrics.csv"),
)
BLIND_REVIEW = Path("results/tables/sentiment_manual_review_blind.csv")
AUDIT_TEMPLATE = Path("results/tables/sentiment_manual_review_template.csv")
COMPLETED_REVIEW = Path("results/tables/sentiment_manual_review_completed.csv")
OUTPUTS = {
    "metrics": Path("results/tables/sentiment_manual_validation_metrics.csv"),
    "confusion": Path("results/tables/sentiment_manual_validation_confusion.csv"),
    "joined": Path("results/tables/sentiment_manual_validation_joined.csv"),
    "errors": Path("results/tables/sentiment_manual_error_analysis.csv"),
    "figure": Path("results/figures/sentiment_manual_validation_comparison.png"),
    "figure_caption": Path(
        "results/figures/sentiment_manual_validation_comparison.caption.md"
    ),
    "metadata": Path("results/tables/sentiment_manual_validation_metadata.json"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in PROTECTED_ARTIFACTS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"protected artifact is missing: {relative}")
        hashes[relative.as_posix()] = _sha256(path)
    return hashes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate completed blind labels and build Phase 4 artifacts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / COMPLETED_REVIEW,
        help="Completed blinded review CSV; defaults to the project copy.",
    )
    parser.add_argument(
        "--single-student-reviewer-attested",
        action="store_true",
        help=(
            "Accept blank repeated reviewer cells after the sole student explicitly "
            "confirms authorship of every manual label."
        ),
    )
    return parser


def _metric_value(metrics: pd.DataFrame, model: str, metric: str) -> float:
    selected = metrics.loc[metrics["model"].eq(model) & metrics["metric"].eq(metric), "value"]
    if len(selected) != 1:
        raise RuntimeError(f"expected one {model} {metric} row")
    return float(selected.iloc[0])


def _validation_figure(metrics: pd.DataFrame, output_dir: Path) -> None:
    metric_specs = (
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro F1"),
    )
    models = ("VADER", "FinBERT")
    colors = {"VADER": "#24466F", "FinBERT": "#B23A48"}
    x = np.arange(len(metric_specs), dtype=float)
    width = 0.34
    with figure_style(profile="word_a4", style="fins"):
        fig, ax = plt.subplots(figsize=(6.27, 3.95), layout=None)
        fig.set_layout_engine(None)
        for model_index, model in enumerate(models):
            positions = x + (model_index - 0.5) * width
            values = [_metric_value(metrics, model, metric) for metric, _ in metric_specs]
            ax.bar(
                positions,
                values,
                width=width,
                color=colors[model],
                label=model,
                alpha=0.95,
            )
            accuracy = metrics.loc[
                metrics["model"].eq(model) & metrics["metric"].eq("accuracy")
            ].iloc[0]
            low = float(accuracy["ci_low"])
            high = float(accuracy["ci_high"])
            ax.errorbar(
                positions[0],
                values[0],
                yerr=[[values[0] - low], [high - values[0]]],
                fmt="none",
                ecolor="#273444",
                elinewidth=1.0,
                capsize=3,
            )
            for metric_index, (position, value) in enumerate(
                zip(positions, values, strict=True)
            ):
                label_y = high + 0.025 if metric_index == 0 else value + 0.022
                ax.text(
                    position,
                    label_y,
                    f"{value:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_title("Student-labelled sentiment model validation", loc="left")
        ax.set_ylabel("Weighted score")
        ax.set_xticks(x, [label for _, label in metric_specs])
        ax.set_ylim(0, 1.04)
        ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper right")
        ax.text(
            0.01,
            0.97,
            "Sample: 100 representative headlines | Source: Student blind review",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#4D5968",
        )
        context = FigureContext(
            title="Student-labelled sentiment model validation",
            note=(
                "Metrics use only the 100-headline representative sample and its "
                "sampling weights. Error bars show approximate 95% Wilson intervals "
                "for weighted accuracy using Kish effective sample size."
            ),
            source="Student blind review; generated by scripts/run_finbert_manual_validation.py",
            sample="100 representative financial headlines",
            units="Weighted classification score",
        )
        exported = export_figure_bundle(
            fig,
            output_dir,
            "sentiment_manual_validation_comparison",
            context=context,
            formats=("png",),
            dpi=300,
        )
        plt.close(fig)
    if not exported["png"].is_file() or exported["png"].stat().st_size == 0:
        raise RuntimeError("manual-validation figure export failed")


def main() -> int:
    args = _parser().parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"completed blind review is missing: {input_path}")
    hashes_before = _protected_hashes()
    submitted = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    blind = pd.read_csv(PROJECT_ROOT / BLIND_REVIEW, dtype=str, keep_default_na=False)
    audit = pd.read_csv(PROJECT_ROOT / AUDIT_TEMPLATE, dtype=str, keep_default_na=False)
    imported = finbert_manual_validation.import_student_review(
        submitted,
        blind,
        single_student_reviewer_attested=args.single_student_reviewer_attested,
    )
    joined = finbert_manual_validation.join_review_to_audit(imported.completed, audit)
    results = finbert_manual_validation.evaluate_manual_review(joined)

    with tempfile.TemporaryDirectory(prefix="finbert-phase4-") as temporary:
        stage = Path(temporary)
        tables = stage / "tables"
        figures = stage / "figures"
        tables.mkdir(parents=True)
        figures.mkdir(parents=True)
        imported.completed.to_csv(tables / COMPLETED_REVIEW.name, index=False)
        results.metrics.to_csv(tables / OUTPUTS["metrics"].name, index=False)
        results.confusion.to_csv(tables / OUTPUTS["confusion"].name, index=False)
        results.joined.to_csv(tables / OUTPUTS["joined"].name, index=False)
        results.error_analysis.to_csv(tables / OUTPUTS["errors"].name, index=False)
        _validation_figure(results.metrics, figures)

        metadata: dict[str, object] = {
            "phase": "phase_4_student_labelled_validation",
            "phase_status": "PASS",
            "review_rows": len(imported.completed),
            "representative_rows": int(
                results.joined["sample_purpose"].eq(
                    finbert_manual_validation.REPRESENTATIVE_PURPOSE
                ).sum()
            ),
            "diagnostic_rows": int(
                results.joined["sample_purpose"].eq(
                    finbert_manual_validation.DIAGNOSTIC_PURPOSE
                ).sum()
            ),
            "label_counts": imported.completed["human_label"].value_counts().to_dict(),
            "confidence_counts": imported.completed["human_confidence"].value_counts().to_dict(),
            "student_comment_rows": int(imported.completed["reviewer_comment"].ne("").sum()),
            "restored_text_count": imported.restored_text_count,
            "normalised_confidence_whitespace_count": (
                imported.normalised_confidence_whitespace_count
            ),
            "text_repair_policy": (
                "Canonical text_raw restored by review_id only; student fields "
                "unchanged."
            ),
            "reviewer_attestation": imported.reviewer_attestation,
            "source_input_sha256": _sha256(input_path),
            "completed_review_sha256": _sha256(tables / COMPLETED_REVIEW.name),
            "paired_comparison": results.paired_summary,
            "claims_status": (
                "Manual classification evidence only; no return predictability or "
                "investment-superiority claim."
            ),
            "confidence_interval_policy": (
                "Approximate 95% Wilson interval for weighted accuracy using Kish "
                "effective sample size; other metric intervals not reported."
            ),
            "mcnemar_policy": (
                "Two-sided exact unweighted McNemar test on the same 100 representative "
                "headlines; small-sample and sampling-design limitations apply."
            ),
            "diagnostic_policy": (
                "Fifty disagreement-enriched rows excluded from performance metrics; "
                "categories assigned only when supported by a student comment."
            ),
            "protected_artifact_hashes_before": hashes_before,
        }
        (tables / OUTPUTS["metadata"].name).write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        promotions = {
            tables / COMPLETED_REVIEW.name: PROJECT_ROOT / COMPLETED_REVIEW,
            tables / OUTPUTS["metrics"].name: PROJECT_ROOT / OUTPUTS["metrics"],
            tables / OUTPUTS["confusion"].name: PROJECT_ROOT / OUTPUTS["confusion"],
            tables / OUTPUTS["joined"].name: PROJECT_ROOT / OUTPUTS["joined"],
            tables / OUTPUTS["errors"].name: PROJECT_ROOT / OUTPUTS["errors"],
            tables / OUTPUTS["metadata"].name: PROJECT_ROOT / OUTPUTS["metadata"],
            figures / OUTPUTS["figure"].name: PROJECT_ROOT / OUTPUTS["figure"],
            figures / OUTPUTS["figure_caption"].name: PROJECT_ROOT / OUTPUTS["figure_caption"],
        }
        for source, destination in promotions.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)

    hashes_after = _protected_hashes()
    if hashes_after != hashes_before:
        raise RuntimeError("Phase 4 changed a protected core artifact")
    metadata_path = PROJECT_ROOT / OUTPUTS["metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["protected_artifact_hashes_after"] = hashes_after
    metadata["generated_artifact_hashes"] = {
        path.as_posix(): _sha256(PROJECT_ROOT / path)
        for path in (COMPLETED_REVIEW, *OUTPUTS.values())
        if path != OUTPUTS["metadata"]
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
