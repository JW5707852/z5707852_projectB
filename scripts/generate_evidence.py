"""Generate the locked core Part B tables and Word/A4-ready figures.

Run after ``scripts/run_part_b.py``. The workflow reads only precomputed project
artifacts, writes report evidence under ``results/tables`` and
``results/figures``, and fails if source reconciliation or figure QA fails.
"""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Matplotlib caches outside the submission tree. This avoids personal paths and
# keeps a clean checkout free of generated cache folders.
_matplotlib_cache = Path(tempfile.gettempdir()) / "portfoyou-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
from PIL import Image  # noqa: E402
from src import evidence  # noqa: E402

from fintools.figures import (  # noqa: E402
    FigureContext,
    export_figure_bundle,
    figure_style,
    validate_axes_labels,
    validate_display_labels,
    validate_figure_context,
    validate_horizontal_grid,
    validate_image_not_blank,
    validate_markers_within_axes,
    validate_no_text_overlap,
    validate_no_tick_label_overlap,
    validate_series_identification,
    validate_titles_within_canvas,
    validate_unique_series_colors,
    validate_word_readability,
)

SOURCE_PATHS = {
    "fund_returns": PROJECT_ROOT / "results/data/fund_returns.csv",
    "fund_weights": PROJECT_ROOT / "results/data/fund_weights.csv",
    "sector_sentiment": PROJECT_ROOT / "results/data/sector_sentiment_index.csv",
    "performance": PROJECT_ROOT / "results/tables/performance_metrics.csv",
    "fusion": PROJECT_ROOT / "results/tables/fusion_comparison.csv",
}
TABLE_FILENAMES = {
    "performance": "performance_table_core.csv",
    "fusion": "fusion_before_after_table.csv",
    "manifest": "evidence_manifest.csv",
    "qa": "figure_qa.csv",
}
FIGURE_STEMS = (
    "growth_of_1_comparison",
    "drawdown_equity_sentiment_tilt",
    "combined_weights_over_time",
    "return_risk_comparison",
    "sector_sentiment_time_series",
    "fusion_before_after",
)

FUND_COLORS = {
    "combined_equal_weight": "#1F3A5F",
    "combined_min_variance": "#B23A48",
    "combined_active_sector_allocation": "#D56F3E",
    "combined_growth_sector_allocation": "#6B5B95",
    "combined_aggressive_sector_allocation": "#8C657E",
    "equity_equal_weight": "#007C89",
    "equity_sentiment_tilt": "#C99700",
    "crypto_equal_weight": "#00A6A6",
    "crypto_min_variance": "#E76F51",
}
RISK_PLOT_LABELS = {
    "combined_equal_weight": "Combined 1/N",
    "combined_min_variance": "Minimum variance",
    "combined_active_sector_allocation": "Active sector",
    "combined_growth_sector_allocation": "Balanced growth",
    "combined_aggressive_sector_allocation": "Aggressive sector & crypto",
    "equity_equal_weight": "Equity 1/N",
    "equity_sentiment_tilt": "Equity sentiment",
    "crypto_equal_weight": "Crypto 1/N",
    "crypto_min_variance": "Crypto minimum variance",
}
SECTOR_COLORS = {
    "Comm": "#1F3A5F",
    "Consumer": "#B23A48",
    "Energy": "#2E7D32",
    "Financials": "#C99700",
    "Healthcare": "#007C89",
    "Industrials": "#6B5B95",
    "Materials": "#D56F3E",
    "RealEstate": "#4C78A8",
    "Tech": "#990F3D",
    "Utilities": "#0D7680",
}


class EvidenceGenerationError(RuntimeError):
    """Raised when an exhibit cannot be generated and verified."""


@dataclass(frozen=True)
class EvidenceBuild:
    """Paths and row counts from one deterministic evidence build."""

    tables: dict[str, Path]
    figures: dict[str, dict[str, Path]]
    table_rows: dict[str, int]
    figure_qa: pd.DataFrame


def _relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_sources() -> dict[str, pd.DataFrame]:
    missing = [str(path) for path in SOURCE_PATHS.values() if not path.exists()]
    if missing:
        raise EvidenceGenerationError(f"missing source artifacts: {missing}")
    return {
        "fund_returns": pd.read_csv(
            SOURCE_PATHS["fund_returns"], parse_dates=["date", "decision_date"]
        ),
        "fund_weights": pd.read_csv(
            SOURCE_PATHS["fund_weights"],
            parse_dates=["date", "decision_date", "first_holding_date"],
        ),
        "sector_sentiment": pd.read_csv(
            SOURCE_PATHS["sector_sentiment"],
            parse_dates=["date", "tradable_signal_source_date"],
        ),
        "performance": pd.read_csv(
            SOURCE_PATHS["performance"],
            parse_dates=["sample_start_date", "sample_end_date"],
        ),
        "fusion": pd.read_csv(
            SOURCE_PATHS["fusion"],
            parse_dates=["sample_start_date", "sample_end_date"],
        ),
    }


def _style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.margins(x=0)


def _format_year_axis(ax: plt.Axes) -> None:
    ax.margins(x=0)
    lower_number, upper_number = ax.get_xlim()
    lower = pd.Timestamp(mdates.num2date(lower_number)).tz_localize(None).normalize()
    upper = pd.Timestamp(mdates.num2date(upper_number)).tz_localize(None).normalize()
    first_tick = pd.Timestamp(year=lower.year, month=1, day=1)
    if first_tick < lower and (lower - first_tick).days > 45:
        first_tick = pd.Timestamp(year=lower.year + 1, month=1, day=1)
    elif first_tick < lower:
        ax.set_xlim(first_tick, upper)
    ticks = pd.date_range(first_tick, upper, freq="YS")
    if ticks.empty:
        ticks = pd.DatetimeIndex([lower])
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _add_metadata(
    fig: plt.Figure,
    *,
    sample: str,
    units: str,
    definition: str,
    source: str,
) -> None:
    color = "#4B5563"
    fig.text(0.02, 0.090, f"Sample: {sample} | Units: {units}", fontsize=8, color=color)
    fig.text(0.02, 0.055, f"Definition: {definition}", fontsize=8, color=color)
    fig.text(0.02, 0.020, f"Source: {source}", fontsize=8, color=color)


def _outside_canvas_issues(fig: plt.Figure) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    issues: list[str] = []
    tick_labels = {
        label
        for ax in fig.axes
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]
    }
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        # Matplotlib retains tick artists just outside explicit axis limits. They
        # are clipped from the rendered axes and should not be mistaken for
        # visible canvas overflow.
        if artist in tick_labels:
            continue
        box = artist.get_window_extent(renderer)
        if (
            box.x0 < canvas.x0 - 2
            or box.x1 > canvas.x1 + 2
            or box.y0 < canvas.y0 - 2
            or box.y1 > canvas.y1 + 2
        ):
            issues.append(f"text_outside_canvas:{artist.get_text()[:40]}")
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        box = legend.get_window_extent(renderer)
        if (
            box.x0 < canvas.x0 - 2
            or box.x1 > canvas.x1 + 2
            or box.y0 < canvas.y0 - 2
            or box.y1 > canvas.y1 + 2
        ):
            issues.append("legend_outside_canvas")
    return issues


def _figure_issues(
    fig: plt.Figure,
    axes: list[plt.Axes],
    context: FigureContext,
    *,
    require_individual_axis_labels: bool = True,
) -> list[str]:
    fig.canvas.draw()
    issues = [issue.code for issue in validate_figure_context(context)]
    issues.extend(issue.code for issue in validate_titles_within_canvas(fig))
    issues.extend(
        issue.code
        for issue in validate_word_readability(
            fig, width_inches=6.27, min_font_size=8.0
        )
    )
    for ax in axes:
        if require_individual_axis_labels:
            issues.extend(issue.code for issue in validate_axes_labels(ax))
        issues.extend(issue.code for issue in validate_display_labels(ax))
        issues.extend(issue.code for issue in validate_horizontal_grid(ax))
        issues.extend(issue.code for issue in validate_no_tick_label_overlap(ax, axis="x"))
        issues.extend(issue.code for issue in validate_no_tick_label_overlap(ax, axis="y"))
        if not fig.legends:
            issues.extend(issue.code for issue in validate_series_identification(ax))
        issues.extend(issue.code for issue in validate_unique_series_colors(ax))
        issues.extend(issue.code for issue in validate_markers_within_axes(ax))
        issues.extend(issue.code for issue in validate_no_text_overlap(ax))
    issues.extend(_outside_canvas_issues(fig))
    return sorted(set(issues))


def _export_figure(
    fig: plt.Figure,
    axes: list[plt.Axes],
    *,
    output_dir: Path,
    stem: str,
    context: FigureContext,
    source_rows: int,
    require_individual_axis_labels: bool = True,
) -> tuple[dict[str, Path], dict[str, object]]:
    issues = _figure_issues(
        fig,
        axes,
        context,
        require_individual_axis_labels=require_individual_axis_labels,
    )
    if issues:
        raise EvidenceGenerationError(f"{stem} failed in-memory QA: {issues}")
    paths = export_figure_bundle(
        fig,
        output_dir,
        stem,
        context=context,
        formats=("png", "pdf"),
        dpi=300,
    )
    image_issues = [issue.code for issue in validate_image_not_blank(paths["png"])]
    with Image.open(paths["png"]) as image:
        width_px, height_px = image.size
    if image_issues:
        raise EvidenceGenerationError(f"{stem} failed exported-image QA: {image_issues}")
    qa = {
        "figure": stem,
        "png_path": _relative(paths["png"]),
        "pdf_path": _relative(paths["pdf"]),
        "caption_path": _relative(paths["caption"]),
        "source_rows": int(source_rows),
        "width_inches": float(fig.get_size_inches()[0]),
        "height_inches": float(fig.get_size_inches()[1]),
        "png_width_pixels": int(width_px),
        "png_height_pixels": int(height_px),
        "minimum_font_points": min(
            text.get_fontsize()
            for text in fig.findobj(match=Text)
            if text.get_visible() and text.get_text().strip()
        ),
        "blank_image_issues": 0,
        "layout_issue_count": 0,
        "qa_status": "PASS",
    }
    plt.close(fig)
    return paths, qa


def _growth_figure(paths: pd.DataFrame) -> tuple[plt.Figure, list[plt.Axes], FigureContext]:
    sample = f"{paths['date'].min():%Y-%m-%d} to {paths['date'].max():%Y-%m-%d}"
    with figure_style(profile="word_a4", style="fins"):
        fig, ax = plt.subplots(figsize=(6.27, 4.50), constrained_layout=False)
        for fund in evidence.CORE_FUNDS:
            group = paths.loc[paths["fund"].eq(fund)]
            ax.plot(
                group["date"],
                group["growth_of_1"],
                label=evidence.FUND_LABELS[fund],
                color=FUND_COLORS[fund],
                linewidth=1.8,
                alpha=0.92,
            )
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1 (US dollars)")
        ax.set_yscale("log")
        wealth_values = paths["growth_of_1"].to_numpy(dtype=float)
        lower = float(wealth_values.min())
        upper = float(wealth_values.max())
        candidate_ticks = np.array([0.75, 1.00, 1.25, 1.50, 2.00])
        visible_ticks = candidate_ticks[
            (candidate_ticks >= lower * 0.95) & (candidate_ticks <= upper * 1.05)
        ]
        ax.set_ylim(lower * 0.97, upper * 1.03)
        ax.yaxis.set_major_locator(FixedLocator(visible_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:.2f}"))
        ax.yaxis.set_minor_locator(NullLocator())
        _format_year_axis(ax)
        _style_axis(ax)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.90))
        fig.suptitle("Core fund growth of $1", x=0.10, ha="left", y=0.985)
        fig.subplots_adjust(left=0.16, right=0.98, top=0.74, bottom=0.23)
        source = (
            "results/data/fund_returns.csv; common-date rebasing; generated by "
            "scripts/generate_evidence.py"
        )
        definition = (
            "Cumulative wealth is the product of one plus each daily fund return; "
            "log scale."
        )
        _add_metadata(
            fig,
            sample=sample,
            units="US dollars per $1 invested",
            definition=definition,
            source=source,
        )
    context = FigureContext(
        title="Core fund growth of $1",
        note=definition,
        source=source,
        sample=sample,
        units="US dollars per $1 invested",
    )
    return fig, [ax], context


def _drawdown_figure(
    paths: pd.DataFrame,
) -> tuple[plt.Figure, list[plt.Axes], FigureContext]:
    selected = paths.loc[paths["fund"].eq("equity_sentiment_tilt")]
    sample = f"{selected['date'].min():%Y-%m-%d} to {selected['date'].max():%Y-%m-%d}"
    with figure_style(profile="word_a4", style="fins"):
        fig, ax = plt.subplots(figsize=(6.27, 3.75), constrained_layout=False)
        ax.fill_between(
            selected["date"],
            100.0 * selected["drawdown"],
            0.0,
            color=FUND_COLORS["equity_sentiment_tilt"],
            alpha=0.28,
        )
        ax.plot(
            selected["date"],
            100.0 * selected["drawdown"],
            color=FUND_COLORS["equity_sentiment_tilt"],
            linewidth=1.6,
        )
        ax.axhline(0.0, color="#4B5563", linewidth=0.8)
        ax.set_title("Equity sentiment tilt drawdown", loc="left", pad=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        _format_year_axis(ax)
        _style_axis(ax)
        fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.27)
        source = (
            "results/data/fund_returns.csv; generated by "
            "scripts/generate_evidence.py"
        )
        definition = "Drawdown equals cumulative wealth divided by its running peak, minus one."
        _add_metadata(
            fig,
            sample=sample,
            units="Percent below the prior wealth peak",
            definition=definition,
            source=source,
        )
    context = FigureContext(
        title="Equity sentiment tilt drawdown",
        note=definition,
        source=source,
        sample=sample,
        units="Percent below the prior wealth peak",
    )
    return fig, [ax], context


def _weight_figure(
    history: pd.DataFrame,
) -> tuple[plt.Figure, list[plt.Axes], FigureContext]:
    sample = (
        f"{history['decision_date'].min():%Y-%m-%d} to "
        f"{history['decision_date'].max():%Y-%m-%d}"
    )
    categories = list(history["display_holding"].cat.categories)
    colors = [
        "#1F3A5F",
        "#B23A48",
        "#2E7D32",
        "#C99700",
        "#007C89",
        "#6B5B95",
        "#8A8F98",
    ]
    with figure_style(profile="word_a4", style="fins"):
        fig, axes = plt.subplots(
            len(evidence.COMBINED_FUNDS),
            1,
            figsize=(6.27, 8.25),
            constrained_layout=False,
        )
        axes = np.atleast_1d(axes)
        for ax, fund in zip(axes, evidence.COMBINED_FUNDS, strict=True):
            group = history.loc[history["fund"].eq(fund)]
            wide = group.pivot(
                index="decision_date",
                columns="display_holding",
                values="target_weight_pct",
            ).sort_index()[categories]
            ax.stackplot(
                wide.index,
                *[wide[category] for category in categories],
                labels=categories,
                colors=colors,
                alpha=0.88,
            )
            ax.set_title(evidence.FUND_LABELS[fund], loc="left", pad=6)
            ax.set_xlabel("Decision date")
            ax.set_ylabel("Target weight (%)")
            ax.set_ylim(0.0, 100.0)
            _format_year_axis(ax)
            _style_axis(ax)
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].set_xlabel("")
        fig.legend(handles, labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 0.94))
        fig.suptitle("Combined-fund target weights", x=0.09, ha="left", y=0.99)
        fig.subplots_adjust(left=0.14, right=0.98, top=0.83, bottom=0.16, hspace=0.60)
        source = (
            "results/data/fund_weights.csv; generated by "
            "scripts/generate_evidence.py"
        )
        definition = (
            "Six peak-weight tickers are shown; all remaining assets are grouped "
            "and weights sum to 100%."
        )
        _add_metadata(
            fig,
            sample=sample,
            units="Percent of target portfolio weight",
            definition=definition,
            source=source,
        )
    context = FigureContext(
        title="Combined-fund target weights",
        note=definition,
        source=source,
        sample=sample,
        units="Percent of target portfolio weight",
    )
    return fig, list(axes), context


def _return_risk_figure(
    performance: pd.DataFrame,
) -> tuple[plt.Figure, list[plt.Axes], FigureContext]:
    sample = (
        f"{performance['sample_start_date'].min():%Y-%m-%d} to "
        f"{performance['sample_end_date'].max():%Y-%m-%d}"
    )
    with figure_style(profile="word_a4", style="fins"):
        fig, ax = plt.subplots(figsize=(6.27, 4.20), constrained_layout=False)
        for row in performance.itertuples(index=False):
            x = float(row.annualised_volatility_pct)
            y = float(row.geometric_annual_return_pct)
            ax.scatter(
                x,
                y,
                s=72,
                label=(
                    f"{RISK_PLOT_LABELS[row.fund]} "
                    f"(Sharpe {row.sharpe_ratio:.3g})"
                ),
                color=FUND_COLORS[row.fund],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
        x_values = performance["annualised_volatility_pct"].to_numpy(dtype=float)
        y_values = performance["geometric_annual_return_pct"].to_numpy(dtype=float)
        ax.set_xlim(x_values.min() - 2.0, x_values.max() + 2.0)
        ax.set_ylim(y_values.min() - 2.0, y_values.max() + 2.0)
        ax.set_title("Core fund return and risk", loc="left", pad=10)
        ax.set_xlabel("Annualised volatility (%)")
        ax.set_ylabel("Geometric annual return (%)")
        _style_axis(ax)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncols=2,
            fontsize=8,
            frameon=False,
            handletextpad=0.4,
            columnspacing=0.9,
        )
        fig.subplots_adjust(left=0.14, right=0.96, top=0.73, bottom=0.25)
        source = (
            "results/tables/performance_metrics.csv; generated by "
            "scripts/generate_evidence.py"
        )
        definition = (
            "Return is geometric at 252 periods; risk is sample daily volatility "
            "times sqrt(252)."
        )
        _add_metadata(
            fig,
            sample=sample,
            units="Annual percent; Sharpe ratio shown beside each fund",
            definition=definition,
            source=source,
        )
    context = FigureContext(
        title="Core fund return and risk",
        note=definition,
        source=source,
        sample=sample,
        units="Annual percent; Sharpe ratio shown beside each fund",
    )
    return fig, [ax], context


def _sentiment_figure(
    sentiment: pd.DataFrame,
) -> tuple[plt.Figure, list[plt.Axes], FigureContext]:
    sample = f"{sentiment['date'].min():%Y-%m-%d} to {sentiment['date'].max():%Y-%m-%d}"
    observed = sentiment["raw_sector_compound"].dropna().to_numpy(dtype=float)
    limit = min(1.0, max(0.25, float(np.ceil(np.max(np.abs(observed)) * 10.0) / 10.0)))
    with figure_style(profile="word_a4", style="fins"):
        fig, axes_array = plt.subplots(
            5, 2, figsize=(6.27, 7.35), sharex=True, sharey=True, constrained_layout=False
        )
        axes = list(axes_array.ravel())
        for ax, sector in zip(axes, evidence.EXPECTED_SECTORS, strict=True):
            group = sentiment.loc[sentiment["sector"].eq(sector)]
            ax.plot(
                group["date"],
                group["raw_sector_compound"],
                color=SECTOR_COLORS[sector],
                linewidth=0.75,
                alpha=0.78,
            )
            ax.axhline(0.0, color="#8A8F98", linewidth=0.6)
            ax.set_title(sector.replace("RealEstate", "Real estate"), loc="left", pad=3)
            ax.set_ylim(-limit, limit)
            _format_year_axis(ax)
            _style_axis(ax)
        fig.supxlabel("Date", y=0.145)
        fig.supylabel("VADER compound score (-1 to 1)", x=0.025)
        fig.suptitle("Ten-sector daily sentiment index", x=0.09, ha="left", y=0.985)
        fig.subplots_adjust(
            left=0.13,
            right=0.98,
            top=0.94,
            bottom=0.20,
            hspace=0.43,
            wspace=0.20,
        )
        source = (
            "results/data/sector_sentiment_index.csv; generated by "
            "scripts/generate_evidence.py"
        )
        definition = (
            "Daily VADER headline scores are averaged to ticker-day, then equally "
            "across observed tickers."
        )
        _add_metadata(
            fig,
            sample=sample,
            units="VADER score; gaps indicate no sector news",
            definition=definition,
            source=source,
        )
    context = FigureContext(
        title="Ten-sector daily sentiment index",
        note=definition,
        source=source,
        sample=sample,
        units="VADER compound score; missing no-news sector-days remain gaps",
    )
    return fig, axes, context


def _fusion_figure(
    paths: pd.DataFrame,
) -> tuple[plt.Figure, list[plt.Axes], FigureContext]:
    selected = paths.loc[paths["fund"].isin(evidence.FUSION_FUNDS)]
    sample = f"{selected['date'].min():%Y-%m-%d} to {selected['date'].max():%Y-%m-%d}"
    with figure_style(profile="word_a4", style="fins"):
        fig, axes = plt.subplots(2, 1, figsize=(6.27, 5.05), constrained_layout=False)
        for fund in evidence.FUSION_FUNDS:
            group = selected.loc[selected["fund"].eq(fund)]
            label = evidence.FUND_LABELS[fund]
            color = FUND_COLORS[fund]
            axes[0].plot(group["date"], group["growth_of_1"], label=label, color=color)
            axes[1].plot(
                group["date"],
                100.0 * group["drawdown"],
                label=label,
                color=color,
            )
        axes[0].set_title("Growth of $1", loc="left", pad=6)
        axes[0].set_ylabel("Wealth (US dollars)")
        axes[0].set_xlabel("Date")
        axes[1].set_title("Drawdown", loc="left", pad=6)
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].set_xlabel("Date")
        for ax in axes:
            _format_year_axis(ax)
            _style_axis(ax)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.945))
        fig.suptitle("Locked sentiment fusion: before versus after", x=0.09, ha="left", y=0.985)
        fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.22, hspace=0.55)
        context_source = (
            "results/data/fund_returns.csv and results/tables/fusion_comparison.csv; "
            "generated by scripts/generate_evidence.py"
        )
        source = (
            "results/data/fund_returns.csv; generated by "
            "scripts/generate_evidence.py"
        )
        definition = (
            "Matched base and one-day-lag sentiment-tilt wealth; drawdown is "
            "wealth/running peak-1."
        )
        _add_metadata(
            fig,
            sample=sample,
            units="US dollars and percent; 0 bps transaction costs",
            definition=definition,
            source=source,
        )
    context = FigureContext(
        title="Locked sentiment fusion: before versus after",
        note=definition,
        source=context_source,
        sample=sample,
        units="US dollars and percent; 0 bps transaction costs",
    )
    return fig, list(axes), context


def _manifest(
    performance: pd.DataFrame,
    fusion: pd.DataFrame,
    figure_paths: dict[str, dict[str, Path]],
    figure_contexts: dict[str, FigureContext],
) -> pd.DataFrame:
    performance_sample = (
        f"{performance['sample_start_date'].min():%Y-%m-%d} to "
        f"{performance['sample_end_date'].max():%Y-%m-%d}"
    )
    fusion_sample = (
        f"{fusion['sample_start_date'].min():%Y-%m-%d} to "
        f"{fusion['sample_end_date'].max():%Y-%m-%d}"
    )
    table_rows = [
        {
            "exhibit_id": "performance_table_core",
            "exhibit_type": "table",
            "title": "Core fund performance across families and methods",
            "primary_output": "results/tables/performance_table_core.csv",
            "companion_outputs": "",
            "sample_period": performance_sample,
            "units": "Percent, ratio, and US dollars as labelled",
            "source_artifacts": "results/tables/performance_metrics.csv",
            "generation_path": "scripts/generate_evidence.py",
            "calculation_definition": performance["calculation_definition"].iloc[0],
        },
        {
            "exhibit_id": "fusion_before_after_table",
            "exhibit_type": "table",
            "title": "Locked sentiment fusion before versus after",
            "primary_output": "results/tables/fusion_before_after_table.csv",
            "companion_outputs": "",
            "sample_period": fusion_sample,
            "units": "Percent, percentage points, ratio, and turnover as labelled",
            "source_artifacts": "results/tables/fusion_comparison.csv",
            "generation_path": "scripts/generate_evidence.py",
            "calculation_definition": fusion["calculation_definition"].iloc[0],
        },
    ]
    figure_metadata = {
        "growth_of_1_comparison": (
            "Core fund growth of $1",
            "US dollars per $1 invested",
            "results/data/fund_returns.csv",
            "Cumulative product of one plus daily fund return.",
        ),
        "drawdown_equity_sentiment_tilt": (
            "Equity sentiment tilt drawdown",
            "Percent below prior wealth peak",
            "results/data/fund_returns.csv",
            "Cumulative wealth divided by its running peak, minus one.",
        ),
        "combined_weights_over_time": (
            "Combined-fund target weights",
            "Percent of target portfolio weight",
            "results/data/fund_weights.csv",
            "Six peak-weight tickers are shown; all remaining assets are grouped.",
        ),
        "return_risk_comparison": (
            "Core fund return and risk",
            "Annual percent and Sharpe ratio",
            "results/tables/performance_metrics.csv",
            "Geometric annual return versus annualised sample volatility at 252 periods.",
        ),
        "sector_sentiment_time_series": (
            "Ten-sector daily sentiment index",
            "VADER compound score",
            "results/data/sector_sentiment_index.csv",
            "Ticker-day mean followed by equal weighting of observed tickers in each sector.",
        ),
        "fusion_before_after": (
            "Locked sentiment fusion before versus after",
            "US dollars and percent",
            "results/data/fund_returns.csv;results/tables/fusion_comparison.csv",
            "Matched base and one-day-lag tilt wealth and drawdown at 0 bps costs.",
        ),
    }
    figure_rows = []
    for stem in FIGURE_STEMS:
        title, units, sources, definition = figure_metadata[stem]
        paths = figure_paths[stem]
        context = figure_contexts[stem]
        figure_rows.append(
            {
                "exhibit_id": stem,
                "exhibit_type": "figure",
                "title": title,
                "primary_output": _relative(paths["png"]),
                "companion_outputs": ";".join(
                    [_relative(paths["pdf"]), _relative(paths["caption"])]
                ),
                "sample_period": context.sample,
                "units": units,
                "source_artifacts": sources,
                "generation_path": "scripts/generate_evidence.py",
                "calculation_definition": definition,
            }
        )
    return pd.DataFrame.from_records([*table_rows, *figure_rows])


def generate_evidence(
    *,
    table_dir: Path | None = None,
    figure_dir: Path | None = None,
) -> EvidenceBuild:
    """Generate and validate all locked core report evidence."""
    tables_root = table_dir or PROJECT_ROOT / "results/tables"
    figures_root = figure_dir or PROJECT_ROOT / "results/figures"
    tables_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    sources = _load_sources()
    performance = evidence.build_performance_report_table(sources["performance"])
    paths = evidence.build_return_paths(sources["fund_returns"])
    comparison_paths = evidence.build_intersected_comparison_paths(
        sources["fund_returns"]
    )
    weights = evidence.build_combined_ticker_weight_history(sources["fund_weights"])
    sentiment = evidence.build_sector_sentiment_series(sources["sector_sentiment"])
    fusion = evidence.build_fusion_report_table(sources["fusion"])

    table_paths = {
        "performance": tables_root / TABLE_FILENAMES["performance"],
        "fusion": tables_root / TABLE_FILENAMES["fusion"],
    }
    performance.to_csv(table_paths["performance"], index=False, date_format="%Y-%m-%d")
    fusion.to_csv(table_paths["fusion"], index=False, date_format="%Y-%m-%d")

    builders = {
        "growth_of_1_comparison": (
            _growth_figure(comparison_paths), len(comparison_paths), True
        ),
        "drawdown_equity_sentiment_tilt": (
            _drawdown_figure(paths),
            int(paths["fund"].eq("equity_sentiment_tilt").sum()),
            True,
        ),
        "combined_weights_over_time": (_weight_figure(weights), len(weights), False),
        "return_risk_comparison": (
            _return_risk_figure(performance),
            len(performance),
            True,
        ),
        "sector_sentiment_time_series": (
            _sentiment_figure(sentiment),
            len(sentiment),
            False,
        ),
        "fusion_before_after": (
            _fusion_figure(paths),
            int(paths["fund"].isin(evidence.FUSION_FUNDS).sum()),
            True,
        ),
    }
    figure_paths: dict[str, dict[str, Path]] = {}
    figure_contexts: dict[str, FigureContext] = {}
    qa_records: list[dict[str, object]] = []
    for stem in FIGURE_STEMS:
        (fig, axes, context), source_rows, require_labels = builders[stem]
        exported, qa = _export_figure(
            fig,
            axes,
            output_dir=figures_root,
            stem=stem,
            context=context,
            source_rows=source_rows,
            require_individual_axis_labels=require_labels,
        )
        figure_paths[stem] = exported
        figure_contexts[stem] = context
        qa_records.append(qa)

    qa_frame = pd.DataFrame.from_records(qa_records)
    table_paths["qa"] = tables_root / TABLE_FILENAMES["qa"]
    qa_frame.to_csv(table_paths["qa"], index=False)
    manifest = _manifest(performance, fusion, figure_paths, figure_contexts)
    table_paths["manifest"] = tables_root / TABLE_FILENAMES["manifest"]
    manifest.to_csv(table_paths["manifest"], index=False)

    return EvidenceBuild(
        tables=table_paths,
        figures=figure_paths,
        table_rows={
            "performance": len(performance),
            "fusion": len(fusion),
            "manifest": len(manifest),
            "qa": len(qa_frame),
        },
        figure_qa=qa_frame,
    )


def main() -> int:
    build = generate_evidence()
    print(
        "Evidence tables: "
        + ", ".join(
            f"{_relative(path)}={build.table_rows[name]} rows"
            for name, path in build.tables.items()
        )
    )
    print(
        "Evidence figures: "
        + ", ".join(
            f"{stem} ({', '.join(sorted(paths))})"
            for stem, paths in build.figures.items()
        )
    )
    print(
        "Figure QA: "
        f"{len(build.figure_qa)} PASS, "
        f"{int(build.figure_qa['layout_issue_count'].sum())} layout issues"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
