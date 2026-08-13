"""Focused tests for core report tables, figure data, and exported evidence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts import generate_evidence
from src import evidence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sources() -> dict[str, pd.DataFrame]:
    return {
        "returns": pd.read_csv(
            PROJECT_ROOT / "results/data/fund_returns.csv",
            parse_dates=["date", "decision_date"],
        ),
        "weights": pd.read_csv(
            PROJECT_ROOT / "results/data/fund_weights.csv",
            parse_dates=["decision_date"],
        ),
        "sentiment": pd.read_csv(
            PROJECT_ROOT / "results/data/sector_sentiment_index.csv",
            parse_dates=["date", "tradable_signal_source_date"],
        ),
        "performance": pd.read_csv(
            PROJECT_ROOT / "results/tables/performance_metrics.csv",
            parse_dates=["sample_start_date", "sample_end_date"],
        ),
        "fusion": pd.read_csv(
            PROJECT_ROOT / "results/tables/fusion_comparison.csv",
            parse_dates=["sample_start_date", "sample_end_date"],
        ),
    }


def test_performance_report_is_core_only_and_traces_to_precise_metrics(
    sources: dict[str, pd.DataFrame],
) -> None:
    actual = evidence.build_performance_report_table(sources["performance"])
    assert tuple(actual["fund"]) == evidence.CORE_FUNDS
    assert "equity_sentiment_21d_coverage_tilt" not in set(actual["fund"])
    expected_periods = actual["asset_family"].map({"crypto": 365}).fillna(252)
    assert actual["periods_per_year"].eq(expected_periods).all()
    assert actual["risk_free_rate_pct"].eq(0.0).all()
    assert actual["source_artifact"].eq(
        "results/tables/performance_metrics.csv"
    ).all()

    precise = sources["performance"].set_index("fund")
    for row in actual.itertuples(index=False):
        source = precise.loc[row.fund]
        assert row.geometric_annual_return_pct == pytest.approx(
            float(f"{100.0 * source.annualised_return:.3g}")
        )
        assert row.annualised_volatility_pct == pytest.approx(
            float(f"{100.0 * source.annualised_volatility:.3g}")
        )
        assert row.sharpe_ratio == pytest.approx(
            float(f"{source.sharpe_ratio:.3g}")
        )
        assert row.maximum_drawdown_pct == pytest.approx(
            float(f"{100.0 * source.maximum_drawdown:.3g}")
        )


def test_growth_and_drawdown_paths_match_direct_formulas(
    sources: dict[str, pd.DataFrame],
) -> None:
    actual = evidence.build_return_paths(sources["returns"])
    assert set(actual["fund"]) == set(evidence.CORE_FUNDS)
    for fund, group in actual.groupby("fund", sort=True):
        source = sources["returns"].loc[
            sources["returns"]["fund"].eq(fund)
        ].sort_values("date")
        expected_growth = (1.0 + source["daily_return"]).cumprod().to_numpy()
        running_peak = np.maximum.accumulate(
            np.concatenate(([1.0], expected_growth))
        )[1:]
        expected_drawdown = expected_growth / running_peak - 1.0
        assert group.sort_values("date")["growth_of_1"].to_numpy() == pytest.approx(
            expected_growth,
            abs=1e-12,
        )
        assert group.sort_values("date")["drawdown"].to_numpy() == pytest.approx(
            expected_drawdown,
            abs=1e-12,
        )


def test_drawdown_path_includes_the_starting_dollar_peak() -> None:
    records: list[dict[str, object]] = []
    for fund in evidence.CORE_FUNDS:
        for date, daily_return, growth in zip(
            pd.to_datetime(["2023-01-03", "2023-01-04"]),
            [-0.10, 0.20],
            [0.90, 1.08],
            strict=True,
        ):
            records.append(
                {
                    "fund": fund,
                    "asset_family": "synthetic",
                    "method": "synthetic",
                    "date": date,
                    "daily_return": daily_return,
                    "growth_of_1": growth,
                }
            )

    paths = evidence.build_return_paths(pd.DataFrame.from_records(records))

    first_drawdowns = paths.groupby("fund")["drawdown"].first()
    assert first_drawdowns.to_numpy() == pytest.approx(
        np.full(len(evidence.CORE_FUNDS), -0.10),
        abs=1e-15,
    )


def test_combined_weight_history_conserves_logged_target_weights(
    sources: dict[str, pd.DataFrame],
) -> None:
    actual = evidence.build_combined_ticker_weight_history(sources["weights"])
    assert set(actual["fund"]) == set(evidence.COMBINED_FUNDS)
    categories = tuple(actual["display_holding"].cat.categories)
    assert len(categories) == 7
    assert categories[-1] == "Other assets"
    sums = actual.groupby(["fund", "decision_date"])["target_weight"].sum()
    assert sums.to_numpy() == pytest.approx(np.ones(len(sums)), abs=1e-12)

    decision = pd.Timestamp("2022-06-30")
    fund = "combined_min_variance"
    source = sources["weights"].loc[
        sources["weights"]["fund"].eq(fund)
        & sources["weights"]["decision_date"].eq(decision)
    ]
    ticker = categories[0]
    expected_ticker_weight = source.loc[
        source["ticker"].eq(ticker), "target_weight"
    ].item()
    observed_ticker_weight = actual.loc[
        actual["fund"].eq(fund)
        & actual["decision_date"].eq(decision)
        & actual["display_holding"].eq(ticker),
        "target_weight",
    ].item()
    assert observed_ticker_weight == pytest.approx(expected_ticker_weight, abs=1e-15)


def test_sector_figure_series_is_the_unmodified_raw_ten_sector_index(
    sources: dict[str, pd.DataFrame],
) -> None:
    actual = evidence.build_sector_sentiment_series(sources["sentiment"])
    assert set(actual["sector"].astype(str)) == set(evidence.EXPECTED_SECTORS)
    assert len(actual) == len(sources["sentiment"])
    expected = sources["sentiment"].sort_values(["sector", "date"])
    observed = actual.assign(sector=actual["sector"].astype(str)).sort_values(
        ["sector", "date"]
    )
    pd.testing.assert_series_equal(
        observed["raw_sector_compound"].reset_index(drop=True),
        expected["raw_sector_compound"].reset_index(drop=True),
        check_names=False,
    )
    no_news = ~observed["has_observed_news"].astype(bool)
    assert observed.loc[no_news, "raw_sector_compound"].isna().all()


def test_fusion_table_is_matched_locked_before_after_evidence(
    sources: dict[str, pd.DataFrame],
) -> None:
    actual = evidence.build_fusion_report_table(sources["fusion"])
    assert tuple(actual["role"]) == ("base", "enhanced")
    assert tuple(actual["fund"]) == evidence.FUSION_FUNDS
    assert actual["sample_start_date"].nunique() == 1
    assert actual["sample_end_date"].nunique() == 1
    assert actual["transaction_cost_bps"].eq(0.0).all()
    assert actual["source_artifact"].eq(
        "results/tables/fusion_comparison.csv"
    ).all()

    source = sources["fusion"].set_index("fund")
    enhanced = actual.set_index("fund").loc["equity_sentiment_tilt"]
    expected_difference = float(
        f"{100.0 * source.loc['equity_sentiment_tilt', 'annualised_return_difference_vs_base']:.3g}"
    )
    assert enhanced["annual_return_difference_pct_points"] == pytest.approx(
        expected_difference
    )


def test_evidence_generator_exports_all_core_tables_figures_and_qa(
    tmp_path: Path,
    sources: dict[str, pd.DataFrame],
) -> None:
    build = generate_evidence.generate_evidence(
        table_dir=tmp_path / "tables",
        figure_dir=tmp_path / "figures",
    )
    assert set(build.tables) == {"performance", "fusion", "manifest", "qa"}
    assert set(build.figures) == set(generate_evidence.FIGURE_STEMS)
    assert build.table_rows == {
        "performance": 9,
        "fusion": 2,
        "manifest": 8,
        "qa": 6,
    }
    for path in build.tables.values():
        assert path.exists() and path.stat().st_size > 0
    for paths in build.figures.values():
        assert set(paths) == {"png", "pdf", "caption"}
        assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    assert build.figure_qa["qa_status"].eq("PASS").all()
    assert build.figure_qa["layout_issue_count"].eq(0).all()
    assert build.figure_qa["minimum_font_points"].ge(8.0).all()

    manifest = pd.read_csv(build.tables["manifest"])
    required_exhibits = {
        "performance_table_core",
        "fusion_before_after_table",
        *generate_evidence.FIGURE_STEMS,
    }
    assert len(manifest) == 8
    assert manifest["exhibit_id"].is_unique
    assert set(manifest["exhibit_id"]) == required_exhibits

    def sample_period(
        frame: pd.DataFrame,
        start_column: str,
        end_column: str | None = None,
    ) -> str:
        end_name = end_column or start_column
        start = pd.to_datetime(frame[start_column], errors="raise").min()
        end = pd.to_datetime(frame[end_name], errors="raise").max()
        return f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"

    core_performance = sources["performance"].loc[
        sources["performance"]["fund"].isin(evidence.CORE_FUNDS)
    ]
    locked_fusion = sources["fusion"].loc[
        sources["fusion"]["fund"].isin(evidence.FUSION_FUNDS)
    ]
    core_returns = sources["returns"].loc[
        sources["returns"]["fund"].isin(evidence.CORE_FUNDS)
    ]
    drawdown_returns = sources["returns"].loc[
        sources["returns"]["fund"].eq("equity_sentiment_tilt")
    ]
    combined_weights = sources["weights"].loc[
        sources["weights"]["fund"].isin(evidence.COMBINED_FUNDS)
    ]
    matched_fusion_returns = sources["returns"].loc[
        sources["returns"]["fund"].isin(evidence.FUSION_FUNDS)
    ]
    fusion_ranges = matched_fusion_returns.groupby("fund")["date"].agg(
        ["min", "max"]
    )
    assert fusion_ranges["min"].nunique() == 1
    assert fusion_ranges["max"].nunique() == 1

    expected_samples = {
        "performance_table_core": sample_period(
            core_performance, "sample_start_date", "sample_end_date"
        ),
        "fusion_before_after_table": sample_period(
            locked_fusion, "sample_start_date", "sample_end_date"
        ),
            "growth_of_1_comparison": sample_period(
                evidence.build_intersected_comparison_paths(core_returns), "date"
            ),
        "drawdown_equity_sentiment_tilt": sample_period(drawdown_returns, "date"),
        "combined_weights_over_time": sample_period(
            combined_weights, "decision_date"
        ),
        "return_risk_comparison": sample_period(
            evidence.build_common_period_performance_table(core_returns),
            "sample_start_date",
            "sample_end_date",
        ),
        "sector_sentiment_time_series": sample_period(
            sources["sentiment"], "date"
        ),
        "fusion_before_after": sample_period(matched_fusion_returns, "date"),
    }
    assert (
        expected_samples["combined_weights_over_time"]
        == "2020-12-31 to 2023-11-30"
    )
    assert (
        expected_samples["sector_sentiment_time_series"]
        == "2020-01-02 to 2023-12-29"
    )
    actual_samples = manifest.set_index("exhibit_id")["sample_period"].to_dict()
    assert actual_samples == expected_samples

    for stem in generate_evidence.FIGURE_STEMS:
        caption_lines = build.figures[stem]["caption"].read_text(
            encoding="utf-8"
        ).splitlines()
        sample_header = caption_lines.index("## Sample")
        caption_sample = caption_lines[sample_header + 1].strip()
        assert caption_sample == actual_samples[stem]


def test_production_paths_are_project_root_relative() -> None:
    for path in generate_evidence.SOURCE_PATHS.values():
        assert path.is_relative_to(generate_evidence.PROJECT_ROOT)
    expected_tables = {
        "performance_table_core.csv",
        "fusion_before_after_table.csv",
        "evidence_manifest.csv",
        "figure_qa.csv",
    }
    assert set(generate_evidence.TABLE_FILENAMES.values()) == expected_tables


def test_return_risk_exhibit_uses_family_specific_annualisation() -> None:
    returns = pd.read_csv(
        PROJECT_ROOT / "results/data/fund_returns.csv",
        parse_dates=["date", "decision_date"],
    )
    performance = evidence.build_common_period_performance_table(returns)
    figure, _, context = generate_evidence._return_risk_figure(performance)
    try:
        assert context.sample == "2021-01-04 to 2023-12-29"
        assert "Common chronological window" in context.note
        assert "sqrt(252) for equity/combined" in context.note
        assert "sqrt(365) for crypto" in context.note
        labels = figure.axes[0].get_legend_handles_labels()[1]
        assert any("Multi-Asset Equal Wt" in label for label in labels)
        assert any("Crypto Equal Weight" in label for label in labels)
        assert not any("1/N" in label for label in labels)
    finally:
        generate_evidence.plt.close(figure)


def test_common_period_crypto_metrics_use_native_calendar(
    sources: dict[str, pd.DataFrame],
) -> None:
    performance = evidence.build_common_period_performance_table(sources["returns"])
    crypto = performance.set_index("fund")

    assert crypto.loc["crypto_equal_weight", "sample_start_date"] == pd.Timestamp(
        "2021-01-04"
    )
    assert crypto.loc["crypto_equal_weight", "sample_end_date"] == pd.Timestamp(
        "2023-12-29"
    )
    assert crypto.loc["crypto_equal_weight", "observations"] == 1090
    assert crypto.loc["crypto_min_variance", "observations"] == 1090
    assert crypto.loc["crypto_equal_weight", "periods_per_year"] == 365
    assert crypto.loc["crypto_equal_weight", "final_growth_of_1_dollars"] == pytest.approx(2.11)
    assert crypto.loc["crypto_equal_weight", "geometric_annual_return_pct"] == pytest.approx(28.3)
    assert crypto.loc["crypto_equal_weight", "sharpe_ratio"] == pytest.approx(0.717)
    assert crypto.loc["crypto_min_variance", "final_growth_of_1_dollars"] == pytest.approx(1.73)
    assert crypto.loc["crypto_min_variance", "geometric_annual_return_pct"] == pytest.approx(20.1)
    assert crypto.loc["crypto_min_variance", "sharpe_ratio"] == pytest.approx(0.608)
    assert (
        crypto.loc["crypto_min_variance", "annualised_volatility_pct"]
        < crypto.loc["crypto_equal_weight", "annualised_volatility_pct"]
    )
    assert (
        crypto.loc["crypto_min_variance", "geometric_annual_return_pct"]
        < crypto.loc["crypto_equal_weight", "geometric_annual_return_pct"]
    )


def test_growth_and_fusion_exhibits_use_investor_facing_names(
    sources: dict[str, pd.DataFrame],
) -> None:
    paths = evidence.build_return_paths(sources["returns"])
    figures = [
        generate_evidence._growth_figure(paths)[0],
        generate_evidence._fusion_figure(paths)[0],
    ]
    try:
        for figure in figures:
            labels = [
                label
                for axis in figure.axes
                for label in axis.get_legend_handles_labels()[1]
            ]
            assert labels
            assert not any("1/N" in label for label in labels)
            assert not any(label.startswith("Combined ") for label in labels)
    finally:
        for figure in figures:
            generate_evidence.plt.close(figure)


def test_weight_exhibit_uses_readable_investor_labels(
    sources: dict[str, pd.DataFrame],
) -> None:
    history = evidence.build_combined_ticker_weight_history(sources["weights"])
    figure, axes, _ = generate_evidence._weight_figure(history)
    try:
        titles = [axis.get_title(loc="left") for axis in axes]
        assert titles == [
            generate_evidence.WEIGHT_PLOT_LABELS[fund]
            for fund in evidence.COMBINED_FUNDS
        ]
        assert all(axis.get_xlabel() == "" for axis in axes[:-1])
        assert axes[-1].get_xlabel() == "Decision date"
        assert all(axis.get_ylabel() == "Weight (%)" for axis in axes)
    finally:
        generate_evidence.plt.close(figure)
