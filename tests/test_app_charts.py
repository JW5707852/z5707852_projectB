"""Focused presentation-contract tests for the PortFoYou Plotly figures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from src.app_artifacts import (
    EXPECTED_SECTORS,
    FUND_LABELS,
    drawdown_history,
    latest_holdings,
    load_app_artifacts,
)
from src.app_charts import (
    HEATMAP_DISPLAY_QUANTILE,
    allocation_donut_figure,
    asset_group_donut_figure,
    custom_portfolio_history_figure,
    drawdown_figure,
    fund_growth_figure,
    growth_comparison_figure,
    holdings_figure,
    risk_return_figure,
    sentiment_heatmap_figure,
    sentiment_trend_figure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_comparison_figures_cover_every_published_fund() -> None:
    artifacts = load_app_artifacts(PROJECT_ROOT)

    growth = growth_comparison_figure(artifacts.fund_returns, FUND_LABELS)
    risk_return = risk_return_figure(artifacts.performance_metrics, FUND_LABELS)

    assert len(growth.data) == len(FUND_LABELS)
    assert len(risk_return.data) == len(FUND_LABELS)
    assert growth.layout.hovermode == "x unified"
    assert growth.layout.xaxis.rangeslider.visible is True
    assert growth.layout.showlegend is False
    assert risk_return.layout.showlegend is False
    assert {trace.name for trace in growth.data} == set(FUND_LABELS.values())

    common_returns = (
        artifacts.fund_returns.pivot(index="date", columns="fund", values="daily_return")
        .reindex(columns=list(FUND_LABELS))
        .sort_index()
        .dropna(how="any")
    )
    expected_growth = (1.0 + common_returns).cumprod()
    assert growth.layout.annotations[0].text.startswith(
        "Common-period comparison, rebased to $1"
    )
    assert growth.layout.annotations[0].y == 1.16
    assert growth.layout.xaxis.rangeselector.y == 1.0
    assert growth.layout.xaxis.rangeselector.yanchor == "bottom"
    for trace in growth.data:
        fund = next(fund for fund, label in FUND_LABELS.items() if label == trace.name)
        assert list(trace.x) == list(common_returns.index)
        np.testing.assert_allclose(trace.y, expected_growth[fund], atol=1e-12, rtol=1e-12)


def test_fact_sheet_and_allocation_figures_keep_client_units() -> None:
    artifacts = load_app_artifacts(PROJECT_ROOT)
    fund = "combined_equal_weight"
    returns = artifacts.fund_returns.loc[artifacts.fund_returns["fund"].eq(fund)]
    holdings = latest_holdings(artifacts.fund_weights, fund)

    growth = fund_growth_figure(returns, fund, FUND_LABELS[fund])
    drawdown = drawdown_figure(drawdown_history(artifacts.fund_returns, fund))
    holding_chart = holdings_figure(holdings, fund)
    donut = allocation_donut_figure(
        {published_fund: 0.2 for published_fund in FUND_LABELS},
        FUND_LABELS,
    )

    assert growth.layout.yaxis.tickprefix == "$"
    assert growth.layout.yaxis.title.text == "Value of $1 invested"
    assert drawdown.layout.yaxis.tickformat == ".0%"
    assert holding_chart.layout.xaxis.tickformat == ".0%"
    assert len(donut.data[0].labels) == len(FUND_LABELS)
    assert donut.layout.showlegend is False


def test_custom_portfolio_figures_show_value_and_asset_type_mix() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "portfolio_value": [100_000.0, 101_000.0],
        }
    )
    asset_mix = pd.DataFrame(
        {"asset_group": ["Stock", "Crypto"], "weight": [0.65, 0.35]}
    )

    value_chart = custom_portfolio_history_figure(history)
    donut = asset_group_donut_figure(asset_mix)

    assert value_chart.layout.yaxis.tickprefix == "A$"
    assert value_chart.layout.yaxis.title.text == "Historical portfolio value (AUD)"
    assert list(donut.data[0].labels) == ["Stock", "Crypto"]
    assert list(donut.data[0].values) == [0.65, 0.35]


def test_sentiment_figures_show_all_sectors_and_selected_trends() -> None:
    artifacts = load_app_artifacts(PROJECT_ROOT)
    selected = ["Tech", "Financials", "Energy", "Healthcare"]
    sentiment = artifacts.sector_sentiment.loc[
        artifacts.sector_sentiment["sector"].isin(selected)
    ].copy()
    sentiment["display_score"] = sentiment.groupby("sector", sort=False)[
        "raw_sector_compound"
    ].transform(lambda values: values.rolling(21, min_periods=1).mean())

    heatmap = sentiment_heatmap_figure(
        artifacts.sector_sentiment,
        EXPECTED_SECTORS,
    )
    trend = sentiment_trend_figure(
        sentiment,
        selected,
        "21-day mean compound score",
    )

    assert list(heatmap.data[0].y) == list(EXPECTED_SECTORS)
    assert heatmap.data[0].zmin < 0 < heatmap.data[0].zmax
    assert heatmap.data[0].colorbar.title.text == "21-day tone"
    assert heatmap.data[0].colorscale[0][1] == "#8E201B"
    assert HEATMAP_DISPLAY_QUANTILE == 0.95
    assert {trace.name for trace in trend.data} == set(selected)
    assert trend.layout.hovermode == "x unified"
    assert trend.layout.showlegend is False
