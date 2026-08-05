"""Professional Plotly figures for the PortFoYou investor interface.

The functions in this module are presentation-only: they transform validated,
precomputed app artifacts into figures without rebuilding a backtest or scoring
sentiment at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

INK = "#14283D"
MUTED = "#617386"
GRID = "#E3E9EF"
PANEL = "#FFFFFF"
POSITIVE = "#0E7C66"
NEGATIVE = "#C8514A"
MODEL_COLORS = {
    "VADER": "#1F3A5F",
    "FinBERT": "#B23A48",
}

# A small number of headline-heavy periods can be much more extreme than the
# rest of the trailing sentiment series.  Cap the *display* scale at this
# percentile so ordinary positive and negative regimes remain legible; hover
# text always retains the unmodified score.
HEATMAP_DISPLAY_QUANTILE = 0.95
ASSET_GROUP_COLORS = {
    "Stock": "#145DA0",
    "Crypto": "#D17B0F",
}

FUND_COLORS = {
    "combined_equal_weight": "#145DA0",
    "combined_min_variance": "#0E7C66",
    "combined_active_sector_allocation": "#C8514A",
    "combined_growth_sector_allocation": "#735DA5",
    "combined_aggressive_sector_allocation": "#B23A48",
    "equity_equal_weight": "#6C5CE7",
    "equity_sentiment_tilt": "#D17B0F",
    "equity_sentiment_21d_coverage_tilt": "#8C657E",
    "crypto_equal_weight": "#00A6A6",
    "crypto_min_variance": "#E76F51",
}

SECTOR_COLORS = {
    "Comm": "#35618F",
    "Consumer": "#E09035",
    "Energy": "#2F8F83",
    "Financials": "#735DA5",
    "Healthcare": "#C75C72",
    "Industrials": "#637381",
    "Materials": "#A76D3B",
    "RealEstate": "#5D8AA8",
    "Tech": "#1C78C0",
    "Utilities": "#4E8B57",
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _finish(
    figure: go.Figure,
    *,
    height: int,
    x_title: str | None = None,
    y_title: str | None = None,
    show_legend: bool = False,
) -> go.Figure:
    """Apply the shared restrained finance-dashboard visual system."""
    figure.update_layout(
        height=height,
        autosize=True,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": INK, "family": "Arial, sans-serif", "size": 12},
        margin={
            "l": 84,
            "r": 18,
            "t": 20,
            "b": 92 if show_legend else 46,
        },
        hoverlabel={"bgcolor": INK, "font_color": "#FFFFFF"},
        hovermode="x unified",
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.20,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11, "color": MUTED},
        },
    )
    figure.update_xaxes(
        title_text=x_title,
        showgrid=False,
        showline=True,
        linecolor=GRID,
        tickfont={"color": MUTED},
        title_font={"color": MUTED},
        fixedrange=False,
        automargin=True,
        title_standoff=12,
    )
    figure.update_yaxes(
        title_text=y_title,
        gridcolor=GRID,
        zerolinecolor=GRID,
        tickfont={"color": MUTED},
        title_font={"color": MUTED},
        fixedrange=False,
        automargin=True,
        title_standoff=12,
    )
    return figure


def growth_comparison_figure(
    fund_returns: pd.DataFrame,
    fund_labels: Mapping[str, str],
) -> go.Figure:
    """Build the common-sample growth comparison across every published fund."""
    funds = list(fund_labels)
    common_returns = (
        fund_returns.loc[
            fund_returns["fund"].isin(funds),
            ["date", "fund", "daily_return"],
        ]
        .pivot(index="date", columns="fund", values="daily_return")
        .reindex(columns=funds)
        .sort_index()
        .dropna(how="any")
    )
    if common_returns.empty:
        raise ValueError("published funds have no common return dates for comparison")
    common_growth = (1.0 + common_returns).cumprod()

    figure = go.Figure()
    for fund, label in fund_labels.items():
        figure.add_trace(
            go.Scatter(
                x=common_growth.index,
                y=common_growth[fund],
                name=label,
                mode="lines",
                line={
                    "color": FUND_COLORS.get(fund, MUTED),
                    "width": 2.7 if fund != "equity_sentiment_21d_coverage_tilt" else 2.0,
                    "dash": "dot" if fund == "equity_sentiment_21d_coverage_tilt" else "solid",
                },
                hovertemplate="Value of $1 invested: $%{y:.3f}<extra></extra>",
            )
        )
    _finish(
        figure,
        height=500,
        x_title=None,
        y_title=None,
        show_legend=False,
    )
    figure.update_yaxes(tickformat=".2f")
    figure.update_layout(
        # Keep the date-range controls in a dedicated row below the chart title.
        # Their previous y=1.08 position overlapped the date disclosure on
        # narrower Streamlit Cloud layouts.
        margin={"l": 84, "r": 18, "t": 92, "b": 76},
    )
    figure.add_annotation(
        text=(
            "Common-period comparison, rebased to $1 "
            f"({common_growth.index.min():%d %b %Y}-{common_growth.index.max():%d %b %Y})"
        ),
        x=0,
        y=1.16,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        font={"size": 13, "color": MUTED},
        align="left",
    )
    figure.update_xaxes(
        rangeslider={"visible": True, "thickness": 0.055, "bgcolor": "#EEF2F6"},
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"count": 2, "label": "2Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ],
            "x": 0,
            "y": 1.0,
            "yanchor": "bottom",
            "bgcolor": "#F3F6F8",
            "activecolor": "#DDE8EE",
            "font": {"color": MUTED, "size": 11},
        },
    )
    return figure


def risk_return_figure(
    performance: pd.DataFrame,
    fund_labels: Mapping[str, str],
) -> go.Figure:
    """Plot annualised return against annualised volatility for quick comparison."""
    figure = go.Figure()
    for row in performance.itertuples(index=False):
        label = fund_labels.get(row.fund, str(row.fund))
        figure.add_trace(
            go.Scatter(
                x=[row.annualised_volatility],
                y=[row.annualised_return],
                name=label,
                mode="markers",
                marker={
                    "size": 13,
                    "color": FUND_COLORS.get(row.fund, MUTED),
                    "line": {"color": "#FFFFFF", "width": 1.5},
                },
                customdata=[[row.sharpe_ratio, row.maximum_drawdown]],
                hovertemplate=(
                    "%{fullData.name}<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}"
                    "<br>Sharpe: %{customdata[0]:.3f}<br>Max drawdown: "
                    "%{customdata[1]:.2%}<extra></extra>"
                ),
            )
        )
    _finish(
        figure,
        height=455,
        x_title="Annual volatility",
        y_title=None,
        show_legend=False,
    )
    figure.update_layout(hovermode="closest")
    figure.add_annotation(
        text="Annual return",
        x=0,
        y=1.04,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        font={"size": 11, "color": MUTED},
    )
    figure.update_xaxes(tickformat=".0%")
    figure.update_yaxes(tickformat=".0%")
    return figure


def fund_growth_figure(
    returns: pd.DataFrame,
    fund: str,
    label: str,
) -> go.Figure:
    """Build a single-fund cumulative wealth figure."""
    selected = returns.sort_values("date")
    color = FUND_COLORS.get(fund, POSITIVE)
    figure = go.Figure(
        go.Scatter(
            x=selected["date"],
            y=selected["growth_of_1"],
            name=label,
            mode="lines",
            line={"color": color, "width": 2.8},
            hovertemplate="Value of $1 invested: $%{y:.3f}<extra></extra>",
        )
    )
    _finish(figure, height=330, y_title="Value of $1 invested")
    figure.update_yaxes(tickprefix="$", tickformat=".2f")
    figure.add_hline(y=1.0, line_color="#9AA8B5", line_width=1, line_dash="dot")
    return figure


def drawdown_figure(drawdown: pd.DataFrame) -> go.Figure:
    """Build a downside-focused drawdown area chart."""
    selected = drawdown.sort_values("date")
    figure = go.Figure(
        go.Scatter(
            x=selected["date"],
            y=selected["drawdown"],
            mode="lines",
            line={"color": NEGATIVE, "width": 2.2},
            fill="tozeroy",
            fillcolor="rgba(200, 81, 74, 0.14)",
            hovertemplate="Drawdown: %{y:.2%}<extra></extra>",
        )
    )
    _finish(figure, height=330, y_title="Drawdown")
    figure.update_yaxes(tickformat=".0%")
    return figure


def holdings_figure(holdings: pd.DataFrame, fund: str) -> go.Figure:
    """Show the largest current targets as a compact horizontal ranking."""
    top = holdings.nlargest(15, "target_weight").sort_values("target_weight")
    figure = go.Figure(
        go.Bar(
            x=top["target_weight"],
            y=top["ticker"],
            orientation="h",
            marker={"color": FUND_COLORS.get(fund, POSITIVE)},
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    _finish(figure, height=420, x_title="Target weight")
    figure.update_xaxes(tickformat=".0%")
    figure.update_yaxes(showgrid=False)
    return figure


def allocation_history_figure(history: pd.DataFrame) -> go.Figure:
    """Build the compounded historical value path for one allocation scenario."""
    selected = history.sort_values("date")
    figure = go.Figure(
        go.Scatter(
            x=selected["date"],
            y=selected["scenario_value"],
            mode="lines",
            line={"color": POSITIVE, "width": 2.8},
            fill="tozeroy",
            fillcolor="rgba(14, 124, 102, 0.10)",
            hovertemplate="Portfolio value: A$%{y:,.0f}<extra></extra>",
        )
    )
    _finish(figure, height=390, y_title="Portfolio value (AUD)")
    figure.update_yaxes(tickprefix="A$", tickformat=",.0f")
    return figure


def custom_portfolio_history_figure(history: pd.DataFrame) -> go.Figure:
    """Show the weighted custom portfolio's compounded historical value."""
    selected = history.sort_values("date")
    figure = go.Figure(
        go.Scatter(
            x=selected["date"],
            y=selected["portfolio_value"],
            mode="lines",
            line={"color": POSITIVE, "width": 2.8},
            fill="tozeroy",
            fillcolor="rgba(14, 124, 102, 0.10)",
            hovertemplate="Portfolio value: A$%{y:,.0f}<extra></extra>",
        )
    )
    _finish(figure, height=390, y_title="Historical portfolio value (AUD)")
    figure.update_yaxes(tickprefix="A$", tickformat=",.0f")
    return figure


def allocation_donut_figure(
    allocations: Mapping[str, float],
    fund_labels: Mapping[str, str],
) -> go.Figure:
    """Build a compact fund-allocation donut using the same fund color system."""
    funds = list(allocations)
    values = [allocations[fund] for fund in funds]
    figure = go.Figure(
        go.Pie(
            labels=[fund_labels.get(fund, fund) for fund in funds],
            values=values,
            hole=0.63,
            sort=False,
            direction="clockwise",
            marker={"colors": [FUND_COLORS.get(fund, MUTED) for fund in funds]},
            textinfo="percent",
            textposition="inside",
            hovertemplate="%{label}: %{value:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        height=340,
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": INK, "family": "Arial, sans-serif", "size": 11},
        showlegend=False,
        annotations=[
            {
                "text": "100%<br><span style='font-size:11px'>allocated</span>",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 21, "color": INK},
            }
        ],
    )
    return figure


def asset_group_donut_figure(asset_mix: pd.DataFrame) -> go.Figure:
    """Show current target weights grouped into stocks and cryptoassets."""
    labels = asset_mix["asset_group"].tolist()
    values = asset_mix["weight"].tolist()
    figure = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.63,
            sort=False,
            direction="clockwise",
            marker={"colors": [ASSET_GROUP_COLORS[label] for label in labels]},
            textinfo="percent",
            textposition="inside",
            hovertemplate="%{label}: %{value:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        height=340,
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": INK, "family": "Arial, sans-serif", "size": 11},
        showlegend=False,
        annotations=[
            {
                "text": "Asset<br><span style='font-size:11px'>mix</span>",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 19, "color": INK},
            }
        ],
    )
    return figure


def sentiment_trend_figure(
    sentiment: pd.DataFrame,
    sectors: Sequence[str],
    y_title: str,
) -> go.Figure:
    """Plot selected sector sentiment trends with an explicit neutral baseline."""
    figure = go.Figure()
    for sector in sectors:
        selected = sentiment.loc[sentiment["sector"].eq(sector)].sort_values("date")
        if selected.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=selected["date"],
                y=selected["display_score"],
                name=sector,
                mode="lines",
                connectgaps=False,
                line={"color": SECTOR_COLORS.get(sector, MUTED), "width": 2.1},
                hovertemplate="Score: %{y:.3f}<extra></extra>",
            )
        )
    _finish(figure, height=430, y_title=y_title, show_legend=False)
    figure.add_hline(y=0, line_color="#9AA8B5", line_width=1, line_dash="dot")
    return figure


def sentiment_model_comparison_figure(
    history: pd.DataFrame,
    *,
    y_title: str,
) -> go.Figure:
    """Overlay validated VADER and FinBERT sector scores without interpolation."""
    figure = go.Figure()
    specifications = (
        ("VADER", "display_vader", "VADER compound"),
        ("FinBERT", "display_finbert", "FinBERT P(positive) - P(negative)"),
    )
    ordered = history.sort_values("date", kind="mergesort")
    for model, column, hover_label in specifications:
        figure.add_trace(
            go.Scatter(
                x=ordered["date"],
                y=ordered[column],
                name=model,
                mode="lines",
                connectgaps=False,
                line={"color": MODEL_COLORS[model], "width": 2.2},
                hovertemplate=f"{hover_label}: %{{y:.3f}}<extra></extra>",
            )
        )
    _finish(
        figure,
        height=440,
        x_title="Date",
        y_title=y_title,
        show_legend=False,
    )
    figure.add_hline(y=0, line_color="#9AA8B5", line_width=1, line_dash="dot")
    figure.update_xaxes(rangeslider={"visible": True, "thickness": 0.08})
    figure.update_yaxes(range=[-1.02, 1.02])
    return figure


def sentiment_heatmap_figure(
    sector_sentiment: pd.DataFrame,
    sectors: Sequence[str],
) -> go.Figure:
    """Show the trailing 21-day sentiment regime across all ten sectors."""
    pivot = (
        sector_sentiment.pivot(index="date", columns="sector", values="raw_sector_compound")
        .sort_index()
        .reindex(columns=list(sectors))
    )
    rolling = pivot.rolling(21, min_periods=1).mean()
    finite = rolling.to_numpy(dtype=float)
    finite = np.abs(finite[np.isfinite(finite)])
    limit = max(
        float(np.quantile(finite, HEATMAP_DISPLAY_QUANTILE)) if finite.size else 0.0,
        0.05,
    )
    figure = go.Figure(
        go.Heatmap(
            z=rolling.to_numpy().T,
            x=rolling.index,
            y=rolling.columns,
            zmin=-limit,
            zmax=limit,
            zmid=0,
            colorscale=[
                [0.0, "#8E201B"],
                [0.35, "#C84740"],
                [0.47, "#F2C5C1"],
                [0.50, "#F8FAFC"],
                [0.53, "#BDE3D7"],
                [0.65, "#24846D"],
                [1.0, "#075D4C"],
            ],
            colorbar={
                "title": {"text": "21-day tone", "side": "right"},
                "thickness": 10,
                "len": 0.72,
                "tickvals": [-limit, 0, limit],
                "ticktext": ["More negative", "Neutral", "More positive"],
                "tickfont": {"color": MUTED, "size": 10},
            },
            hovertemplate="%{y}<br>%{x|%d %b %Y}<br>21-day score: %{z:.3f}<extra></extra>",
        )
    )
    _finish(figure, height=360, y_title=None)
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return figure
