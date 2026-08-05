"""PortFoYou: a professional investor view of precomputed project results."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from src.app_artifacts import (
    EXPECTED_SECTORS,
    FUND_LABELS,
    AppArtifactError,
    AppArtifacts,
    FinBERTAppArtifacts,
    apply_annual_management_fee,
    calculate_allocation_scenario,
    drawdown_history,
    latest_holdings,
    load_app_artifacts,
    load_finbert_app_artifacts,
    sentiment_model_history,
)
from src.app_charts import (
    ASSET_GROUP_COLORS,
    FUND_COLORS,
    MODEL_COLORS,
    PLOTLY_CONFIG,
    SECTOR_COLORS,
    allocation_donut_figure,
    allocation_history_figure,
    asset_group_donut_figure,
    custom_portfolio_history_figure,
    drawdown_figure,
    fund_growth_figure,
    growth_comparison_figure,
    holdings_figure,
    risk_return_figure,
    sentiment_heatmap_figure,
    sentiment_model_comparison_figure,
    sentiment_trend_figure,
)
from src.custom_portfolio import (
    ASSET_GROUPS,
    CustomPortfolioError,
    calculate_custom_portfolio,
    load_investable_asset_returns,
)

PROJECT_ROOT = Path(__file__).resolve().parent
NAVIGATION = (
    "Fund Screener",
    "Fund Profile",
    "Fund Allocation",
    "Portfolio Simulator",
    "Sector Sentiment",
)
DEFAULT_SENTIMENT_SECTORS = ("Tech", "Financials", "Energy", "Healthcare")

FUND_DESCRIPTIONS = {
    "combined_equal_weight": "Equal weights across 50 US equities and 10 cryptoassets.",
    "combined_min_variance": (
        "A long-only equity and crypto portfolio designed to minimise estimated risk."
    ),
    "combined_active_sector_allocation": (
        "An exploratory multi-asset allocation: 70% sector inverse-volatility "
        "equity core, 20% in the two strongest lagged-sentiment sectors, and "
        "10% equally across cryptoassets."
    ),
    "combined_growth_sector_allocation": (
        "A balanced-growth allocation: 80% sector inverse-volatility equity "
        "core, 15% split across the three strongest lagged-sentiment sectors, "
        "and 5% equally across cryptoassets."
    ),
    "combined_aggressive_sector_allocation": (
        "An exploratory high-growth allocation: 50% sector inverse-volatility "
        "equity core, 30% split across the three strongest lagged-sentiment "
        "sectors, and 20% equally across cryptoassets."
    ),
    "equity_equal_weight": "Equal weights across 50 US equities.",
    "equity_sentiment_tilt": (
        "A US equity portfolio adjusted using the prior trading day's sector sentiment."
    ),
    "equity_sentiment_21d_coverage_tilt": (
        "An exploratory US equity strategy using a 21-day sector sentiment signal."
    ),
    "crypto_equal_weight": (
        "Equal weights across the ten supplied cryptocurrencies on their native daily calendar."
    ),
    "crypto_min_variance": (
        "A long-only, fully invested minimum-variance portfolio across the ten supplied "
        "cryptocurrencies."
    ),
}

METRIC_HELP = {
    "value_of_one": (
        "The ending value of $1 invested at the start of the historical period, "
        "with daily returns compounded."
    ),
    "cumulative_return": (
        "The total percentage gain or loss across the full historical period."
    ),
    "annual_return": (
        "Compound annual growth rate calculated from the full daily return history."
    ),
    "annual_volatility": (
        "The annualised standard deviation of daily returns; higher values indicate "
        "larger day-to-day fluctuations."
    ),
    "sharpe": "Annualised return per unit of volatility, using a 0% risk-free rate.",
    "maximum_drawdown": (
        "The largest historical fall from a previous portfolio-value peak."
    ),
    "target_weight": "The percentage of the portfolio assigned to a fund or security.",
    "sentiment": (
        "Headline tone ranges from negative to positive. It describes news language; "
        "it is not a return forecast."
    ),
    "news_days": "The share of sector trading days with at least one mapped headline.",
    "company_coverage": (
        "The average share of companies in a sector represented by headlines on "
        "covered trading days."
    ),
    "pearson": "Measures the linear relationship between the two model scores.",
    "spearman": "Measures whether the two models rank sector sentiment similarly.",
    "agreement": (
        "The share of headlines assigned the same descriptive label by both models."
    ),
}


@st.cache_data
def _load_artifacts() -> AppArtifacts:
    """Load and validate only the four published CSV artifacts."""
    return load_app_artifacts(PROJECT_ROOT)


@st.cache_data
def _load_asset_returns() -> pd.DataFrame:
    """Load the precomputed individual-security return artifact."""
    return load_investable_asset_returns(PROJECT_ROOT)


@st.cache_data
def _load_finbert_artifacts(
    vader_sector_sentiment: pd.DataFrame,
) -> FinBERTAppArtifacts:
    """Load the optional precomputed FinBERT robustness artifacts."""
    return load_finbert_app_artifacts(PROJECT_ROOT, vader_sector_sentiment)


def _label(fund: str) -> str:
    return FUND_LABELS.get(fund, fund.replace("_", " ").title())


def _percent(value: float) -> str:
    return f"{value:.2%}"


def _currency(value: float) -> str:
    return f"A${value:,.0f}"


def _plotly(figure: object, *, key: str) -> None:
    """Render a consistently configured interactive figure."""
    st.plotly_chart(
        figure,
        width="stretch",
        theme=None,
        config=PLOTLY_CONFIG,
        key=key,
    )


def _chart_legend(items: list[tuple[str, str]], *, label: str) -> None:
    """Render a responsive legend outside Plotly so long labels stay visible."""
    entries = "".join(
        (
            '<span class="pf-chart-legend-item" role="listitem">'
            f'<span class="pf-chart-legend-dot" style="background:{escape(color)}"></span>'
            f"<span>{escape(name)}</span></span>"
        )
        for name, color in items
    )
    st.markdown(
        f'<div class="pf-chart-legend" role="list" aria-label="{escape(label)}">{entries}</div>',
        unsafe_allow_html=True,
    )


def _page_intro(title: str, description: str) -> None:
    """Render a compact finance-app page heading."""
    st.header(title, anchor=title.lower().replace(" ", "-"))
    st.markdown(f'<p class="pf-page-copy">{description}</p>', unsafe_allow_html=True)


def _performance_display(performance: pd.DataFrame) -> pd.DataFrame:
    """Return a compact, sortable performance view with numeric columns."""
    fund_order = {fund: position for position, fund in enumerate(FUND_LABELS)}
    ordered = performance.assign(
        _display_order=performance["fund"].map(fund_order)
    ).sort_values("_display_order")
    return pd.DataFrame(
        {
            "Fund": ordered["fund"].map(_label),
            "Cumulative return (%)": (ordered["final_growth_of_1"] - 1.0) * 100,
            "Annual return (%)": ordered["annualised_return"] * 100,
            "Annual volatility (%)": ordered["annualised_volatility"] * 100,
            "Sharpe ratio": ordered["sharpe_ratio"],
            "Maximum drawdown (%)": ordered["maximum_drawdown"] * 100,
        }
    ).reset_index(drop=True)


def _render_research_info(
    artifacts: AppArtifacts,
) -> None:
    """Keep provenance, metric definitions, assumptions, and risk in one control."""
    performance = artifacts.performance_metrics
    first = performance["sample_start_date"].min().date()
    last = performance["sample_end_date"].max().date()
    news_first = artifacts.sector_sentiment["date"].min().date()
    news_last = artifacts.sector_sentiment["date"].max().date()
    with st.popover(
        "Data & methodology",
        icon=":material/info:",
        type="tertiary",
        width="stretch",
        help="View data coverage, calculation definitions and risk disclosures.",
    ):
        st.markdown("### Data & methodology")
        st.write(
            "Historical research for comparing systematic portfolios. Results are "
            "simulations, not live fund records, forecasts or personal advice."
        )
        st.markdown("**Coverage**")
        st.write(
            f"Fund performance: {first:%d %b %Y} to {last:%d %b %Y}. "
            f"Headline sentiment: {news_first:%d %b %Y} to {news_last:%d %b %Y}. "
            "Individual-asset coverage is shown in Portfolio Simulator."
        )
        st.markdown("**Data provenance**")
        st.write(
            "Fund returns, holdings and individual-security returns are precomputed "
            "from adjusted closing prices. Equity and crypto returns are calculated "
            "on their native calendars before crypto is aligned to equity trading "
            "dates. Sector sentiment is precomputed from supplied headlines. The "
            "interface performs display and allocation arithmetic only."
        )
        st.markdown("**Portfolio construction**")
        st.write(
            "Long-only, fully invested portfolios. Weights are set monthly using "
            "only prior observations and take effect on the next available return date. "
            "Multi-asset portfolios use the equity trading calendar; crypto-only portfolios "
            "use crypto's daily calendar."
        )
        st.markdown("**Performance definitions**")
        st.write(
            "Returns are compounded; volatility and Sharpe ratio are annualised using "
            "252 trading days for equity-calendar funds and 365 calendar days for crypto-only "
            "funds, with a 0% risk-free rate. Maximum drawdown is measured "
            "from the running portfolio-value peak. Transaction costs are 0 bps."
        )
        st.markdown("**Sentiment methodology**")
        st.write(
            "Headlines are mapped to the next available equity trading day. Sector "
            "scores first average headlines by company and then equal-weight companies. "
            "Any portfolio signal is delayed by one trading day. VADER uses its "
            "compound score; FinBERT uses positive probability minus negative probability."
        )
        st.markdown("**Risk disclosures**")
        st.write(
            "Historical performance may not repeat. Prices can fall sharply, and "
            "diversification does not prevent losses. Results exclude investor taxes, "
            "fees and slippage. Sentiment measures are noisy and do not establish "
            "predictive ability."
        )


def _render_app_header(
    artifacts: AppArtifacts,
) -> str:
    """Render a compact market-site masthead, utility control, and top navigation."""
    brand, utility = st.columns([5, 1.25], vertical_alignment="center")
    with brand:
        st.markdown(
            """
            <div class="pf-market-masthead">
                <span class="pf-wordmark">PortFoYou</span>
                <span class="pf-product-label">Fund analytics</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with utility:
        _render_research_info(artifacts)

    page = st.segmented_control(
        "Primary navigation",
        NAVIGATION,
        default=NAVIGATION[0],
        required=True,
        key="main_navigation",
        label_visibility="collapsed",
        width="stretch",
    )
    return str(page)


def _render_compare(artifacts: AppArtifacts) -> None:
    _page_intro(
        "Fund Screener",
        "Compare historical return, risk and drawdown across the available strategies.",
    )
    performance = artifacts.performance_metrics
    best_growth = performance.loc[performance["final_growth_of_1"].idxmax()]
    best_sharpe = performance.loc[performance["sharpe_ratio"].idxmax()]
    lowest_risk = performance.loc[performance["annualised_volatility"].idxmin()]

    metrics = st.columns(3)
    metrics[0].metric(
        "Highest cumulative return",
        _percent(float(best_growth["final_growth_of_1"] - 1.0)),
        help=f"{METRIC_HELP['cumulative_return']} Strategy: {_label(str(best_growth['fund']))}.",
    )
    metrics[1].metric(
        "Highest Sharpe ratio",
        f"{best_sharpe['sharpe_ratio']:.3f}",
        help=f"{METRIC_HELP['sharpe']} Strategy: {_label(str(best_sharpe['fund']))}.",
    )
    metrics[2].metric(
        "Lowest annual volatility",
        _percent(float(lowest_risk["annualised_volatility"])),
        help=(
            f"{METRIC_HELP['annual_volatility']} "
            f"Strategy: {_label(str(lowest_risk['fund']))}."
        ),
    )

    st.subheader(
        "Fund performance",
        help=(
            "Metrics use each fund's native history. Cross-fund charts use the explicitly "
            "intersected date range."
        ),
    )
    with st.container(border=True):
        st.dataframe(
            _performance_display(performance),
            width="stretch",
            height=215,
            hide_index=True,
            column_config={
                "Fund": st.column_config.TextColumn("Fund", width=340),
                "Cumulative return (%)": st.column_config.NumberColumn(
                    "Cumulative return", width=125, format="%.2f%%",
                    help=METRIC_HELP["cumulative_return"],
                ),
                "Annual return (%)": st.column_config.NumberColumn(
                    "Annual return", width=115, format="%.2f%%",
                    help=METRIC_HELP["annual_return"],
                ),
                "Annual volatility (%)": st.column_config.NumberColumn(
                    "Annual volatility", width=125, format="%.2f%%",
                    help=METRIC_HELP["annual_volatility"],
                ),
                "Sharpe ratio": st.column_config.NumberColumn(
                    width=105, format="%.3f", help=METRIC_HELP["sharpe"]
                ),
                "Maximum drawdown (%)": st.column_config.NumberColumn(
                    "Maximum drawdown", width=135, format="%.2f%%",
                    help=METRIC_HELP["maximum_drawdown"],
                ),
            },
        )

    growth_column, risk_column = st.columns([1.65, 1], gap="large")
    with growth_column:
        st.subheader(
            "Cumulative performance — common period",
            help=(
                "Compares every published fund over their shared historical dates, "
                "with each path rebased to $1. Fact-sheet metrics retain each "
                "fund's native history."
            ),
        )
        with st.container(border=True):
            _chart_legend(
                [(name, FUND_COLORS[fund]) for fund, name in FUND_LABELS.items()],
                label="Portfolio series",
            )
            _plotly(
                growth_comparison_figure(artifacts.fund_returns, FUND_LABELS),
                key="fund_growth_comparison",
            )
    with risk_column:
        st.subheader(
            "Annual return and risk",
            help="Each point compares annualised return with annualised volatility.",
        )
        with st.container(border=True):
            _chart_legend(
                [(name, FUND_COLORS[fund]) for fund, name in FUND_LABELS.items()],
                label="Portfolio markers",
            )
            _plotly(
                risk_return_figure(performance, FUND_LABELS),
                key="risk_return_map",
            )

def _render_fact_sheet(artifacts: AppArtifacts) -> None:
    _page_intro(
        "Fund Profile",
        "Review one strategy's performance, downside risk and current model holdings.",
    )
    selector, context = st.columns([2.2, 1], vertical_alignment="bottom")
    funds = list(FUND_LABELS)
    with selector, st.container(border=True):
        selected_fund = st.selectbox(
            "Fund",
            funds,
            format_func=_label,
            key="fact_sheet_fund",
            help="Select a strategy to view its historical profile and model holdings.",
        )
    row = artifacts.performance_metrics.set_index("fund").loc[selected_fund]
    with context:
        st.markdown(
            f'<div class="pf-context-chip">Holdings updated '
            f"<strong>{row['current_holdings_date'].date():%d %b %Y}</strong></div>",
            unsafe_allow_html=True,
        )

    st.subheader(_label(selected_fund))
    st.write(FUND_DESCRIPTIONS[selected_fund])
    if row["asset_family"] == "crypto":
        st.caption(
            "Crypto-native convention: daily crypto calendar, 365 periods per year, "
            "monthly rebalancing after a 252-observation expanding window, 0% risk-free "
            "rate and 0 bps transaction costs."
        )
    if selected_fund == "equity_sentiment_21d_coverage_tilt":
        st.caption(
            "Exploratory strategy: this variation was designed after reviewing the "
            "original results and should not be treated as a new independent test."
        )
    if selected_fund == "combined_active_sector_allocation":
        st.caption(
            "Exploratory fixed design: monthly rebalancing; 252-day historical "
            "volatility; 20% maximum sector weight; 5% maximum stock weight; "
            "and 0% transaction costs. It was not selected from a final-period "
            "performance grid."
        )
    if selected_fund == "combined_growth_sector_allocation":
        st.caption(
            "Exploratory fixed design: monthly rebalancing; 252-day historical "
            "volatility; 15% maximum sector weight; 3% maximum stock weight; "
            "0.5% maximum cryptoasset weight; and 0% transaction costs."
        )
    if selected_fund == "combined_aggressive_sector_allocation":
        st.caption(
            "Exploratory high-growth design: monthly rebalancing; 252-day "
            "historical volatility; 25% maximum sector weight; 5% maximum stock "
            "weight; 2% maximum cryptoasset weight; and 0% transaction costs."
        )

    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Value of $1 invested",
        f"${row['final_growth_of_1']:.3f}",
        help=METRIC_HELP["value_of_one"],
    )
    metric_columns[1].metric(
        "Annual return",
        _percent(row["annualised_return"]),
        help=METRIC_HELP["annual_return"],
    )
    metric_columns[2].metric(
        "Annual volatility",
        _percent(row["annualised_volatility"]),
        help=METRIC_HELP["annual_volatility"],
    )
    metric_columns[3].metric(
        "Sharpe ratio",
        f"{row['sharpe_ratio']:.3f}",
        help=METRIC_HELP["sharpe"],
    )
    metric_columns[4].metric(
        "Maximum drawdown",
        _percent(row["maximum_drawdown"]),
        help=METRIC_HELP["maximum_drawdown"],
    )

    returns = artifacts.fund_returns.loc[artifacts.fund_returns["fund"].eq(selected_fund)]
    growth_column, drawdown_column = st.columns(2, gap="large")
    with growth_column:
        st.subheader(
            "Value of $1 invested",
            help=METRIC_HELP["value_of_one"],
        )
        with st.container(border=True):
            _plotly(
                fund_growth_figure(returns, selected_fund, _label(selected_fund)),
                key=f"fund_growth_{selected_fund}",
            )
    with drawdown_column:
        st.subheader("Drawdown", help=METRIC_HELP["maximum_drawdown"])
        with st.container(border=True):
            _plotly(
                drawdown_figure(drawdown_history(artifacts.fund_returns, selected_fund)),
                key=f"fund_drawdown_{selected_fund}",
            )

    holdings = latest_holdings(artifacts.fund_weights, selected_fund)
    holdings_date = row["current_holdings_date"].date()
    st.subheader(
        f"Model holdings · {holdings_date:%d %b %Y}",
        help=(
            "Portfolio weights from the most recent historical rebalance. "
            "They are not live trade instructions."
        ),
    )
    chart_column, table_column = st.columns([1.25, 1], gap="large")
    with chart_column, st.container(border=True):
        _plotly(
            holdings_figure(holdings, selected_fund),
            key=f"holdings_{selected_fund}",
        )
    with table_column:
        table = holdings.copy()
        table["target_weight"] = table["target_weight"] * 100
        if "sector" in table:
            missing_sector = table["sector"].isna()
            table.loc[missing_sector, "sector"] = np.where(
                table.loc[missing_sector, "ticker"].str.endswith("-USD"),
                "Crypto",
                "Equity",
            )
        st.dataframe(
            table,
            width="stretch",
            height=420,
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "target_weight": st.column_config.NumberColumn(
                    "Target weight",
                    format="%.2f%%",
                    help=METRIC_HELP["target_weight"],
                ),
                "sector": st.column_config.TextColumn("Asset class / sector", width="medium"),
            },
        )
    st.caption("Chart: 15 largest positions. Table: all model holdings.")


def _allocation_inputs(funds: list[str]) -> dict[str, float]:
    default = 100.0 / len(funds)
    allocations: dict[str, float] = {}
    for row_start in range(0, len(funds), 2):
        input_columns = st.columns(2, gap="large")
        for offset, fund in enumerate(funds[row_start : row_start + 2]):
            position = row_start + offset
            with input_columns[offset]:
                allocations[fund] = st.number_input(
                    f"{_label(fund)} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=default,
                    step=1.0,
                    help=FUND_DESCRIPTIONS[fund],
                    key=f"allocation_{position}",
                )
    return allocations


def _render_fund_allocation(artifacts: AppArtifacts) -> None:
    _page_intro(
        "Fund Allocation",
        "Combine the available funds and review the historical result of a fixed allocation.",
    )

    funds = list(FUND_LABELS)
    with st.container(border=True):
        st.subheader("Portfolio settings")
        amount_column, fee_column, status_column = st.columns([1.25, 1.25, 1], gap="large")
        with amount_column:
            amount = st.number_input(
                "Investment amount (AUD)",
                min_value=1_000.0,
                max_value=10_000_000.0,
                value=100_000.0,
                step=1_000.0,
                key="scenario_amount",
                help="Starting amount used to illustrate the historical portfolio value.",
            )
        with fee_column:
            annual_fee_pct = st.number_input(
                "Hypothetical annual management fee (%)",
                min_value=0.0,
                max_value=3.0,
                value=0.0,
                step=0.05,
                format="%.2f",
                key="scenario_annual_management_fee",
                help=(
                    "Optional product-level scenario assumption. It is not an observed "
                    "historical fund cost or an estimate of investor fees."
                ),
            )
        st.markdown('<div class="pf-section-label">Target weights</div>', unsafe_allow_html=True)
        percentages = _allocation_inputs(funds)
        total = float(sum(percentages.values()))
        with status_column:
            st.metric(
                "Total weight",
                f"{total:.1f}%",
                help="Target weights must add to 100% before results are calculated.",
            )
            st.progress(
                min(max(total / 100.0, 0.0), 1.0),
                text="Required total: 100%",
            )

    if not np.isclose(total, 100.0, atol=1e-8, rtol=0.0):
        st.error(
            f"Target weights must total 100%. Current total: {total:.1f}%."
        )
        return

    fractions = {fund: percentage / 100.0 for fund, percentage in percentages.items()}
    scenario = calculate_allocation_scenario(
        artifacts.fund_returns,
        fractions,
        amount,
    )
    fee_scenario = apply_annual_management_fee(scenario, annual_fee_pct / 100.0)
    fee_adjusted_return = (
        fee_scenario.fee_adjusted_ending_value / scenario.initial_value - 1.0
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Gross ending value",
        _currency(fee_scenario.gross_ending_value),
        help="Historical gross allocation outcome before the optional scenario fee.",
    )
    metric_columns[1].metric(
        "Fee-adjusted ending value",
        _currency(fee_scenario.fee_adjusted_ending_value),
        help=(
            "Historical allocation outcome after applying the selected hypothetical annual fee "
            "daily."
        ),
    )
    metric_columns[2].metric(
        "Estimated fee drag",
        _currency(fee_scenario.estimated_fee_drag),
        help="Gross ending value less the fee-adjusted ending value in this hypothetical scenario.",
    )
    metric_columns[3].metric(
        "Fee-adjusted total return",
        _percent(fee_adjusted_return),
        help="Total historical return after the selected hypothetical annual fee drag.",
    )
    st.caption(
        f"Scenario assumption: {annual_fee_pct:.2f}% annual management fee, applied daily over "
        "the 252-trading-day allocation calendar. Taxes, brokerage, slippage, bid-ask spreads, "
        "and changing fund fees remain excluded."
    )

    path_column, mix_column = st.columns([1.55, 1], gap="large")
    with path_column:
        st.subheader(
            "Fee-adjusted portfolio value",
            anchor="fee-adjusted-portfolio-value",
            help=(
                "Historical allocation value in Australian dollars after the selected "
                "hypothetical fee."
            ),
        )
        with st.container(border=True):
            _plotly(
                allocation_history_figure(
                    fee_scenario.history.assign(
                        scenario_value=fee_scenario.history[
                            "fee_adjusted_scenario_value"
                        ]
                    )
                ),
                key="allocation_history",
            )
    with mix_column:
        st.subheader(
            "Target allocation",
            anchor="target-allocation",
            help=METRIC_HELP["target_weight"],
        )
        with st.container(border=True):
            _chart_legend(
                [
                    (f"{_label(fund)} — {fraction:.0%}", FUND_COLORS[fund])
                    for fund, fraction in fractions.items()
                ],
                label="Target allocation",
            )
            _plotly(
                allocation_donut_figure(fractions, FUND_LABELS),
                key="allocation_mix",
            )

    allocation_table = scenario.summary.copy()
    allocation_table["Fund"] = allocation_table["fund"].map(_label)
    allocation_table["allocation_fraction"] = allocation_table["allocation_fraction"] * 100
    st.subheader("Allocation breakdown")
    st.dataframe(
        allocation_table,
        width="stretch",
        height=215,
        hide_index=True,
        column_order=(
            "Fund",
            "allocation_fraction",
            "initial_allocation",
            "fund_growth_of_1",
            "historical_ending_value",
        ),
        column_config={
            "Fund": st.column_config.TextColumn(width="large"),
            "allocation_fraction": st.column_config.NumberColumn(
                "Allocation", format="%.1f%%", help=METRIC_HELP["target_weight"]
            ),
            "initial_allocation": st.column_config.NumberColumn(
                "Initial allocation", format="A$%,.0f"
            ),
            "fund_growth_of_1": st.column_config.NumberColumn(
                "Value of $1 invested", format="$%.3f", help=METRIC_HELP["value_of_one"]
            ),
            "historical_ending_value": st.column_config.NumberColumn(
                "Ending value", format="A$%,.0f"
            ),
        },
    )


def _initialise_custom_portfolio_state() -> None:
    if "custom_row_ids" not in st.session_state:
        st.session_state["custom_row_ids"] = [0, 1]
        st.session_state["custom_next_row_id"] = 2
        st.session_state["custom_group_0"] = "Stock"
        st.session_state["custom_ticker_0"] = "NVDA"
        st.session_state["custom_weight_0"] = 50.0
        st.session_state["custom_group_1"] = "Crypto"
        st.session_state["custom_ticker_1"] = "BTC-USD"
        st.session_state["custom_weight_1"] = 50.0


def _add_custom_asset_row() -> None:
    row_ids = list(st.session_state["custom_row_ids"])
    if len(row_ids) >= 15:
        return
    row_id = int(st.session_state["custom_next_row_id"])
    st.session_state["custom_row_ids"] = [*row_ids, row_id]
    st.session_state["custom_next_row_id"] = row_id + 1
    st.session_state[f"custom_group_{row_id}"] = "Stock"
    st.session_state[f"custom_weight_{row_id}"] = 0.0


def _remove_custom_asset_row(row_id: int) -> None:
    row_ids = list(st.session_state["custom_row_ids"])
    if len(row_ids) <= 2:
        return
    st.session_state["custom_row_ids"] = [item for item in row_ids if item != row_id]
    for prefix in ("custom_group", "custom_ticker", "custom_weight"):
        st.session_state.pop(f"{prefix}_{row_id}", None)


def _equal_weight_custom_assets() -> None:
    row_ids = list(st.session_state["custom_row_ids"])
    equal_weight = 100.0 / len(row_ids)
    for row_id in row_ids:
        st.session_state[f"custom_weight_{row_id}"] = equal_weight


def _render_custom_portfolio(asset_returns: pd.DataFrame) -> None:
    _page_intro(
        "Portfolio Simulator",
        "Build a fund-like portfolio from individual US stocks and cryptoassets, "
        "then inspect its weighted historical behaviour.",
    )
    st.info(
        "Historical simulation only — not personal advice, an optimisation, or an "
        "expectation of future performance."
    )
    _initialise_custom_portfolio_state()

    metadata = (
        asset_returns[["ticker", "asset_group", "sector"]]
        .drop_duplicates("ticker")
        .sort_values(["asset_group", "ticker"])
        .reset_index(drop=True)
    )
    tickers_by_group = {
        group: metadata.loc[metadata["asset_group"].eq(group), "ticker"].tolist()
        for group in ASSET_GROUPS
    }
    sector_by_ticker = metadata.set_index("ticker")["sector"].to_dict()

    with st.container(border=True):
        settings_left, settings_right = st.columns([1.35, 1], gap="large")
        with settings_left:
            amount = st.number_input(
                "Starting portfolio value (AUD)",
                min_value=1_000.0,
                max_value=10_000_000.0,
                value=100_000.0,
                step=1_000.0,
                key="custom_portfolio_amount",
                help="Starting value used to scale the weighted historical portfolio path.",
            )
        with settings_right:
            st.caption("Quick start")
            st.button(
                "Use equal weights",
                on_click=_equal_weight_custom_assets,
                width="stretch",
                help="Split 100% equally across the current holding rows.",
            )

        st.markdown('<div class="pf-section-label">Selected holdings</div>', unsafe_allow_html=True)
        st.caption("Choose 2-15 distinct assets. Use + Add asset to create another row.")
        row_ids = list(st.session_state["custom_row_ids"])
        selections: list[tuple[str, float]] = []
        for position, row_id in enumerate(row_ids, start=1):
            group_key = f"custom_group_{row_id}"
            ticker_key = f"custom_ticker_{row_id}"
            weight_key = f"custom_weight_{row_id}"
            if group_key not in st.session_state:
                st.session_state[group_key] = "Stock"
            group = str(st.session_state[group_key])
            available_tickers = tickers_by_group[group]
            if (
                ticker_key not in st.session_state
                or st.session_state[ticker_key] not in available_tickers
            ):
                used = {ticker for ticker, _ in selections}
                st.session_state[ticker_key] = next(
                    (ticker for ticker in available_tickers if ticker not in used),
                    available_tickers[0],
                )
            if weight_key not in st.session_state:
                st.session_state[weight_key] = 0.0

            type_column, ticker_column, weight_column, detail_column, remove_column = st.columns(
                [1.0, 1.25, 1.0, 1.1, 0.55], vertical_alignment="bottom"
            )
            with type_column:
                group = st.selectbox(
                    f"Type {position}",
                    ASSET_GROUPS,
                    key=group_key,
                )
            available_tickers = tickers_by_group[str(group)]
            if st.session_state[ticker_key] not in available_tickers:
                st.session_state[ticker_key] = available_tickers[0]
            with ticker_column:
                ticker = st.selectbox(
                    f"Asset {position}",
                    available_tickers,
                    key=ticker_key,
                )
            with weight_column:
                weight = st.number_input(
                    f"Weight {position} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=1.0,
                    key=weight_key,
                    help="Long-only target weight; short positions and leverage are disabled.",
                )
            with detail_column:
                st.caption("Sector")
                st.write(sector_by_ticker[str(ticker)])
            with remove_column:
                st.button(
                    "Remove",
                    key=f"custom_remove_{row_id}",
                    on_click=_remove_custom_asset_row,
                    args=(row_id,),
                    disabled=len(row_ids) <= 2,
                )
            selections.append((str(ticker), float(weight)))

        add_column, status_column = st.columns([1, 1.2], vertical_alignment="center")
        with add_column:
            st.button(
                "+ Add asset",
                on_click=_add_custom_asset_row,
                disabled=len(row_ids) >= 15,
                width="stretch",
            )
        total = float(sum(weight for _, weight in selections))
        with status_column:
            st.metric(
                "Total portfolio weight",
                f"{total:.2f}%",
                help="The selected asset weights must total exactly 100%.",
            )
            st.progress(min(max(total / 100.0, 0.0), 1.0), text="Required total: 100%")

    tickers = [ticker for ticker, _ in selections]
    if len(set(tickers)) != len(tickers):
        st.error("Each holding must be a different stock or cryptoasset.")
        return
    if sum(weight > 0 for _, weight in selections) < 2:
        st.error("At least two selected assets must have a positive weight.")
        return
    if not np.isclose(total, 100.0, atol=1e-8, rtol=0.0):
        st.error(f"Target weights must total 100%. Current total: {total:.2f}%.")
        return

    weights = {ticker: weight / 100.0 for ticker, weight in selections}
    scenario = calculate_custom_portfolio(asset_returns, weights, amount)
    total_return = scenario.ending_value / scenario.initial_value - 1.0
    first_date = scenario.history["date"].min().date()
    last_date = scenario.history["date"].max().date()

    metrics_top = st.columns(3)
    metrics_top[0].metric(
        "Historical ending value",
        _currency(scenario.ending_value),
        help="Ending value after compounding the weighted daily portfolio returns.",
    )
    metrics_top[1].metric(
        "Cumulative return",
        _percent(total_return),
        help=METRIC_HELP["cumulative_return"],
    )
    metrics_top[2].metric(
        "Annual geometric return",
        _percent(scenario.annualised_return),
        help=METRIC_HELP["annual_return"],
    )
    metrics_bottom = st.columns(3)
    metrics_bottom[0].metric(
        "Annual volatility",
        _percent(scenario.annualised_volatility),
        help=METRIC_HELP["annual_volatility"],
    )
    metrics_bottom[1].metric(
        "Sharpe ratio",
        f"{scenario.sharpe_ratio:.3f}",
        help=METRIC_HELP["sharpe"],
    )
    metrics_bottom[2].metric(
        "Maximum drawdown",
        _percent(scenario.maximum_drawdown),
        help=METRIC_HELP["maximum_drawdown"],
    )

    history_column, mix_column = st.columns([1.55, 1], gap="large")
    with history_column:
        st.subheader("Weighted portfolio history")
        with st.container(border=True):
            _plotly(
                custom_portfolio_history_figure(scenario.history),
                key="custom_portfolio_history",
            )
    with mix_column:
        st.subheader("Current asset-type mix")
        with st.container(border=True):
            _chart_legend(
                [
                    (
                        f"{row.asset_group} — {row.weight:.1%}",
                        ASSET_GROUP_COLORS[row.asset_group],
                    )
                    for row in scenario.asset_mix.itertuples(index=False)
                ],
                label="Asset-type allocation",
            )
            _plotly(asset_group_donut_figure(scenario.asset_mix), key="asset_type_mix")

    holdings = scenario.holdings.copy()
    holdings["weight"] = holdings["weight"] * 100.0
    st.subheader("Portfolio holdings")
    st.dataframe(
        holdings,
        width="stretch",
        hide_index=True,
        height=min(38 * len(holdings) + 42, 430),
        column_order=("ticker", "asset_group", "sector", "weight", "initial_allocation"),
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "asset_group": st.column_config.TextColumn("Asset type", width="small"),
            "sector": st.column_config.TextColumn("Sector", width="medium"),
            "weight": st.column_config.NumberColumn("Target weight", format="%.2f%%"),
            "initial_allocation": st.column_config.NumberColumn(
                "Starting allocation", format="A$%,.0f"
            ),
        },
    )
    st.caption(
        f"Historical sample: {first_date:%d %b %Y} to {last_date:%d %b %Y}. "
        "Constant target weights are applied to each equity-calendar daily return. "
        "Crypto returns were calculated on the native 7-day calendar before alignment; "
        "no missing returns are imputed. Metrics use 252 trading days, a 0% risk-free "
        "rate and a 0 bps transaction-cost research assumption."
    )


def _render_vader_sentiment(artifacts: AppArtifacts) -> None:
    sentiment_all = artifacts.sector_sentiment.copy()
    latest = sentiment_all["date"].max().date()
    observed = sentiment_all["has_observed_news"]
    latest_score = float(
        sentiment_all.loc[
            sentiment_all["date"].dt.date.eq(latest), "raw_sector_compound"
        ].mean()
    )
    if latest_score > 0.05:
        latest_tone = "Positive"
    elif latest_score < -0.05:
        latest_tone = "Negative"
    else:
        latest_tone = "Mixed"
    metrics = st.columns(4)
    metrics[0].metric(
        "Sectors tracked",
        len(EXPECTED_SECTORS),
        help="The ten US equity sectors represented in the research universe.",
    )
    metrics[1].metric(
        "Days with sector news",
        _percent(observed.mean()),
        help=METRIC_HELP["news_days"],
    )
    metrics[2].metric(
        "Average company coverage",
        _percent(sentiment_all["ticker_coverage_share"].mean()),
        help=METRIC_HELP["company_coverage"],
    )
    metrics[3].metric(
        "Latest market tone",
        latest_tone,
        help=(
            f"Average sector sentiment on the latest covered day was {latest_score:+.3f}. "
            f"{METRIC_HELP['sentiment']}"
        ),
    )

    st.subheader("Sector sentiment heatmap", help=METRIC_HELP["sentiment"])
    with st.container(border=True):
        _plotly(
            sentiment_heatmap_figure(sentiment_all, EXPECTED_SECTORS),
            key="sentiment_heatmap",
        )
    st.caption(
        "21-day average headline tone: green is more positive, red more negative, "
        "and pale cells are close to neutral."
    )

    with st.container(border=True):
        control_column, series_column = st.columns([2.2, 1], gap="large")
        with control_column:
            selected_sectors = st.multiselect(
                "Sectors",
                list(EXPECTED_SECTORS),
                default=list(DEFAULT_SENTIMENT_SECTORS),
                key="sentiment_sectors",
                help="Choose the sectors to display in the time-series chart.",
            )
        with series_column:
            view = st.radio(
                "Frequency",
                ("21-day average", "Daily"),
                horizontal=True,
                key="sentiment_series",
                help="Use the 21-day average to reduce day-to-day headline noise.",
            )
    if not selected_sectors:
        st.error("Select at least one sector.")
        return

    sentiment = sentiment_all.loc[sentiment_all["sector"].isin(selected_sectors)].copy()
    if view == "21-day average":
        sentiment["display_score"] = sentiment.groupby("sector", sort=False)[
            "raw_sector_compound"
        ].transform(lambda values: values.rolling(21, min_periods=1).mean())
        y_label = "21-day average sentiment"
        chart_caption = (
            "The line averages available observations across the trailing 21 trading "
            "days; days without sector news remain missing."
        )
    else:
        sentiment["display_score"] = sentiment["raw_sector_compound"]
        y_label = "Daily sentiment"
        chart_caption = (
            "Days without sector news appear as gaps rather than neutral observations."
        )

    st.subheader("Sentiment over time", help=METRIC_HELP["sentiment"])
    with st.container(border=True):
        _chart_legend(
            [(sector, SECTOR_COLORS[sector]) for sector in selected_sectors],
            label="Sector series",
        )
        _plotly(
            sentiment_trend_figure(sentiment, selected_sectors, y_label),
            key="sentiment_trends",
        )
    st.caption(chart_caption)

def _render_neural_sentiment(
    artifacts: AppArtifacts,
    finbert_artifacts: FinBERTAppArtifacts,
) -> None:
    st.subheader(
        "Sentiment model comparison",
        help=(
            "Compares a finance lexicon model with a finance-trained language model "
            "to show how measured headline tone changes with the method used."
        ),
    )
    st.write(
        "VADER and FinBERT score the same headlines using different methods. The "
        "comparison shows where the resulting market tone is consistent or differs."
    )

    comparison = finbert_artifacts.model_comparison
    headline_overall = comparison.loc[
        comparison["observation_unit"].eq("clean_headline_row") & comparison["sector"].eq("All")
    ].iloc[0]
    sector_overall = comparison.loc[
        comparison["observation_unit"].eq("matched_date_sector") & comparison["sector"].eq("All")
    ].iloc[0]

    st.markdown("#### Overall comparison")
    sector_metrics = st.columns(3)
    sector_metrics[0].metric(
        "Sector-days compared",
        f"{int(sector_overall['paired_observation_count']):,}",
        help="Trading-day and sector observations with scores from both models.",
    )
    sector_metrics[1].metric(
        "Pearson correlation",
        f"{sector_overall['pearson_correlation']:.3f}",
        help=METRIC_HELP["pearson"],
    )
    sector_metrics[2].metric(
        "Rank correlation",
        f"{sector_overall['spearman_correlation']:.3f}",
        help=METRIC_HELP["spearman"],
    )

    st.markdown("#### Headline classification")
    headline_metrics = st.columns(4)
    headline_metrics[0].metric(
        "Same-label rate",
        _percent(float(headline_overall["descriptive_label_agreement_rate"])),
        help=METRIC_HELP["agreement"],
    )
    headline_metrics[1].metric(
        "Opposite-signal rate",
        _percent(float(headline_overall["opposite_sign_rate"])),
        help="The share of headlines classified positive by one model and negative by the other.",
    )
    headline_metrics[2].metric(
        "VADER neutral rate",
        _percent(float(headline_overall["vader_neutral_rate"])),
        help="The share of headlines VADER classified as neutral.",
    )
    headline_metrics[3].metric(
        "FinBERT neutral rate",
        _percent(float(headline_overall["finbert_neutral_rate"])),
        help="The share of headlines FinBERT classified as neutral.",
    )

    manual = finbert_artifacts.manual_validation
    if manual is None:
        if finbert_artifacts.manual_validation_error:
            st.caption("Student-labelled validation is temporarily unavailable.")
        else:
            review_status = str(finbert_artifacts.metadata["manual_review_status"])
            st.caption(f"Manual headline review status: {review_status}")
    else:
        manual_metrics = manual.metrics

        def manual_value(model: str, metric: str) -> float:
            selected = manual_metrics.loc[
                manual_metrics["model"].eq(model)
                & manual_metrics["metric"].eq(metric),
                "value",
            ]
            return float(selected.iloc[0])

        vader_accuracy = manual_value("VADER", "accuracy")
        finbert_accuracy = manual_value("FinBERT", "accuracy")
        accuracy_difference = manual_value(
            "Paired comparison",
            "weighted_accuracy_difference_finbert_minus_vader",
        )
        mcnemar_p = manual_value("Paired comparison", "mcnemar_exact_p_value")
        representative_rows = int(manual.metadata["representative_rows"])
        discordant_pairs = round(
            manual_value("Paired comparison", "vader_only_correct")
            + manual_value("Paired comparison", "finbert_only_correct")
        )
        st.markdown("#### Student-labelled validation")
        st.caption(
            f"Classification evidence from {representative_rows} representative "
            "blind-reviewed headlines. "
            "The separate 50-headline disagreement sample is excluded from these metrics."
        )
        validation_metrics = st.columns(4)
        validation_metrics[0].metric("VADER weighted accuracy", _percent(vader_accuracy))
        validation_metrics[1].metric("FinBERT weighted accuracy", _percent(finbert_accuracy))
        validation_metrics[2].metric(
            "FinBERT minus VADER",
            f"{accuracy_difference * 100:+.1f} percentage points",
        )
        validation_metrics[3].metric("Exact McNemar p-value", f"{mcnemar_p:.3f}")

        validation_table = pd.DataFrame(
            {
                "Model": ["VADER", "FinBERT"],
                "Weighted accuracy": [vader_accuracy, finbert_accuracy],
                "Balanced accuracy": [
                    manual_value("VADER", "balanced_accuracy"),
                    manual_value("FinBERT", "balanced_accuracy"),
                ],
                "Macro F1": [
                    manual_value("VADER", "macro_f1"),
                    manual_value("FinBERT", "macro_f1"),
                ],
            }
        )
        st.dataframe(
            validation_table,
            width="stretch",
            hide_index=True,
            height=118,
            column_config={
                "Model": st.column_config.TextColumn("Model"),
                "Weighted accuracy": st.column_config.NumberColumn(
                    "Weighted accuracy", format="%.1f%%"
                ),
                "Balanced accuracy": st.column_config.NumberColumn(
                    "Balanced accuracy", format="%.1f%%"
                ),
                "Macro F1": st.column_config.NumberColumn("Macro F1", format="%.1f%%"),
            },
        )
        st.caption(
            "FinBERT has the higher point estimates in this small sample, but the "
            f"paired exact test gives p={mcnemar_p:.3f}. This is classification "
            "validation, not evidence of return predictability or investment superiority."
        )
        st.warning(
            "Preliminary evidence: the validation sample is small, and only "
            f"{discordant_pairs} paired headlines contribute information to the McNemar "
            "comparison. Statistical power is therefore limited, so the current effect "
            "estimates are imprecise and do not establish that either model is superior."
        )

    history = sentiment_model_history(
        artifacts.sector_sentiment,
        finbert_artifacts.sector_sentiment,
    )
    st.markdown("#### Compare sector history")
    sector_control, date_control, series_control = st.columns([1, 1.65, 1.35], gap="large")
    with sector_control:
        selected_sector = st.selectbox(
            "Sector",
            list(EXPECTED_SECTORS),
            index=list(EXPECTED_SECTORS).index("Tech"),
            key="finbert_comparison_sector",
        )
    minimum_date = history["date"].min().date()
    maximum_date = history["date"].max().date()
    with date_control:
        selected_range = st.date_input(
            "Date range",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
            key="finbert_comparison_dates",
        )
    with series_control:
        comparison_view = st.radio(
            "Frequency",
            ("21-day average", "Daily"),
            horizontal=False,
            key="finbert_comparison_series",
        )
    start_date, end_date = minimum_date, maximum_date
    if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
        start_date, end_date = selected_range
    selected_history = history.loc[
        history["sector"].eq(selected_sector)
        & history["date"].dt.date.ge(start_date)
        & history["date"].dt.date.le(end_date)
    ].copy()
    if comparison_view == "21-day average":
        for model_column, display_column in (
            ("vader_score", "display_vader"),
            ("finbert_score", "display_finbert"),
        ):
            rolling = selected_history[model_column].rolling(21, min_periods=1).mean()
            selected_history[display_column] = rolling.where(selected_history[model_column].notna())
        y_title = "21-day average sentiment"
    else:
        selected_history["display_vader"] = selected_history["vader_score"]
        selected_history["display_finbert"] = selected_history["finbert_score"]
        y_title = "Daily sentiment"

    with st.container(border=True):
        _chart_legend(
            [(model, color) for model, color in MODEL_COLORS.items()],
            label="Sentiment models",
        )
        _plotly(
            sentiment_model_comparison_figure(
                selected_history,
                y_title=y_title,
            ),
            key="finbert_sector_comparison",
        )
    st.caption("Days without sector news remain missing and appear as gaps.")

    st.markdown("#### Results by sector")
    sector_risk = comparison.loc[
        comparison["observation_unit"].eq("matched_date_sector") & comparison["sector"].ne("All"),
        [
            "sector",
            "paired_observation_count",
            "pearson_correlation",
            "spearman_correlation",
            "descriptive_label_agreement_rate",
            "vader_mean",
            "finbert_mean",
        ],
    ].copy()
    sector_risk["descriptive_label_agreement_rate"] *= 100
    st.dataframe(
        sector_risk,
        width="stretch",
        height=385,
        hide_index=True,
        column_config={
            "sector": st.column_config.TextColumn("Sector"),
            "paired_observation_count": st.column_config.NumberColumn(
                "Sector-days", format="%,d"
            ),
            "pearson_correlation": st.column_config.NumberColumn(
                "Pearson correlation", format="%.3f", help=METRIC_HELP["pearson"]
            ),
            "spearman_correlation": st.column_config.NumberColumn(
                "Rank correlation", format="%.3f", help=METRIC_HELP["spearman"]
            ),
            "descriptive_label_agreement_rate": st.column_config.NumberColumn(
                "Same-label rate", format="%.1f%%", help=METRIC_HELP["agreement"]
            ),
            "vader_mean": st.column_config.NumberColumn("Average VADER", format="%.3f"),
            "finbert_mean": st.column_config.NumberColumn("Average FinBERT", format="%.3f"),
        },
    )

    st.markdown("#### Review model disagreements")
    st.caption("Diagnostic examples where model classifications differ; not an accuracy sample.")
    examples = finbert_artifacts.disagreements.copy()
    disagreement_labels = {
        "neutral_non_neutral": "One model neutral; the other non-neutral",
        "opposite_sign": "Opposite positive and negative labels",
    }
    available_types = sorted(examples["sampling_stratum"].unique())
    filter_sector, filter_type, filter_year, filter_limit = st.columns(4)
    with filter_sector:
        example_sector = st.selectbox(
            "Sector filter",
            ["All sectors", *EXPECTED_SECTORS],
            key="finbert_example_sector",
        )
    with filter_type:
        example_type = st.selectbox(
            "Disagreement category",
            ["All types", *available_types],
            format_func=lambda value: disagreement_labels.get(value, value),
            key="finbert_example_type",
        )
    available_years = sorted(examples["year"].astype(int).unique())
    with filter_year:
        example_year = st.selectbox(
            "Year",
            ["All years", *available_years],
            key="finbert_example_year",
        )
    with filter_limit:
        example_limit = st.selectbox(
            "Rows",
            [5, 10, 15, 20],
            index=1,
            key="finbert_example_limit",
        )
    if example_sector != "All sectors":
        examples = examples.loc[examples["sector"].eq(example_sector)]
    if example_type != "All types":
        examples = examples.loc[examples["sampling_stratum"].eq(example_type)]
    if example_year != "All years":
        examples = examples.loc[examples["year"].eq(int(example_year))]
    examples = examples.sort_values(
        ["date", "sector", "ticker", "text_raw"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).head(int(example_limit))
    examples["disagreement_type"] = examples["sampling_stratum"].map(disagreement_labels)
    display_examples = examples[
        [
            "date",
            "ticker",
            "sector",
            "text_raw",
            "vader_compound",
            "vader_label",
            "finbert_score",
            "finbert_label",
            "disagreement_type",
        ]
    ]
    st.dataframe(
        display_examples,
        width="stretch",
        height=max(155, min(520, 36 * len(display_examples) + 42)),
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "ticker": st.column_config.TextColumn("Ticker"),
            "sector": st.column_config.TextColumn("Sector"),
            "text_raw": st.column_config.TextColumn("Headline", width="large"),
            "vader_compound": st.column_config.NumberColumn(
                "VADER score",
                format="%.3f",
                help="Lexicon-based compound headline score from -1 to +1.",
            ),
            "vader_label": st.column_config.TextColumn("VADER label"),
            "finbert_score": st.column_config.NumberColumn(
                "FinBERT score",
                format="%.3f",
                help="Positive probability minus negative probability, from -1 to +1.",
            ),
            "finbert_label": st.column_config.TextColumn("FinBERT label"),
            "disagreement_type": st.column_config.TextColumn("Difference", width="large"),
        },
    )
    if display_examples.empty:
        st.caption("No disagreement examples match the selected filters.")


def _render_sentiment(
    artifacts: AppArtifacts,
    finbert_artifacts: FinBERTAppArtifacts | None,
) -> None:
    _page_intro(
        "Sector Sentiment",
        "Monitor headline tone across US equity sectors and compare sentiment models.",
    )
    vader_tab, robustness_tab = st.tabs(
        ["Sector Overview", "Model Comparison"],
        key="sentiment_model_tab",
        on_change="rerun",
    )
    if vader_tab.open:
        with vader_tab:
            _render_vader_sentiment(artifacts)
    if robustness_tab.open:
        with robustness_tab:
            if finbert_artifacts is None:
                st.info(
                    "Model comparison data is temporarily unavailable. "
                    "Sector Overview remains available."
                )
            else:
                _render_neural_sentiment(artifacts, finbert_artifacts)


def _render_css() -> None:
    """Apply a restrained professional-finance presentation layer."""
    st.markdown(
        """
<style>
    :root {
        --pf-ink: #172b3f;
        --pf-muted: #637282;
        --pf-teal: #0f6ea8;
        --pf-teal-soft: #edf5fa;
        --pf-navy: #12324d;
        --pf-border: #d9e1e8;
        --pf-panel: #ffffff;
    }

    .stApp { color: var(--pf-ink); background: #ffffff; }
    .block-container {
        max-width: 1360px;
        padding-top: 1.1rem;
        padding-bottom: 2.5rem;
    }
    [data-testid="stHeader"] { background: transparent; height: 1.65rem; }
    [data-testid="stToolbar"] { display: none; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stSidebar"] {
        border-right: 1px solid #203b53;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #dfe8f0;
    }
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.35rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 0.55rem 0.65rem;
        border-radius: 0.5rem;
        transition: background 120ms ease, transform 120ms ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.07);
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: rgba(48, 197, 164, 0.16);
        box-shadow: inset 3px 0 0 #30c5a4;
    }
    .pf-side-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.4rem 0 1.35rem 0;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #203b53;
    }
    .pf-side-mark {
        display: grid;
        place-items: center;
        width: 2.35rem;
        height: 2.35rem;
        border-radius: 0.55rem;
        background: #30c5a4;
        color: #082033;
        font-weight: 800;
        letter-spacing: -0.04em;
    }
    .pf-side-name { color: #ffffff; font-size: 1.05rem; font-weight: 750; }
    .pf-side-sub { color: #9db0bf; font-size: 0.72rem; letter-spacing: 0.08em; }
    .pf-side-status {
        margin-top: 1.2rem;
        padding: 0.85rem;
        border: 1px solid #29455c;
        border-radius: 0.55rem;
        background: rgba(255,255,255,0.035);
        color: #c9d6e0;
        font-size: 0.78rem;
        line-height: 1.6;
    }
    .pf-side-status strong { color: #ffffff; }
    .pf-status-dot {
        display: inline-block;
        width: 0.48rem;
        height: 0.48rem;
        margin-right: 0.35rem;
        border-radius: 50%;
        background: #30c5a4;
        box-shadow: 0 0 0 3px rgba(48,197,164,0.13);
    }

    .pf-brand-kicker,
    .pf-eyebrow,
    .pf-section-label {
        margin: 0 0 0.3rem 0 !important;
        color: var(--pf-teal) !important;
        font-size: 0.71rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .pf-page-copy {
        max-width: 820px;
        margin: -0.25rem 0 1rem 0 !important;
        color: var(--pf-muted) !important;
        font-size: 0.93rem !important;
        line-height: 1.5;
    }
    h1, h2, h3, h4 { letter-spacing: -0.025em; color: var(--pf-ink); }
    h1 { font-size: clamp(2.15rem, 4vw, 3.15rem) !important; margin-bottom: 0.15rem !important; }
    h2 { margin-top: 0.55rem !important; font-size: 1.72rem !important; }
    h3 { margin-top: 1.1rem !important; font-size: 1.12rem !important; }
    h4 { font-size: 1rem !important; }
    .pf-market-masthead {
        display: flex;
        align-items: baseline;
        gap: 0.75rem;
        min-height: 2.25rem;
        padding: 0.1rem 0;
    }
    .pf-wordmark {
        color: var(--pf-navy);
        font-size: 1.32rem;
        font-weight: 800;
        letter-spacing: -0.035em;
    }
    .pf-product-label {
        color: var(--pf-muted);
        font-size: 0.79rem;
        font-weight: 650;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    [data-testid="stSegmentedControl"] {
        margin: 0.35rem 0 0.9rem 0;
        padding: 0.18rem 0;
        border-top: 1px solid var(--pf-border);
        border-bottom: 1px solid var(--pf-border);
    }
    [data-testid="stSegmentedControl"] button {
        min-height: 2.35rem;
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        color: var(--pf-muted) !important;
        font-size: 0.88rem;
        font-weight: 650;
    }
    [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        color: var(--pf-navy) !important;
        box-shadow: inset 0 -2px 0 var(--pf-teal);
    }
    .pf-inline-note {
        margin: -0.2rem 0 0.9rem 0 !important;
        color: var(--pf-muted) !important;
        font-size: 0.82rem !important;
    }
    .pf-section-label {
        margin-top: 0.8rem !important;
        color: var(--pf-ink) !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.02em;
        text-transform: none;
    }
    .pf-asof-card {
        padding: 0.9rem 1rem;
        border: 1px solid var(--pf-border);
        border-radius: 0.65rem;
        background: var(--pf-panel);
        box-shadow: 0 7px 22px rgba(11, 31, 51, 0.055);
    }
    .pf-asof-card strong,
    .pf-asof-card small { display: block; }
    .pf-asof-card strong { margin-top: 0.25rem; color: var(--pf-ink); }
    .pf-asof-card small { margin-top: 0.2rem; color: var(--pf-muted); }
    .pf-asof-label {
        color: var(--pf-teal);
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }
    .pf-live-dot {
        display: inline-block;
        width: 0.45rem;
        height: 0.45rem;
        margin-right: 0.35rem;
        border-radius: 50%;
        background: var(--pf-teal);
    }
    .pf-context-chip {
        margin-bottom: 0.02rem;
        padding: 0.55rem 0.7rem;
        border-left: 2px solid var(--pf-teal);
        background: #f5f8fa;
        color: var(--pf-muted);
        font-size: 0.72rem;
    }
    .pf-context-chip strong { color: var(--pf-ink); }
    .pf-chart-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem 0.7rem;
        margin: 0.15rem 0 0.15rem 0;
        padding: 0.5rem 0.1rem;
        border-bottom: 1px solid #e7ecf0;
        background: transparent;
    }
    .pf-chart-legend-item {
        display: inline-flex;
        flex: 1 1 165px;
        align-items: flex-start;
        min-width: 0;
        color: var(--pf-muted);
        font-size: 0.74rem;
        font-weight: 600;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }
    .pf-chart-legend-dot {
        flex: 0 0 auto;
        width: 0.58rem;
        height: 0.58rem;
        margin: 0.22rem 0.42rem 0 0;
        border-radius: 50%;
        box-shadow: 0 0 0 2px #ffffff;
    }
    .pf-coverage-period {
        margin: 0.35rem 0 0.8rem 0;
        padding: 0.8rem 0.9rem;
        border-left: 3px solid var(--pf-teal);
        background: #f7f9fb;
        line-height: 1.45;
    }
    .pf-coverage-period small {
        display: block;
        margin-bottom: 0.28rem;
        color: var(--pf-muted);
        font-size: 0.68rem;
        letter-spacing: 0.07em;
    }
    .pf-coverage-period strong { color: var(--pf-ink); }
    .pf-coverage-period span { margin: 0 0.4rem; color: var(--pf-muted); }

    [data-testid="stMetric"] {
        min-height: 82px;
        padding: 0.72rem 0.82rem;
        border: 1px solid var(--pf-border);
        border-radius: 0.18rem;
        background: var(--pf-panel);
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--pf-muted);
        font-size: 0.76rem;
        font-weight: 650;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--pf-ink);
        font-size: clamp(1.3rem, 1.8vw, 1.72rem);
        font-weight: 720;
        letter-spacing: -0.03em;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--pf-border) !important;
        border-radius: 0.18rem !important;
        background: var(--pf-panel);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--pf-border);
        border-radius: 0.18rem;
        overflow: hidden;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid var(--pf-border) !important;
        border-radius: 0.24rem !important;
        box-shadow: inset 0 -1px 0 #f3f6f8;
        min-height: 2.55rem;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:last-child {
        border-left: 1px solid var(--pf-border);
        padding-left: 0.55rem;
    }
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stMultiSelect"] > div > div {
        background: #ffffff;
    }
    [data-testid="stAlert"] {
        border-radius: 0.18rem;
        border-width: 1px;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 1.4rem;
        border-bottom: 1px solid var(--pf-border);
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        padding-left: 0;
        padding-right: 0;
        font-weight: 650;
    }
    hr { border-color: var(--pf-border) !important; }
    footer { visibility: hidden; }

    @media (max-width: 900px) {
        .block-container { padding: 0.7rem 1rem 2rem 1rem; }
        .pf-asof-card { margin-top: 0.65rem; }
        .pf-market-masthead { padding-top: 0.25rem; }
        [data-testid="stMetric"] { min-height: 76px; }
        [data-testid="stSegmentedControl"] button { font-size: 0.78rem; }
        .pf-chart-legend-item { flex-basis: 100%; }
    }
</style>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="PortFoYou | Fund Analytics",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _render_css()

    try:
        artifacts = _load_artifacts()
    except AppArtifactError as exc:
        st.error(f"Core market research data is unavailable.\n\n{exc}")
        st.info("Rebuild the core project artifacts, then reload this page.")
        st.stop()

    page = _render_app_header(artifacts)

    if page == "Fund Screener":
        _render_compare(artifacts)
    elif page == "Fund Profile":
        _render_fact_sheet(artifacts)
    elif page == "Fund Allocation":
        _render_fund_allocation(artifacts)
    elif page == "Portfolio Simulator":
        try:
            asset_returns = _load_asset_returns()
        except CustomPortfolioError as exc:
            _page_intro(
                "Portfolio Simulator",
                "Build a fund-like portfolio from individual US stocks and "
                "cryptoassets, then inspect its weighted historical behaviour.",
            )
            st.error(f"Portfolio Simulator data is unavailable.\n\n{exc}")
            st.info("Rebuild the individual-asset artifact, then reload this page.")
        else:
            _render_custom_portfolio(asset_returns)
    else:
        try:
            finbert_artifacts = _load_finbert_artifacts(artifacts.sector_sentiment)
        except AppArtifactError:
            finbert_artifacts = None
        _render_sentiment(artifacts, finbert_artifacts)


if __name__ == "__main__":
    main()
