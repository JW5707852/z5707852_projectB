# Prompt log - investor-facing finance product UI refactor

## What I wanted

Replace research-oriented and unclear customer-facing terminology throughout the
app, reduce repeated branding, redesign the layout as a professional financial
product, and consolidate data sources, sample periods and methodology behind one
information control.

## Prompt(s)

"Check all similar data-statistic expressions, not only the five fund titles. This
is customer-facing software. Do not repeat PortFoYou on every page or make the
interface look like a presentation. Reference the layouts used by Eastmoney, Sina
Finance and Yahoo Finance. Put sources and sample dates behind a dedicated icon,
use financial terms familiar to investors, and add clickable explanations beside
specialist terms."

## What the assistant produced

- A compact masthead and horizontal navigation for Fund Screener, Fund Profile,
  Portfolio Simulator and Sector Sentiment.
- One visible PortFoYou wordmark instead of repeated page-level branding.
- A single **Data & methodology** information control containing coverage,
  provenance, calculation conventions, sentiment timing and risk disclosures.
- Investor-facing performance terms including cumulative return, annual return,
  annual volatility, Sharpe ratio, maximum drawdown, target weight, ending value
  and total return.
- Contextual help icons on non-obvious metrics, section headings and controls.
- Shorter tabs and control labels for sentiment analytics and model comparison.
- A flatter, denser visual system with thin separators, compact KPI cells,
  sortable tables and responsive top navigation.

## What was wrong or risky

The earlier interface mixed internal methodology language with customer actions,
used large headings and elevated cards that resembled presentation slides, repeated
the product name and research disclaimers, and scattered sample dates across pages.
Long strategy names could also be clipped in tables and charts. A denser layout can
create mobile overflow unless navigation, legends and tables are tested separately.

## What I changed and why

The student's explicit correction was applied across every active page and the
optional sentiment-model comparison view. Internal fund IDs, historical results,
allocation arithmetic and CSV contracts were not changed. The layout follows the
shared information architecture seen on mainstream finance sites: horizontal
product navigation, metrics first, tables for comparison, charts beneath the key
numbers, and methodology available on demand.

