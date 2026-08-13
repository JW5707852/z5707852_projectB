# Prompt log - professional PortFoYou frontend redesign

## What I wanted

Make the PortFoYou Streamlit investor journey more attractive and professional
by learning from established financial research products, while retaining the
existing calculations, artifacts, fund identities, and assessment scope.

## Prompt(s)

> Referencing professional financial software to make the entire front-end interface
  more visually appealing and professional.

Earlier interface feedback from the student also stated that internal CSV paths
were unnecessary and unsuitable for a client-facing application.

## What the assistant produced

- Audited all five Streamlit views in a local browser before changing them.
- Used the information hierarchy common to professional fund research products:
  a restrained finance color system, standardised comparison metrics, clear
  as-of dates, risk-return context, fund-level performance and holdings, and
  prominent research limitations.
- Reworked the app shell with a dark research sidebar, a compact product header,
  publication status, responsive cards, and consistent spacing.
- Replaced generic Streamlit charts with presentation-only Plotly figures for
  growth, risk-return comparison, drawdown, holdings, allocation, and sentiment.
- Added a ten-sector sentiment heatmap while retaining a focused selectable trend
  view and the complete sector coverage table.
- Reorganised allocation inputs into a compact lab and added a historical value
  path, allocation donut, and typed scenario table.
- Kept all calculations based on the same four precomputed Part B artifacts; no
  backtest, optimizer, or sentiment model was moved into the deployed app.
- Added focused chart-contract and AppTest coverage.

## What was wrong or risky

- The first Plotly draft used axis titles and margins that clipped tick labels in
  narrow chart columns. The figures were revised and visually checked again.
- The original holdings display filled every missing sector value with
  `Crypto / not applicable`. Combined-fund equity rows also lack a stored sector,
  so this mislabeled equities. The display now uses the `-USD` ticker suffix only
  to distinguish missing-sector crypto rows from missing-sector equity rows.
- Five KPI cards can truncate long labels or values on narrower screens. Labels
  and date formats were shortened, and metric typography was reduced slightly.
- Plotly is now a deployed app dependency, so it was added explicitly to
  `requirements.txt` rather than relying on the local environment.
- The visual references informed hierarchy and interaction patterns only. No
  external brand assets, proprietary data, screenshots, or code were copied.

## What I changed and why

I confirmed that internal artifact paths, repeated provenance blocks, and
research-oriented labels should not appear throughout the client interface. I
kept the compact finance-dashboard hierarchy, investor-facing metric names,
responsive legends, and contextual help; I rejected repeated branding, large
presentation-style cards, and unexplained method shorthand because the product
is intended for individual investors with basic financial literacy.
