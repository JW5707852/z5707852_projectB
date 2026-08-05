# Prompt log - professional PortFoYou frontend redesign

## What I wanted

Make the PortFoYou Streamlit investor journey more attractive and professional
by learning from established financial research products, while retaining the
existing calculations, artifacts, fund identities, and assessment scope.

## Prompt(s)

> 参考专业的财经软件，使整个前端界面更美观，更专业

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

The student has already confirmed that internal artifact paths should not appear
in the client interface. Review of the broader redesign is still pending. Before
submission, the student should record here which visual and wording choices they
kept, changed, or rejected, and why those choices fit PortFoYou's intended user.
