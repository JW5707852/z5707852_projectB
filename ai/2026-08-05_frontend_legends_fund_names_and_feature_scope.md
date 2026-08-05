# Prompt log - frontend legends, fund names and feature scope

## What I wanted

Improve chart-label visibility, replace unclear allocation labels with
professional fund names, and determine whether PortFoYou supports a portfolio
built from individually selected financial assets.

## Prompt(s)

"Modify the program: chart data names are crowded at the edge and many cannot be
seen; labels such as Combined 1/N (%) are unclear and should use professional
names; confirm whether a custom financial-asset portfolio builder exists and, if
not, write a prompt to build it."

## What the assistant produced

- Responsive HTML chart legends outside the Plotly canvas for fund comparison,
  risk-return, allocation, and sector-trend charts.
- Professional client-facing fund names while retaining the stable internal fund
  IDs and all precomputed results.
- Clearer allocation input labels and fund-description help text.
- A visible capability notice explaining that the current allocation journey is
  fund-level, not asset-level.
- A separate implementation prompt for a future custom asset portfolio builder.

## What was wrong or risky

The former horizontal Plotly legends were positioned below the plotting area, so
long labels could be clipped in narrow columns or mobile layouts. The label
"Combined 1/N" exposed methodology shorthand without identifying its asset scope.
The allocation page did not explicitly distinguish allocation across PortFoYou
funds from selecting individual stocks or cryptoassets.

## What I changed and why

The requested correction was applied only to presentation and product-scope copy:
fund identifiers, historical calculations, and published CSV values were not
changed. The future asset-builder prompt keeps asset returns as a reproducible
build-time artifact so the deployed app remains lightweight and look-ahead-safe.

