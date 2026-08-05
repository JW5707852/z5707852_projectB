# Prompt log - Fund Screener chart-control layout

## What I wanted

Fix the Fund Screener common-period chart so the 1Y, 2Y, and All range controls
do not obscure the chart's sample-period disclosure.

## Prompt(s)

"fund screener这个部分icon会被时间说明挡住，修复该问题"

## What the assistant produced

It identified that the Plotly title and range selector were both positioned in
the chart's top margin. It increased the dedicated top margin, moved the title
to its own higher row, and anchored the range-selector controls immediately
below it. A chart test now locks these layout coordinates.

## What was wrong or risky

The controls overlapped the disclosure on the narrower deployed layout, making
the time filters difficult to read and use. Moving controls without a test
could allow a future chart-style update to reintroduce the collision.

## What I changed and why

Student confirmation pending: review the redeployed chart on the target screen
size and confirm that the separated title and control rows are visually clear.
