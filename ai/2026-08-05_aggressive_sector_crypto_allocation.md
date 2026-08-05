# Prompt log - aggressive sector and crypto allocation

## What I wanted

Add an explicitly aggressive, but still constrained, risk-profile fund alongside
the existing defensive, balanced-growth, and active-sector options.

## Prompt(s)

"Add an aggressive fund with 50% inverse-volatility equity core, 30% divided
between the three highest already-lagged sector sentiment signals, and 20%
equally across cryptoassets. Use monthly rebalancing, a 252-day historical
volatility window, long-only fully invested weights, a 25% sector cap, 5% stock
cap, 2% cryptoasset cap, and zero transaction costs."

## What the assistant produced

It added `combined_aggressive_sector_allocation` with a separate method,
exposure audit, app fact sheet, report evidence, and focused tests. It reused
the already tested lagged-signal engine rather than duplicating portfolio logic.

## What was wrong or risky

The initial return-versus-risk figure used seven nearby arrow labels, which
overlapped after the aggressive fund was added. The first colour choice also
duplicated another fund's colour. Neither issue changed the calculations, but
both made the report exhibit unreliable.

## What I changed and why

I gave the aggressive fund a distinct colour and replaced the crowded scatter
annotations with a readable two-column legend that includes each fund's Sharpe
ratio. Figure QA then passed with no clipping or overlap. I regenerated the
core, evidence, and FinBERT display metadata, and checked the fixed 50/30/20
sleeves, caps, signal lag, actual artifacts, and app interaction. The fund is
labelled exploratory and high-growth; it is not a forecast or proof that
sentiment predicts returns.
