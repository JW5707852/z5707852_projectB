# Prompt log - balanced growth sector allocation

## What I wanted

Add a second predeclared risk-profile fund after the active-sector allocation:
80% defensive sector inverse-volatility equity core, 15% lagged-sentiment
satellite, and 5% crypto. Keep it separate from every existing fund.

## Prompt(s)

"Increase the earlier defensive design's offensive allocation by a further 10%,
but retain a controlled risk profile. Use 80% inverse-volatility equity core,
15% across the three highest already-lagged sector sentiment signals, and 5%
equally across cryptoassets. Maintain monthly rebalancing, 252-day volatility,
long-only fully invested weights, a 15% sector cap, 3% stock cap, 0.5% crypto
asset cap, and zero transaction costs."

## What the assistant produced

It reused the independently tested sector-allocation engine with a distinct,
fixed `combined_growth_sector_allocation` identity and method. It wrote a
separate exposure audit, included the fund in the published CSV artifacts,
evidence and Streamlit fact sheet, and added unit and actual-CSV tests.

## What was wrong or risky

Calling the fund "defensive" would be misleading after raising the active
satellite allocation to 15%. Adding the new fund also changes the hash of the
published core artifacts, so the existing FinBERT display metadata would no
longer accurately document the protected artifacts unless it was refreshed.

## What I changed and why

I labelled it "Balanced Growth Sector Allocation" and retained the strict
lagged-signal, rolling 252-day risk calculation. I regenerated the core build,
the report evidence, and the FinBERT post-processing metadata without rerunning
or retuning FinBERT. I verified the 80/15/5 sleeves, three selected sectors,
caps, CSV contracts, return reproduction, app loading, and visual QA. This is a
fixed exploratory risk-profile design, not proof that the sentiment signal
predicts returns.
