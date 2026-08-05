# Prompt log - active sector allocation

## What I wanted

Add one clearly labelled exploratory multi-asset fund that is materially
different from the existing equal-weight and minimum-variance funds, without
changing their calculations or selecting a parameter from the final test period.

## Prompt(s)

"On the existing precomputed fund universe, add a monthly active allocation with
a 70% sector inverse-volatility equity core, a 20% satellite split across the
two highest already-lagged sector-sentiment signals, and a 10% equally weighted
crypto sleeve. Use a 252-trading-day historical volatility lookback, stay
long-only and fully invested, cap sector, stock, and crypto-asset weights at
20%, 5%, and 2%, and keep transaction costs at zero. Preserve every existing
fund and show the new fund in the app and evidence."

## What the assistant produced

It added `src/active_sector.py`, integrated the new
`combined_active_sector_allocation` fund into `scripts/run_part_b.py`, generated
an exposure audit, extended the app labels/fact sheet and report evidence, and
added focused regression tests. The fixed strategy uses 70% inverse-volatility
sector allocations, 20% across two ranked sectors, and 10% equally across the
ten cryptoassets.

## What was wrong or risky

The first version assumed all ten tradable sector z-scores were present at every
monthly decision. The supplied index has two sector-decision cells with no usable
current z-score. Replacing those with a neutral zero or a future observation
would silently alter the intended signal or introduce look-ahead. It also needed
an explicit check that the stated 252-day volatility window was actually rolling,
not the build's general expanding history.

## What I changed and why

I required a fixed rolling 252-day return window for the sector risk calculation.
For a missing current tradable signal, the build carries forward only that
sector's most recent prior usable, already-lagged value and records
`signal_was_missing=True` in the published weight log. This maintains the
strictly earlier information set. I checked the fixed 70/20/10 sleeves, caps,
strict lag, three direct return reconstructions, future-signal invariance, actual
CSV contracts, app artifact loading, and figure QA. The strategy remains an
exploratory design, not evidence that sentiment predicts returns.
