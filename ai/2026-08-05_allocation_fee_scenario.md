# Prompt log - allocation fee scenario

## What I wanted

Improve only Fund Allocation with an optional annual management-fee scenario, while leaving published fund artifacts, backtests, and report evidence unchanged.

## Prompt(s)

"Add a 0.00% to 3.00% annual management-fee control to Fund Allocation. Apply it as daily compounding on the allocation’s 252-day common calendar, show gross and fee-adjusted outcomes, and test zero and positive fee cases without changing fund CSVs."

## What the assistant produced

It added a small pure fee-overlay helper and a Streamlit control with fee-adjusted metrics and chart output.

## What was wrong or risky

Applying a fee by simply subtracting annual_fee/252 from daily return would not match the required compounding convention. Applying the fee to fund artifacts would also incorrectly present a hypothetical investor scenario as fund history.

## What I changed and why

I use `(1 + gross_daily_return) * (1 - annual_fee) ** (1/252) - 1` only on a copy of the allocation scenario history. At 0%, the helper returns the gross daily returns exactly. The page labels the fee as hypothetical and states the omitted costs. I will check the tests and UI output before relying on it.
