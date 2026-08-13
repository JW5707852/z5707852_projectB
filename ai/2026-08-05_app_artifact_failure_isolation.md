# Prompt log - app artifact failure isolation

## What I wanted

Keep PortFoYou's required fund, allocation, and sentiment pages available when
the optional individual-asset simulator artifact is missing or malformed, while
showing a useful project-relative error message.

## Prompt(s)

Check the current application integrity

After the audit identified the failure-isolation issue: "accept"

## What the assistant produced

The app now loads the four core artifacts before rendering navigation, loads
`investable_asset_returns.csv` only when Portfolio Simulator is selected, and
reports the relevant relative artifact path when either contract fails. Focused
Streamlit regression tests cover both the isolated optional failure and the core
artifact error message.

## What was wrong or risky

The previous startup path loaded the optional simulator artifact globally. A
missing or malformed simulator CSV therefore stopped every app page, including
pages that did not depend on it. The shared generic error also hid the failing
file or schema detail.

## What I changed and why

The student explicitly accepted the scoped audit finding. The correction keeps
core and optional artifact failures separate so one optional feature cannot make
the required investor journey unavailable, while retaining clear and safe
project-relative diagnostics.
