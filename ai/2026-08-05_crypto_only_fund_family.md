# Prompt log - crypto-only fund family

## What I wanted

Add two crypto-only PortFoYou funds without changing the existing combined or equity fund calculations: equal weight and long-only minimum variance, evaluated on crypto's own daily calendar.

## Prompt(s)

"Add a separate crypto-only fund family with equal-weight and minimum-variance funds. Use the supplied ten coins, the native daily calendar, a 252-observation expanding estimation window, monthly decisions, the next crypto return date for holdings, 365 annualisation, 0% risk-free rate, and 0 bps transaction costs. Preserve the existing non-crypto artifacts and update validation, app, and report evidence."

## What the assistant produced

It proposed a separate crypto builder, per-fund calendar validation, per-family annualisation, a 365-day fact-sheet disclosure, and intersection-based cross-family comparison paths. It also added focused timing, constraint, return-reproduction, holdings-date, and regression-hash checks.

## What was wrong or risky

The pre-existing validation assumed every fund shared the equity calendar and hard-coded 252 periods. Reusing that path would remove weekend crypto returns. A cross-family growth chart that simply plotted each native wealth series would not compare the same holding period.

## What I changed and why

I kept the crypto panel separate until after its crypto-native backtests complete. I changed validation to require sorted, unique dates within each fund and 365 periods only for the crypto family. The comparison chart now rebases all displayed funds on their explicit shared-date intersection; fact sheets retain each fund's native history. I will verify the generated artifacts and numerical outputs before using any results or interpretation in the report.
