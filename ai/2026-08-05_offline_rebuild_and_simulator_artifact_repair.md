# Prompt log - offline rebuild and simulator artifact repair

## What I wanted

Repair the incomplete individual-asset return artifact and verify that the
entire project can rebuild from the supplied raw-data bundle without relying on
the live source during the build.

## Prompt(s)

"你解决问题1和2" (solve issues 1 and 2).

## What the assistant produced

- Downloaded the official provided data ZIP to a temporary location outside the
  submission folder and checked the ZIP contains all three expected Parquet
  files.
- Set `FINS_DATA_ZIP` only for the build and test commands, then ran
  `scripts/run_part_b.py` successfully.
- Regenerated `results/data/investable_asset_returns.csv`; BTC-USD now shares
  the common 2020-01-03 to 2023-12-29, 1,005-observation sample with the other
  selectable assets.
- Replaced one stale five-fund assertion in `tests/test_fusion.py` with the
  published fund-identity contract.

## What was wrong or risky

The prior individual-asset artifact had BTC-USD ending on 2021-01-28 while the
other selectable assets continued to 2023-12-29. The simulator correctly
rejected this mixed-sample file, so the UI was unavailable. The original full
test run also could not access the external hosts in this environment. A test
still assumed the pre-crypto five-fund family, producing a false regression
failure after a valid rebuild.

## What I changed and why

Student confirmation is required: confirm that retaining the downloaded source
ZIP outside the submission folder is appropriate and that the regenerated
artifacts are the intended submission outputs. The ZIP must not be committed.
The code change keeps the test aligned with the same published fund contract
used by the build, rather than hard-coding a historical subset.
