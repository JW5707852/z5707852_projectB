"""Create the deterministic blind FinBERT review file from the audit template."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.finbert_review import build_blind_review  # noqa: E402

AUDIT_PATH = PROJECT_ROOT / "results/tables/sentiment_manual_review_template.csv"
BLIND_PATH = PROJECT_ROOT / "results/tables/sentiment_manual_review_blind.csv"


def main() -> int:
    if not AUDIT_PATH.is_file():
        raise FileNotFoundError(f"manual-review audit template is missing: {AUDIT_PATH}")
    audit = pd.read_csv(AUDIT_PATH, keep_default_na=False)
    blind = build_blind_review(audit)
    descriptor, temporary_name = tempfile.mkstemp(prefix="finbert-blind-review-", suffix=".csv")
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        blind.to_csv(temporary_path, index=False)
        os.replace(temporary_path, BLIND_PATH)
        BLIND_PATH.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Wrote {len(blind)} blinded review rows to {BLIND_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
