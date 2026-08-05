"""Opt-in real-model smoke test for the pinned ProsusAI/finbert revision."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src import finbert_innovation

RUN_REAL_MODEL = os.environ.get("RUN_FINBERT_SMOKE") == "1"


@pytest.mark.skipif(not RUN_REAL_MODEL, reason="set RUN_FINBERT_SMOKE=1")
def test_pinned_finbert_model_smoke() -> None:
    cache = Path(
        os.environ.get(
            "FINBERT_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "portfoyou-huggingface"),
        )
    )
    loaded = finbert_innovation.load_pinned_finbert(
        cache_dir=cache,
        device=os.environ.get("FINBERT_DEVICE", "auto"),
    )
    assert loaded.resolved_revision == finbert_innovation.MODEL_REVISION
    assert loaded.id2label == {0: "positive", 1: "negative", 2: "neutral"}

    text = [
        "Company reports record profit and raises its outlook.",
        "Company warns that losses widened sharply.",
        "The board schedules its annual meeting for next month.",
        "Shares rise despite weaker quarterly revenue.",
        "Debt falls, but management does not increase guidance.",
    ]
    panel = pd.DataFrame({"title": text, "text_raw": text})
    inference = finbert_innovation.infer_distinct_titles(
        panel,
        loaded,
        batch_size=5,
        max_length=finbert_innovation.DEFAULT_MAX_LENGTH,
    )
    scores = inference.title_scores
    probabilities = scores[
        ["probability_positive", "probability_negative", "probability_neutral"]
    ].to_numpy()
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(
        scores["finbert_score"],
        scores["probability_positive"] - scores["probability_negative"],
        atol=1e-15,
        rtol=0.0,
    )
    assert set(scores["finbert_label"]).issubset({"positive", "negative", "neutral"})
