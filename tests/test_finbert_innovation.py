"""Pure Phase 1 tests for pinned FinBERT inference contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from src import finbert_innovation


class FakeTokenizer:
    def __init__(self) -> None:
        self.inference_text_count = 0

    def __call__(self, texts, **kwargs):
        values = list(texts)
        if kwargs.get("return_length"):
            return {"length": [len(text.split()) + 2 for text in values]}
        self.inference_text_count += len(values)
        identifiers = [1 if "profit" in text else 2 for text in values]
        return {
            "input_ids": torch.tensor([[identifier, 0] for identifier in identifiers]),
            "attention_mask": torch.ones((len(values), 2), dtype=torch.long),
        }


class FakeModel:
    def __init__(self) -> None:
        self.rows_seen = 0
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, input_ids, attention_mask):
        del attention_mask
        self.rows_seen += len(input_ids)
        rows = []
        for identifier in input_ids[:, 0].tolist():
            rows.append([3.0, 1.0, 0.0] if identifier == 1 else [0.0, 3.0, 1.0])
        return SimpleNamespace(logits=torch.tensor(rows, dtype=torch.float32))


def _loaded(tokenizer=None, model=None) -> finbert_innovation.LoadedFinBERT:
    return finbert_innovation.LoadedFinBERT(
        tokenizer=tokenizer or FakeTokenizer(),
        model=model or FakeModel(),
        device=torch.device("cpu"),
        resolved_revision=finbert_innovation.MODEL_REVISION,
        id2label={"0": "positive", "1": "negative", "2": "neutral"},
    )


def test_label_mapping_is_exact_and_wrong_order_is_rejected() -> None:
    assert (
        finbert_innovation.normalise_label_mapping(
            {"0": "positive", "1": "negative", "2": "neutral"}
        )
        == finbert_innovation.EXPECTED_ID2LABEL
    )
    with pytest.raises(finbert_innovation.FinBERTValidationError, match="mapping"):
        finbert_innovation.normalise_label_mapping({0: "negative", 1: "positive", 2: "neutral"})


def test_probability_validation_and_score_formula() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.05, 0.85, 0.10]])
    actual = finbert_innovation.probabilities_to_scores(
        probabilities, finbert_innovation.EXPECTED_ID2LABEL
    )
    np.testing.assert_allclose(
        actual["finbert_score"],
        actual["probability_positive"] - actual["probability_negative"],
        atol=1e-15,
        rtol=0.0,
    )
    assert actual["finbert_label"].tolist() == ["positive", "neutral", "negative"]
    with pytest.raises(finbert_innovation.FinBERTValidationError, match="sum"):
        finbert_innovation.validate_probabilities(np.array([[0.5, 0.5, 0.5]]))
    with pytest.raises(finbert_innovation.FinBERTValidationError, match="finite"):
        finbert_innovation.validate_probabilities(np.array([[np.nan, 0.5, 0.5]]))


def test_duplicate_headlines_are_inferred_once_and_truncation_is_counted() -> None:
    titles = [
        "profit rises",
        "profit rises",
        "loss widens sharply after earnings announcement",
    ]
    panel = pd.DataFrame({"title": titles, "text_raw": titles})
    tokenizer = FakeTokenizer()
    model = FakeModel()
    actual = finbert_innovation.infer_distinct_titles(
        panel,
        _loaded(tokenizer, model),
        batch_size=2,
        max_length=5,
    )
    assert len(actual.title_scores) == 2
    assert tokenizer.inference_text_count == 2
    assert model.rows_seen == 2
    assert model.eval_called
    assert actual.metrics.truncated_title_count == 1
    assert actual.metrics.truncated_title_percentage == pytest.approx(50.0)
    probability_columns = [
        "probability_positive",
        "probability_negative",
        "probability_neutral",
    ]
    np.testing.assert_allclose(
        actual.title_scores[probability_columns].sum(axis=1),
        1.0,
        atol=1e-6,
        rtol=0.0,
    )


def test_pilot_selection_is_reproducible_and_unique() -> None:
    text = [f"headline {number}" for number in range(30)]
    panel = pd.DataFrame({"text_raw": text + text[:5]})
    first = finbert_innovation.deterministic_pilot_sample(panel, sample_size=12, seed=5545)
    second = finbert_innovation.deterministic_pilot_sample(panel, sample_size=12, seed=5545)
    assert first.equals(second)
    assert len(first) == 12
    assert not first["text_raw"].duplicated().any()


def test_device_and_external_cache_guards(tmp_path) -> None:
    assert str(finbert_innovation.select_device("cpu")) == "cpu"
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(finbert_innovation.FinBERTValidationError, match="outside"):
        finbert_innovation.validate_external_cache(project / "model-cache", project)
    external = tmp_path / "external-cache"
    assert finbert_innovation.validate_external_cache(external, project) == external
