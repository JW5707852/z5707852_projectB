"""Build-only, pinned FinBERT inference for headline sentiment robustness.

This module is independent of the deployed Streamlit application. It performs
real pretrained-model inference on unchanged ``text_raw`` headlines and keeps
all model, probability, device, and truncation decisions auditable.
"""

from __future__ import annotations

import importlib.metadata
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src import etl

MODEL_NAME = "ProsusAI/finbert"
MODEL_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
MODEL_LICENSE = "Apache-2.0"
EXPECTED_ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}
TEXT_INPUT_COLUMN = "text_raw"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 128
PILOT_RANDOM_SEED = 5545
PROBABILITY_TOLERANCE = 1e-5


class FinBERTValidationError(RuntimeError):
    """Raised when model provenance or inference output is inconsistent."""


@dataclass(frozen=True)
class LoadedFinBERT:
    """Pinned tokenizer, model, and verified runtime provenance."""

    tokenizer: Any
    model: Any
    device: torch.device
    resolved_revision: str
    id2label: dict[int, str]


@dataclass(frozen=True)
class InferenceMetrics:
    """Observed compute and truncation evidence for one inference run."""

    distinct_title_count: int
    batch_size: int
    max_length: int
    device: str
    inference_seconds: float
    titles_per_second: float
    truncated_title_count: int
    truncated_title_percentage: float
    maximum_observed_token_length: int
    peak_rss_mb: float


@dataclass(frozen=True)
class FinBERTInference:
    """One score row per distinct title plus benchmark evidence."""

    title_scores: pd.DataFrame
    metrics: InferenceMetrics


def process_peak_rss_mb() -> float:
    """Return process peak resident memory in MiB on macOS or Linux."""
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def package_versions() -> dict[str, str]:
    """Return exact build-only dependency versions used by inference."""
    return {
        "transformers_version": importlib.metadata.version("transformers"),
        "huggingface_hub_version": importlib.metadata.version("huggingface-hub"),
        "torch_version": importlib.metadata.version("torch"),
    }


def normalise_label_mapping(id2label: Any) -> dict[int, str]:
    """Validate the pinned model's label IDs without relying on key types."""
    if not isinstance(id2label, dict):
        raise FinBERTValidationError("FinBERT id2label must be a dictionary")
    try:
        normalised = {int(key): str(value).strip().lower() for key, value in id2label.items()}
    except (TypeError, ValueError) as exc:
        raise FinBERTValidationError("FinBERT id2label contains invalid keys") from exc
    if normalised != EXPECTED_ID2LABEL:
        raise FinBERTValidationError(
            "FinBERT label mapping differs from the pinned contract: "
            f"expected {EXPECTED_ID2LABEL}, received {normalised}"
        )
    return normalised


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA, MPS, or CPU without silently changing an explicit request."""
    choice = str(requested).strip().lower()
    if choice not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, mps, cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise FinBERTValidationError("CUDA was requested but is not available")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise FinBERTValidationError("MPS was requested but is not available")
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_external_cache(cache_dir: Path, project_root: Path) -> Path:
    """Reject a Hugging Face cache placed inside the submission folder."""
    cache = Path(cache_dir).expanduser().resolve()
    root = Path(project_root).resolve()
    if cache == root or cache.is_relative_to(root):
        raise FinBERTValidationError("Hugging Face cache must remain outside the submission folder")
    return cache


def load_pinned_finbert(
    *,
    cache_dir: Path,
    device: str = "auto",
) -> LoadedFinBERT:
    """Load the exact approved model revision and verify its label mapping."""
    from huggingface_hub import model_info
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    selected_device = select_device(device)
    try:
        info = model_info(MODEL_NAME, revision=MODEL_REVISION)
        resolved_revision = str(info.sha)
        if resolved_revision != MODEL_REVISION:
            raise FinBERTValidationError(
                f"Resolved FinBERT revision differs from the pinned revision: {resolved_revision}"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            cache_dir=str(cache_dir),
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            cache_dir=str(cache_dir),
        )
    except FinBERTValidationError:
        raise
    except Exception as exc:
        raise FinBERTValidationError(
            f"Could not load pinned ProsusAI/finbert revision {MODEL_REVISION}: {exc}"
        ) from exc

    id2label = normalise_label_mapping(model.config.id2label)
    try:
        model.to(selected_device)
        model.eval()
    except Exception as exc:
        raise FinBERTValidationError(
            f"Could not initialise FinBERT on device {selected_device}: {exc}"
        ) from exc
    return LoadedFinBERT(
        tokenizer=tokenizer,
        model=model,
        device=selected_device,
        resolved_revision=resolved_revision,
        id2label=id2label,
    )


def _validate_text_input(panel: pd.DataFrame) -> None:
    etl.require_columns(panel, {TEXT_INPUT_COLUMN}, "FinBERT headline panel")
    if panel[TEXT_INPUT_COLUMN].isna().any():
        raise FinBERTValidationError("text_raw contains missing titles")
    if not panel[TEXT_INPUT_COLUMN].map(lambda value: isinstance(value, str)).all():
        raise FinBERTValidationError("text_raw must contain unchanged strings")
    if "title" in panel and not panel["title"].equals(panel[TEXT_INPUT_COLUMN]):
        raise FinBERTValidationError("text_raw no longer matches title")


def distinct_titles(panel: pd.DataFrame) -> pd.DataFrame:
    """Return each identical, unchanged headline exactly once in stable order."""
    _validate_text_input(panel)
    titles = panel[[TEXT_INPUT_COLUMN]].drop_duplicates(keep="first").reset_index(drop=True)
    if titles.duplicated(TEXT_INPUT_COLUMN).any():
        raise FinBERTValidationError("distinct-title table is not unique")
    return titles


def deterministic_pilot_sample(
    panel: pd.DataFrame,
    *,
    sample_size: int,
    seed: int = PILOT_RANDOM_SEED,
) -> pd.DataFrame:
    """Select a reproducible random subset of distinct titles for benchmarking."""
    titles = distinct_titles(panel)
    if sample_size < 1 or sample_size > len(titles):
        raise ValueError("sample_size must be between one and the distinct-title count")
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(titles), size=sample_size, replace=False))
    return titles.iloc[positions].reset_index(drop=True)


def validate_probabilities(
    probabilities: np.ndarray,
    *,
    tolerance: float = PROBABILITY_TOLERANCE,
) -> np.ndarray:
    """Validate finite three-class probabilities and their row sums."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise FinBERTValidationError(
            f"FinBERT probabilities must have shape (n, 3), received {values.shape}"
        )
    if not np.isfinite(values).all():
        raise FinBERTValidationError("FinBERT probabilities contain non-finite values")
    if (values < 0.0).any() or (values > 1.0).any():
        raise FinBERTValidationError("FinBERT probabilities fall outside [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=tolerance, rtol=0.0):
        raise FinBERTValidationError("FinBERT probabilities do not sum to one")
    return values


def probabilities_to_scores(
    probabilities: np.ndarray,
    id2label: dict[int, str],
) -> pd.DataFrame:
    """Convert verified softmax probabilities to the approved score and label."""
    mapping = normalise_label_mapping(id2label)
    values = validate_probabilities(probabilities)
    positive_id = next(key for key, label in mapping.items() if label == "positive")
    negative_id = next(key for key, label in mapping.items() if label == "negative")
    neutral_id = next(key for key, label in mapping.items() if label == "neutral")
    winning_ids = values.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "probability_positive": values[:, positive_id],
            "probability_negative": values[:, negative_id],
            "probability_neutral": values[:, neutral_id],
            "finbert_score": values[:, positive_id] - values[:, negative_id],
            "finbert_label": [mapping[int(index)] for index in winning_ids],
        }
    )
    expected_score = frame["probability_positive"] - frame["probability_negative"]
    if not np.allclose(frame["finbert_score"], expected_score, atol=1e-15, rtol=0.0):
        raise FinBERTValidationError("FinBERT score formula is inconsistent")
    return frame


def infer_distinct_titles(
    panel: pd.DataFrame,
    loaded: LoadedFinBERT,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> FinBERTInference:
    """Infer one FinBERT probability vector per distinct unchanged title."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_length < 2:
        raise ValueError("max_length must be at least two tokens")
    mapping = normalise_label_mapping(loaded.id2label)
    titles = distinct_titles(panel)
    all_probabilities: list[np.ndarray] = []
    truncated_count = 0
    maximum_observed_length = 0
    started = time.perf_counter()

    loaded.model.eval()
    with torch.inference_mode():
        for start in range(0, len(titles), batch_size):
            stop = min(start + batch_size, len(titles))
            texts = titles[TEXT_INPUT_COLUMN].iloc[start:stop].tolist()
            try:
                untruncated = loaded.tokenizer(
                    texts,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                    return_length=True,
                )
                lengths = np.asarray(untruncated["length"], dtype=int)
                if lengths.size:
                    truncated_count += int((lengths > max_length).sum())
                    maximum_observed_length = max(maximum_observed_length, int(lengths.max()))
                encoded = loaded.tokenizer(
                    texts,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {name: tensor.to(loaded.device) for name, tensor in encoded.items()}
                logits = loaded.model(**encoded).logits
                if logits.ndim != 2 or logits.shape != (len(texts), 3):
                    raise FinBERTValidationError(
                        "FinBERT logits have unexpected shape "
                        f"{tuple(logits.shape)} for rows {start}:{stop}"
                    )
                probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                all_probabilities.append(validate_probabilities(probabilities))
            except FinBERTValidationError:
                raise
            except Exception as exc:
                raise FinBERTValidationError(
                    f"FinBERT inference failed on device {loaded.device} for "
                    f"rows {start}:{stop}: {exc}"
                ) from exc
            if progress_callback is not None:
                progress_callback(stop, len(titles), time.perf_counter() - started)

    duration = time.perf_counter() - started
    if duration <= 0.0:
        raise FinBERTValidationError("FinBERT inference duration is not positive")
    combined = validate_probabilities(np.vstack(all_probabilities))
    if len(combined) != len(titles):
        raise FinBERTValidationError("FinBERT inference changed distinct-title count")
    scored = pd.concat([titles, probabilities_to_scores(combined, mapping)], axis=1)
    if scored.duplicated(TEXT_INPUT_COLUMN).any():
        raise FinBERTValidationError("FinBERT title score cache is not unique")
    metrics = InferenceMetrics(
        distinct_title_count=len(scored),
        batch_size=batch_size,
        max_length=max_length,
        device=str(loaded.device),
        inference_seconds=duration,
        titles_per_second=len(scored) / duration,
        truncated_title_count=truncated_count,
        truncated_title_percentage=100.0 * truncated_count / len(scored),
        maximum_observed_token_length=maximum_observed_length,
        peak_rss_mb=process_peak_rss_mb(),
    )
    return FinBERTInference(title_scores=scored, metrics=metrics)


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_LENGTH",
    "EXPECTED_ID2LABEL",
    "MODEL_LICENSE",
    "MODEL_NAME",
    "MODEL_REVISION",
    "PILOT_RANDOM_SEED",
    "FinBERTInference",
    "FinBERTValidationError",
    "InferenceMetrics",
    "LoadedFinBERT",
    "deterministic_pilot_sample",
    "distinct_titles",
    "infer_distinct_titles",
    "load_pinned_finbert",
    "normalise_label_mapping",
    "package_versions",
    "probabilities_to_scores",
    "process_peak_rss_mb",
    "select_device",
    "validate_external_cache",
    "validate_probabilities",
]
