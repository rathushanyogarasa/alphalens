"""sentiment_cache.py
=====================
Persistent headline → sentiment cache using a CSV file keyed by SHA-256
hash of the (headline, model_name) pair.

Why this matters
----------------
FinBERT inference on 16,000+ headlines takes ~45 minutes on CPU.
With caching, a re-run loads scores in <5 seconds from disk, making
test_11 and test_10 practical to run iteratively.

Cache invalidation
------------------
The cache key is SHA-256(headline + "|" + model_name).  Changing the
model checkpoint (e.g. retraining FinBERT) does NOT auto-invalidate the
cache — call ``clear_cache()`` or delete the CSV after retraining.

Usage
-----
    from src.sentiment_cache import CachedPredictor

    model, model_name = load_model()
    predictor = CachedPredictor(model, model_name)
    # Identical to model.predict() but caches results:
    preds = predictor.predict(headlines)

    # In test scripts (drop-in replacement):
    news_df = predictor.predict_dataframe(news_df)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

_CACHE_COLS = ["cache_key", "label_name", "confidence", "prob_negative",
               "prob_neutral", "prob_positive"]


def _cache_path(model_name: str) -> Path:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return config.PROCESSED_DATA_DIR / f"sentiment_cache_{model_name}.csv"


def _make_key(headline: str, model_name: str) -> str:
    return hashlib.sha256(f"{headline}|{model_name}".encode()).hexdigest()[:16]


def load_cache(model_name: str) -> dict[str, dict]:
    """Load cache from disk.  Returns dict: cache_key -> prediction dict."""
    path = _cache_path(model_name)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str)
        cache = {}
        for _, row in df.iterrows():
            cache[row["cache_key"]] = {
                "label_name":    row["label_name"],
                "confidence":    float(row["confidence"]),
                "probabilities": {
                    "negative": float(row.get("prob_negative", 0.0)),
                    "neutral":  float(row.get("prob_neutral",  0.0)),
                    "positive": float(row.get("prob_positive", 0.0)),
                },
            }
        logger.info("Sentiment cache loaded: %d entries (%s)", len(cache), path)
        return cache
    except Exception as exc:
        logger.warning("Failed to load cache %s: %s", path, exc)
        return {}


def save_cache(cache: dict[str, dict], model_name: str) -> None:
    """Append new cache entries to disk (merges with existing)."""
    path = _cache_path(model_name)
    rows = []
    for key, p in cache.items():
        probs = p.get("probabilities", {})
        rows.append({
            "cache_key":    key,
            "label_name":   p["label_name"],
            "confidence":   p["confidence"],
            "prob_negative": probs.get("negative", 0.0),
            "prob_neutral":  probs.get("neutral",  0.0),
            "prob_positive": probs.get("positive", 0.0),
        })
    new_df = pd.DataFrame(rows, columns=_CACHE_COLS)

    if path.exists():
        try:
            existing = pd.read_csv(path, dtype=str)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["cache_key"], keep="last")
        except Exception:
            combined = new_df
    else:
        combined = new_df

    combined.to_csv(path, index=False)
    logger.info("Sentiment cache saved: %d entries -> %s", len(combined), path)


def clear_cache(model_name: str) -> None:
    """Delete the on-disk cache for model_name (use after retraining)."""
    path = _cache_path(model_name)
    if path.exists():
        path.unlink()
        logger.info("Cleared sentiment cache: %s", path)


class CachedPredictor:
    """Drop-in wrapper around FinBERTClassifier / VADERBaseline.

    Checks cache before running inference.  Saves new results back to disk
    in batches so no work is lost if interrupted.

    Args:
        model: FinBERTClassifier or VADERBaseline instance.
        model_name: String identifier used as the cache file suffix
            (e.g. ``"finbert"`` or ``"vader"``).
        save_every: Persist cache every N new predictions (default 500).
    """

    def __init__(self, model, model_name: str, save_every: int = 500):
        self.model       = model
        self.model_name  = model_name
        self.save_every  = save_every
        self._cache      = load_cache(model_name)
        self._new_count  = 0

    # ------------------------------------------------------------------

    def predict(self, texts: list[str]) -> list[dict]:
        """Return predictions for texts, using cache where available."""
        # Separate hits from misses
        keys      = [_make_key(t, self.model_name) for t in texts]
        miss_pos  = [i for i, k in enumerate(keys) if k not in self._cache]

        if miss_pos:
            logger.info("Cache miss: %d / %d headlines → running inference",
                        len(miss_pos), len(texts))
            miss_texts = [texts[i] for i in miss_pos]
            new_preds  = self.model.predict(miss_texts)

            for i, pred in zip(miss_pos, new_preds):
                entry = {
                    "label_name":    pred["label_name"],
                    "confidence":    pred["confidence"],
                    "probabilities": pred.get("probabilities", {}),
                }
                self._cache[keys[i]] = entry
                self._new_count += 1
                if self._new_count % self.save_every == 0:
                    save_cache(self._cache, self.model_name)

            save_cache(self._cache, self.model_name)

        return [{"text": texts[i], **self._cache[keys[i]]} for i in range(len(texts))]

    def predict_dataframe(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Run predict on news_df['headline'] and merge results back.

        Returns the same DataFrame with added columns:
        ``label_name``, ``confidence``.

        This is a drop-in replacement for the manual loop in test scripts:

            for i in range(0, len(texts), config.BATCH_SIZE):
                preds.extend(model.predict(texts[i:i+config.BATCH_SIZE]))
        """
        texts = news_df["headline"].tolist()
        preds = self.predict(texts)

        out = news_df.copy().reset_index(drop=True)
        out["label_name"] = [p["label_name"] for p in preds]
        out["confidence"] = [p["confidence"]  for p in preds]
        return out

    @property
    def cache_size(self) -> int:
        return len(self._cache)
