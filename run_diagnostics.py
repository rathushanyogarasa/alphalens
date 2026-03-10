from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config
from src.model import FinBERTClassifier, VADERBaseline
from src.evaluate import run_evaluation
from src.backtest import run_backtest
from src.signal_validation import run_signal_validation
from src.model_trust import run_trust_scoring


def load_best_model():
    weights = config.MODEL_DIR / "weights.pt"
    if weights.exists():
        print(f"[INFO] Loading trained FinBERT model from: {config.MODEL_DIR}")
        return FinBERTClassifier.load(config.MODEL_DIR)
    print("[WARN] No trained FinBERT checkpoint found, falling back to VADERBaseline")
    return VADERBaseline()


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    model = load_best_model()

    print_header("TEST 1 — SENTIMENT EVALUATION")
    eval_metrics = run_evaluation(model)
    print(json.dumps(eval_metrics, indent=2, default=str))

    print_header("TEST 2 — SIGNAL VALIDATION")
    sig_results = run_signal_validation(model, model_name="finbert_manual")
    if "hit_rate" in sig_results:
        print("\nHit rate summary:")
        print(sig_results["hit_rate"])
    if "ic_decay" in sig_results and sig_results["ic_decay"] is not None:
        print("\nIC decay:")
        print(sig_results["ic_decay"])
    if "quantile_returns" in sig_results and sig_results["quantile_returns"] is not None:
        print("\nQuantile returns:")
        print(sig_results["quantile_returns"])

    print_header("TEST 3 — BACKTEST")
    bt_metrics = run_backtest(model, model_name="finbert_manual")
    print(json.dumps(bt_metrics, indent=2, default=str))

    print_header("TEST 4 — TRUST SCORE")
    trust_report = run_trust_scoring(
        eval_metrics=eval_metrics,
        backtest_metrics=bt_metrics,
        validation_metrics=sig_results,
    )

    if trust_report is not None:
        print(trust_report.summary())
        trust_path = config.METRICS_DIR / "trust_report_manual.json"
        with open(trust_path, "w", encoding="utf-8") as f:
            json.dump(trust_report.to_dict(), f, indent=2)
        print(f"\nSaved trust report to: {trust_path}")

    print_header("TEST 5 — QUICK FILE CHECK")
    files_to_check = [
        config.METRICS_DIR / "finbert_metrics.csv",       # written by evaluate.save_metrics()
        config.METRICS_DIR / "vader_metrics.csv",
        config.METRICS_DIR / "backtest_metrics_finbert_manual.csv",
        config.METRICS_DIR / "hit_rate_finbert_manual.csv",
        config.METRICS_DIR / "ic_decay_finbert_manual.csv",
        config.METRICS_DIR / "quantile_returns_finbert_manual.csv",
    ]

    for path in files_to_check:
        print(f"{path}: {'FOUND' if path.exists() else 'MISSING'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
