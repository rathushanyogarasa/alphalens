"""Model evaluation utilities for AlphaLens.

Provides functions to evaluate sentiment classifiers on a held-out test
set, generate confusion-matrix and comparison plots, persist metric CSVs,
and print a formatted comparison table to the console.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.model import ID2LABEL, VADERBaseline
from src.plot_style import GOLD, NAVY, FIG_DPI, apply_plot_style

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label order used consistently across all evaluation functions
# ---------------------------------------------------------------------------

_LABEL_NAMES: list[str] = ["negative", "neutral", "positive"]


# ---------------------------------------------------------------------------
# 1. evaluate_model
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    test_df: pd.DataFrame,
    model_name: str,
) -> dict:
    """Evaluate a sentiment model on a held-out test set.

    Runs ``model.predict()`` in batches of ``config.BATCH_SIZE`` and
    computes a comprehensive set of classification metrics.

    Args:
        model: Any object exposing a ``predict(texts: list[str])``
            interface that returns dicts with a ``"label"`` key —
            compatible with both :class:`~src.model.FinBERTClassifier`
            and :class:`~src.model.VADERBaseline`.
        test_df: DataFrame with ``text`` (str) and ``label`` (int)
            columns, as produced by :mod:`src.data_prep`.
        model_name: Short identifier used in log messages and when
            building output file names (e.g. ``"vader"`` or
            ``"finbert"``).

    Returns:
        dict: Evaluation metrics with the following keys:

            - ``model_name`` (str)
            - ``accuracy`` (float)
            - ``weighted_f1`` (float)
            - ``macro_f1`` (float)
            - ``per_class`` (dict): Maps each label name to a nested
              dict of ``precision``, ``recall``, and ``f1``.
            - ``confusion_matrix`` (np.ndarray): Shape ``(3, 3)``.
    """
    texts = test_df["text"].tolist()
    true_labels = test_df["label"].tolist()

    logger.info("Evaluating [%s] on %d test samples …", model_name, len(texts))

    # Batch predict
    predictions: list[dict] = []
    for start in range(0, len(texts), config.BATCH_SIZE):
        batch = texts[start : start + config.BATCH_SIZE]
        predictions.extend(model.predict(batch))

    pred_labels = [p["label"] for p in predictions]

    accuracy = accuracy_score(true_labels, pred_labels)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=_LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": round(report[name]["precision"], 4),
            "recall": round(report[name]["recall"], 4),
            "f1": round(report[name]["f1-score"], 4),
        }
        for name in _LABEL_NAMES
    }

    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }

    logger.info(
        "[%s] accuracy=%.4f | weighted_f1=%.4f | macro_f1=%.4f",
        model_name,
        accuracy,
        weighted_f1,
        macro_f1,
    )
    for name, scores in per_class.items():
        logger.info(
            "  [%s] %s → precision=%.4f recall=%.4f f1=%.4f",
            model_name,
            name,
            scores["precision"],
            scores["recall"],
            scores["f1"],
        )

    return metrics


# ---------------------------------------------------------------------------
# 2. plot_confusion_matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    """Plot and save a seaborn confusion-matrix heatmap.

    Args:
        cm: Confusion matrix array of shape ``(3, 3)`` as returned by
            :func:`evaluate_model`.
        model_name: Short model identifier used in the plot title and
            output filename (e.g. ``"vader"`` or ``"finbert"``).
    """
    apply_plot_style()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / f"confusion_matrix_{model_name}.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.light_palette(NAVY, as_cmap=True),
        xticklabels=_LABEL_NAMES,
        yticklabels=_LABEL_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix — {model_name.upper()}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", out_path)


# ---------------------------------------------------------------------------
# 3. plot_model_comparison
# ---------------------------------------------------------------------------


def plot_model_comparison(vader_metrics: dict, finbert_metrics: dict) -> None:
    """Create a side-by-side bar chart comparing VADER and FinBERT.

    Compares ``accuracy`` and ``weighted_f1`` for both models.
    FinBERT bars are navy; VADER bars are gold.

    Args:
        vader_metrics: Metrics dict returned by :func:`evaluate_model`
            for the VADER baseline.
        finbert_metrics: Metrics dict returned by :func:`evaluate_model`
            for the FinBERT model.
    """
    apply_plot_style()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "model_comparison.png"

    metrics_keys = ["accuracy", "weighted_f1", "macro_f1"]
    labels = ["Accuracy", "Weighted F1", "Macro F1"]

    vader_vals = [vader_metrics[k] for k in metrics_keys]
    finbert_vals = [finbert_metrics[k] for k in metrics_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_finbert = ax.bar(x - width / 2, finbert_vals, width, label="FinBERT", color=NAVY)
    bars_vader = ax.bar(x + width / 2, vader_vals, width, label="VADER", color=GOLD)

    # Annotate bar tops
    for bar in bars_finbert:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=NAVY,
        )
    for bar in bars_vader:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=GOLD,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — FinBERT vs VADER")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Model comparison plot saved → %s", out_path)


# ---------------------------------------------------------------------------
# 4. save_metrics
# ---------------------------------------------------------------------------


def save_metrics(metrics: dict, filename: str) -> None:
    """Persist evaluation metrics to a CSV file in ``config.METRICS_DIR``.

    Scalar metrics (``accuracy``, ``weighted_f1``, ``macro_f1``) are
    written to one row.  Per-class metrics are appended as additional
    rows, one per class.  The confusion matrix is excluded because it
    is non-tabular.

    Args:
        metrics: Metrics dict as returned by :func:`evaluate_model`.
        filename: Output filename (e.g. ``"vader_metrics.csv"``).
            Saved under ``config.METRICS_DIR``.
    """
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.METRICS_DIR / filename

    rows: list[dict] = [
        {
            "model": metrics["model_name"],
            "class": "overall",
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_f1": metrics["macro_f1"],
            "precision": None,
            "recall": None,
            "f1": None,
        }
    ]
    for class_name, scores in metrics.get("per_class", {}).items():
        rows.append(
            {
                "model": metrics["model_name"],
                "class": class_name,
                "accuracy": None,
                "weighted_f1": None,
                "macro_f1": None,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Metrics saved → %s", out_path)


# ---------------------------------------------------------------------------
# 5. print_comparison_table
# ---------------------------------------------------------------------------


def print_comparison_table(vader_metrics: dict, finbert_metrics: dict) -> None:
    """Print a formatted Unicode comparison table to the console.

    Displays accuracy, weighted F1, and macro F1 side-by-side for the
    VADER baseline and FinBERT model.

    Args:
        vader_metrics: Metrics dict for the VADER baseline.
        finbert_metrics: Metrics dict for the FinBERT model.

    Example output::

        ┌──────────────────┬─────────┬──────────┐
        │ Metric           │  VADER  │ FinBERT  │
        ├──────────────────┼─────────┼──────────┤
        │ Accuracy         │  0.682  │  0.871   │
        │ Weighted F1      │  0.671  │  0.868   │
        │ Macro F1         │  0.598  │  0.849   │
        └──────────────────┴─────────┴──────────┘
    """
    rows = [
        ("Accuracy", vader_metrics["accuracy"], finbert_metrics["accuracy"]),
        ("Weighted F1", vader_metrics["weighted_f1"], finbert_metrics["weighted_f1"]),
        ("Macro F1", vader_metrics["macro_f1"], finbert_metrics["macro_f1"]),
    ]

    col0_w, col1_w, col2_w = 18, 9, 10

    # Attempt UTF-8 output on Windows; fall back to ASCII borders silently.
    import sys as _sys

    _use_unicode = True
    if hasattr(_sys.stdout, "reconfigure"):
        try:
            _sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            _use_unicode = False
    else:
        try:
            "┌".encode(_sys.stdout.encoding or "ascii")
        except (UnicodeEncodeError, LookupError):
            _use_unicode = False

    if _use_unicode:
        tl, tr, bl, br = "┌", "┐", "└", "┘"
        h, v, tm, bm, lm, rm, cross = "─", "│", "┬", "┴", "├", "┤", "┼"
    else:
        tl, tr, bl, br = "+", "+", "+", "+"
        h, v, tm, bm, lm, rm, cross = "-", "|", "+", "+", "+", "+", "+"

    h_top = f"{tl}{h * col0_w}{tm}{h * col1_w}{tm}{h * col2_w}{tr}"
    h_hdr = f"{v} {'Metric':<{col0_w - 2}} {v} {'VADER':^{col1_w - 2}} {v} {'FinBERT':^{col2_w - 2}} {v}"
    h_mid = f"{lm}{h * col0_w}{cross}{h * col1_w}{cross}{h * col2_w}{rm}"
    h_bot = f"{bl}{h * col0_w}{bm}{h * col1_w}{bm}{h * col2_w}{br}"

    print(h_top)
    print(h_hdr)
    print(h_mid)
    for metric, vader_val, finbert_val in rows:
        print(
            f"{v} {metric:<{col0_w - 2}} {v} {vader_val:^{col1_w - 2}.3f} {v} {finbert_val:^{col2_w - 2}.3f} {v}"
        )
    print(h_bot)


# ---------------------------------------------------------------------------
# 6. run_evaluation
# ---------------------------------------------------------------------------


def run_evaluation(finbert_model) -> dict:
    """Run the full evaluation pipeline for both VADER and FinBERT.

    Steps:

    1. Load the held-out test split from ``config.PROCESSED_DATA_DIR``.
    2. Evaluate :class:`~src.model.VADERBaseline` on the test set.
    3. Evaluate *finbert_model* on the test set.
    4. Generate and save confusion-matrix plots for both models.
    5. Generate and save the side-by-side model comparison plot.
    6. Save per-model metric CSVs to ``config.METRICS_DIR``.
    7. Print the formatted comparison table to the console.

    Args:
        finbert_model: A trained :class:`~src.model.FinBERTClassifier`
            instance (or any object with a compatible ``predict``
            interface).

    Returns:
        dict: A dict with keys ``"vader"`` and ``"finbert"``, each
        mapping to the corresponding metrics dict from
        :func:`evaluate_model`.
    """
    from src.data_prep import load_splits  # avoid circular at module level

    logger.info("=== AlphaLens Evaluation Pipeline ===")

    _, _, test_df = load_splits()
    logger.info("Test set: %d samples", len(test_df))

    # VADER baseline
    vader = VADERBaseline()
    vader_metrics = evaluate_model(vader, test_df, model_name="vader")

    # FinBERT
    finbert_metrics = evaluate_model(finbert_model, test_df, model_name="finbert")

    # Plots
    plot_confusion_matrix(vader_metrics["confusion_matrix"], model_name="vader")
    plot_confusion_matrix(finbert_metrics["confusion_matrix"], model_name="finbert")
    plot_model_comparison(vader_metrics, finbert_metrics)

    # Persist
    save_metrics(vader_metrics, "vader_metrics.csv")
    save_metrics(finbert_metrics, "finbert_metrics.csv")

    # Console summary
    print_comparison_table(vader_metrics, finbert_metrics)

    logger.info("=== Evaluation pipeline complete ===")
    return {"vader": vader_metrics, "finbert": finbert_metrics}
