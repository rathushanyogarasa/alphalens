"""Training pipeline for the FinBERT sentiment classifier.

Provides :class:`FinBERTTrainer` which handles dataset construction,
the training loop with gradient clipping and a linear warm-up schedule,
per-epoch validation, and best-checkpoint saving.

Typical usage::

    from src.train import run_training
    model = run_training()
"""

import logging
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.model import FinBERTClassifier
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
# PyTorch Dataset
# ---------------------------------------------------------------------------


class SentimentDataset(Dataset):
    """PyTorch Dataset wrapping a sentiment DataFrame.

    Each item is a ``(text, label)`` pair drawn from *df*.

    Args:
        df: DataFrame with at least ``text`` (str) and ``label`` (int)
            columns, as produced by :mod:`src.data_prep`.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.texts: list[str] = df["text"].tolist()
        self.labels: list[int] = df["label"].tolist()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        """Return the (text, label) pair at position *idx*.

        Args:
            idx: Integer index into the dataset.

        Returns:
            tuple[str, int]: The raw text string and its integer label.
        """
        return self.texts[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def _collate_fn(
    batch: list[tuple[str, int]],
    tokenizer,
    device: torch.device,
) -> tuple[dict, torch.Tensor]:
    """Tokenise and collate a list of (text, label) pairs into tensors.

    Args:
        batch: List of ``(text, label)`` tuples from :class:`SentimentDataset`.
        tokenizer: HuggingFace tokenizer used to encode the texts.
        device: Target device for all returned tensors.

    Returns:
        tuple[dict, torch.Tensor]: A pair of ``(encoding, labels)`` where
        *encoding* contains ``input_ids`` and ``attention_mask`` tensors
        on *device* and *labels* is a ``(batch,)`` int64 tensor.
    """
    texts, labels = zip(*batch)
    encoding = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=config.MAX_LENGTH,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    return encoding, label_tensor


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class FinBERTTrainer:
    """Manages the full training and validation lifecycle for :class:`FinBERTClassifier`.

    Constructs :class:`SentimentDataset` and :class:`DataLoader` objects
    for both splits, then runs a standard transformer fine-tuning loop
    with cross-entropy loss, gradient clipping, and a linear warm-up
    schedule.  The best checkpoint (by macro F1 on the validation set)
    is saved to ``config.MODEL_DIR``.

    Args:
        model: An initialised :class:`FinBERTClassifier` to fine-tune.
        train_df: Training split DataFrame with ``text`` and ``label`` columns.
        val_df: Validation split DataFrame with ``text`` and ``label`` columns.
    """

    def __init__(
        self,
        model: FinBERTClassifier,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> None:
        # Reproducibility
        _set_seeds(config.RANDOM_SEED)

        self.model = model
        self.device = model.device

        # Datasets & loaders
        train_ds = SentimentDataset(train_df)
        val_ds = SentimentDataset(val_df)

        collate = lambda batch: _collate_fn(  # noqa: E731
            batch, model.tokenizer, self.device
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate,
        )

        # Optimiser
        self.optimiser = torch.optim.AdamW(
            model.parameters(), lr=config.LEARNING_RATE
        )

        # Scheduler with linear warm-up over first 10 % of steps
        total_steps = len(self.train_loader) * config.EPOCHS
        warmup_steps = max(1, int(0.10 * total_steps))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimiser,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_f1: float = -1.0

        logger.info(
            "FinBERTTrainer ready | train=%d batches | val=%d batches "
            "| total_steps=%d | warmup_steps=%d",
            len(self.train_loader),
            len(self.val_loader),
            total_steps,
            warmup_steps,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        """Run one training epoch and return the mean cross-entropy loss.

        Returns:
            float: Average loss across all training batches.
        """
        self.model.train()
        total_loss = 0.0

        for encoding, labels in tqdm(
            self.train_loader, desc="  train", leave=False, unit="batch"
        ):
            self.optimiser.zero_grad()
            logits = self.model(encoding["input_ids"], encoding["attention_mask"])
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()
            self.scheduler.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _eval_epoch(self) -> tuple[float, float]:
        """Evaluate on the validation set.

        Returns:
            tuple[float, float]: ``(val_loss, macro_f1)`` averaged over
            all validation batches.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for encoding, labels in tqdm(
                self.val_loader, desc="  val  ", leave=False, unit="batch"
            ):
                logits = self.model(encoding["input_ids"], encoding["attention_mask"])
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        val_loss = total_loss / len(self.val_loader)
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return val_loss, val_f1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Run the full training loop for ``config.EPOCHS`` epochs.

        Each epoch:

        1. Runs a full forward + backward pass over the training set.
        2. Evaluates cross-entropy loss and macro F1 on the validation set.
        3. Logs epoch metrics (loss, F1, learning rate, elapsed time).
        4. Saves the model checkpoint to ``config.MODEL_DIR`` if
           validation F1 has improved.

        Returns:
            dict: Training history with keys ``"epoch"``, ``"train_loss"``,
            ``"val_loss"``, ``"val_f1"`` — each mapping to a list of
            per-epoch values.
        """
        history: dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
        }

        logger.info("Starting training for %d epoch(s) …", config.EPOCHS)
        run_start = time.time()

        for epoch in range(1, config.EPOCHS + 1):
            epoch_start = time.time()

            train_loss = self._train_epoch()
            val_loss, val_f1 = self._eval_epoch()

            elapsed = time.time() - epoch_start
            current_lr = self.scheduler.get_last_lr()[0]

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "val_f1=%.4f | lr=%.2e | elapsed=%.1fs",
                epoch,
                config.EPOCHS,
                train_loss,
                val_loss,
                val_f1,
                current_lr,
                elapsed,
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["val_f1"].append(round(val_f1, 4))

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.model.save(config.MODEL_DIR)
                logger.info(
                    "  ✓ New best val_f1=%.4f — checkpoint saved → %s",
                    val_f1,
                    config.MODEL_DIR,
                )

        total_time = time.time() - run_start
        logger.info(
            "Training complete in %.1fs | best_val_f1=%.4f", total_time, self.best_val_f1
        )
        return history


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _set_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: The integer seed value to apply.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seeds set to %d", seed)


def plot_training_curves(history: dict) -> None:
    """Plot training and validation loss curves and save to disk.

    Args:
        history: Dict returned by :meth:`FinBERTTrainer.train` with keys
            ``"epoch"``, ``"train_loss"``, and ``"val_loss"``.
    """
    apply_plot_style()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "training_curves.png"

    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Train loss",
        color=NAVY,
    )
    axes[0].plot(
        epochs,
        history["val_loss"],
        marker="s",
        label="Val loss",
        color=GOLD,
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 curve
    axes[1].plot(
        epochs,
        history["val_f1"],
        marker="^",
        color=NAVY,
        label="Val macro-F1",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Validation F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved → %s", out_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_training() -> FinBERTClassifier:
    """Run the full training pipeline end-to-end.

    Steps:

    1. Load train/val/test splits from ``config.PROCESSED_DATA_DIR``
       via :func:`src.data_prep.load_splits`.
    2. Optionally subsample the training set when ``config.QUICK_TEST``
       is enabled.
    3. Initialise :class:`FinBERTClassifier` and :class:`FinBERTTrainer`.
    4. Run :meth:`FinBERTTrainer.train`.
    5. Plot training curves.
    6. Reload the best checkpoint from ``config.MODEL_DIR``.
    7. Return the best model.

    Returns:
        FinBERTClassifier: The fine-tuned model loaded from the best
        validation-F1 checkpoint.
    """
    from src.data_prep import load_splits  # avoid circular at module level

    logger.info("=== AlphaLens Training Pipeline ===")

    train_df, val_df, _ = load_splits()

    if config.QUICK_TEST:
        n = min(len(train_df), config.QUICK_TEST_SAMPLES)
        train_df = train_df.sample(n, random_state=config.RANDOM_SEED).reset_index(drop=True)
        logger.info("QUICK_TEST enabled — training on %d samples", n)

    model = FinBERTClassifier()
    trainer = FinBERTTrainer(model, train_df, val_df)
    history = trainer.train()

    plot_training_curves(history)

    best_model = FinBERTClassifier.load(config.MODEL_DIR)
    logger.info("=== Training pipeline complete ===")
    return best_model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_training()
