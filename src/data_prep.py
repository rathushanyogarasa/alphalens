"""Data preparation pipeline for AlphaLens.

Loads financial sentiment datasets from HuggingFace, merges and cleans
them, performs a stratified train/val/test split, and persists the splits
to disk for use by downstream training and evaluation modules.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

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
# Label constants
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
INT_TO_NAME: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_phrasebank() -> pd.DataFrame:
    """Load the Financial PhraseBank dataset from HuggingFace.

    Uses ``nickmuchi/financial-classification``, a Parquet-native mirror of
    the Financial PhraseBank corpus (equivalent to ``sentences_75agree``).
    Label mapping: 0 = negative, 1 = neutral, 2 = positive.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:

            - ``text`` (str): The financial sentence.
            - ``label`` (int): Integer label — negative=0, neutral=1, positive=2.
            - ``label_name`` (str): Human-readable label string.
            - ``source`` (str): Constant value ``"phrasebank"``.
    """
    from datasets import load_dataset  # local import to keep top-level clean

    logger.info("Loading Financial PhraseBank (nickmuchi/financial-classification) …")
    rows: list[dict] = []
    try:
        ds = load_dataset("nickmuchi/financial-classification")
        for split_name, split in ds.items():
            logger.info("  Processing PhraseBank split: '%s' (%d rows)", split_name, len(split))
            for item in split:
                label = int(item["labels"])
                rows.append(
                    {
                        "text": str(item["text"]).strip(),
                        "label": label,
                        "label_name": INT_TO_NAME[label],
                        "source": "phrasebank",
                    }
                )
    except Exception as exc:
        logger.warning(
            "PhraseBank load failed (%s). Falling back to synthetic seed set.",
            exc,
        )
        seeds = [
            ("Company beat earnings estimates and raised guidance", 2),
            ("Results were in line with analyst expectations", 1),
            ("Firm missed revenue and cut outlook", 0),
        ]
        for i in range(300):
            text, label = seeds[i % len(seeds)]
            rows.append(
                {
                    "text": f"{text} [{i}]",
                    "label": label,
                    "label_name": INT_TO_NAME[label],
                    "source": "phrasebank",
                }
            )

    df = pd.DataFrame(rows)
    df = df[df["text"] != ""]
    logger.info("PhraseBank loaded: %d rows", len(df))
    return df


def load_fiqa() -> pd.DataFrame:
    """Load the FiQA sentiment dataset from HuggingFace and discretise scores.

    Continuous sentiment scores are mapped to three classes:

    - score < -0.1  → negative (0)
    - score >  0.1  → positive (2)
    - otherwise     → neutral  (1)

    Returns:
        pd.DataFrame: A DataFrame with the following columns:

            - ``text`` (str): The financial text snippet.
            - ``label`` (int): Discretised integer label.
            - ``label_name`` (str): Human-readable label string.
            - ``source`` (str): Constant value ``"fiqa"``.
    """
    from datasets import load_dataset

    logger.info("Loading FiQA sentiment dataset …")
    rows: list[dict] = []
    try:
        ds = load_dataset("pauri32/fiqa-2018", trust_remote_code=True)
        for split_name, split in ds.items():
            logger.info("  Processing FiQA split: '%s' (%d rows)", split_name, len(split))
            for item in split:
                # FiQA uses 'sentence' as the primary text field
                text = item.get("sentence") or item.get("text") or item.get("question") or ""
                score = float(item.get("sentiment_score", item.get("score", 0.0)))

                if score < -0.1:
                    label = 0
                elif score > 0.1:
                    label = 2
                else:
                    label = 1

                rows.append(
                    {
                        "text": str(text).strip(),
                        "label": label,
                        "label_name": INT_TO_NAME[label],
                        "source": "fiqa",
                    }
                )
    except Exception as exc:
        logger.warning("FiQA load failed (%s). Falling back to synthetic seed set.", exc)
        seeds = [
            ("Stock plunged after weak quarterly guidance", 0),
            ("Management reiterated full-year targets", 1),
            ("Profit margins expanded beyond expectations", 2),
        ]
        for i in range(300):
            text, label = seeds[i % len(seeds)]
            rows.append(
                {
                    "text": f"{text} [{i}]",
                    "label": label,
                    "label_name": INT_TO_NAME[label],
                    "source": "fiqa",
                }
            )

    df = pd.DataFrame(rows)
    df = df[df["text"] != ""]  # drop blank rows
    logger.info("FiQA loaded: %d rows after cleaning", len(df))
    return df


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_datasets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate, deduplicate, and shuffle multiple sentiment DataFrames.

    Args:
        dfs: List of DataFrames produced by individual loader functions.
            Each must have columns ``text``, ``label``, ``label_name``,
            and ``source``.

    Returns:
        pd.DataFrame: A single shuffled DataFrame with duplicate texts
        removed and a clean integer index.
    """
    logger.info("Merging %d dataset(s) …", len(dfs))

    # Log per-source class distribution before merge
    for df in dfs:
        source = df["source"].iloc[0] if len(df) else "unknown"
        dist = df["label_name"].value_counts().to_dict()
        logger.info("  [%s] class distribution: %s", source, dist)

    combined = pd.concat(dfs, ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset="text")
    after = len(combined)
    logger.info("Dropped %d duplicate texts (%d → %d rows)", before - after, before, after)

    combined = combined.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(
        drop=True
    )

    # Log combined class distribution
    dist = combined["label_name"].value_counts().to_dict()
    logger.info("Combined dataset: %d rows | class distribution: %s", len(combined), dist)

    return combined


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a stratified train / validation / test split.

    Split ratios are read from ``config.TRAIN_SPLIT``, ``config.VAL_SPLIT``,
    and ``config.TEST_SPLIT``.

    Args:
        df: The merged and shuffled DataFrame to split.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            ``(train_df, val_df, test_df)`` — three non-overlapping subsets.
    """
    val_test_size = config.VAL_SPLIT + config.TEST_SPLIT
    relative_test_size = config.TEST_SPLIT / val_test_size

    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_size,
        stratify=df["label"],
        random_state=config.RANDOM_SEED,
    )

    val_df, test_df = train_test_split(
        val_test_df,
        test_size=relative_test_size,
        stratify=val_test_df["label"],
        random_state=config.RANDOM_SEED,
    )

    for name, subset in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = subset["label_name"].value_counts().to_dict()
        logger.info("  [%s] %d rows | %s", name, len(subset), dist)

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Persist the three dataset splits to CSV files.

    Files are written to ``config.PROCESSED_DATA_DIR`` as
    ``train.csv``, ``val.csv``, and ``test.csv``.

    Args:
        train: Training split DataFrame.
        val: Validation split DataFrame.
        test: Test split DataFrame.
    """
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        path: Path = config.PROCESSED_DATA_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved %s split → %s (%d rows)", name, path, len(df))


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved train / validation / test splits from disk.

    Reads ``train.csv``, ``val.csv``, and ``test.csv`` from
    ``config.PROCESSED_DATA_DIR``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            ``(train_df, val_df, test_df)`` loaded from CSV.

    Raises:
        FileNotFoundError: If any of the three CSV files is missing.
            The error message includes the expected file path so the
            user knows to run the data preparation pipeline first.
    """
    dfs: list[pd.DataFrame] = []
    for name in ("train", "val", "test"):
        path: Path = config.PROCESSED_DATA_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}\n"
                "Run `python src/data_prep.py` (or call run_data_prep()) "
                "to generate the processed splits."
            )
        df = pd.read_csv(path)
        logger.info("Loaded %s split ← %s (%d rows)", name, path, len(df))
        dfs.append(df)

    return tuple(dfs)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_data_prep() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full data preparation pipeline end-to-end.

    Steps:

    1. Load Financial PhraseBank from HuggingFace.
    2. Load FiQA sentiment dataset from HuggingFace.
    3. Merge and deduplicate both datasets.
    4. Optionally subsample to ``config.QUICK_TEST_SAMPLES`` rows when
       ``config.QUICK_TEST`` is ``True``.
    5. Stratified train / val / test split.
    6. Save splits to ``config.PROCESSED_DATA_DIR``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            ``(train_df, val_df, test_df)`` ready for model training.
    """
    logger.info("=== AlphaLens Data Preparation Pipeline ===")

    if config.QUICK_TEST:
        logger.info("QUICK_TEST enabled — using synthetic local dataset")
        seeds = [
            ("Revenue and earnings beat expectations with raised guidance", 2),
            ("Results were broadly in line with market expectations", 1),
            ("Company missed estimates and cut full-year outlook", 0),
        ]
        rows: list[dict] = []
        for i in range(max(config.QUICK_TEST_SAMPLES * 3, 300)):
            text, label = seeds[i % len(seeds)]
            rows.append(
                {
                    "text": f"{text} [{i}]",
                    "label": label,
                    "label_name": INT_TO_NAME[label],
                    "source": "quick_test",
                }
            )
        merged_df = pd.DataFrame(rows)
        train_df, val_df, test_df = split_dataset(merged_df)
        save_splits(train_df, val_df, test_df)
        logger.info("=== Data preparation complete (quick-test synthetic) ===")
        return train_df, val_df, test_df

    phrasebank_df = load_phrasebank()
    fiqa_df = load_fiqa()

    merged_df = merge_datasets([phrasebank_df, fiqa_df])

    if config.QUICK_TEST:
        logger.info(
            "QUICK_TEST enabled — subsampling to %d rows", config.QUICK_TEST_SAMPLES
        )
        merged_df = (
            merged_df.groupby("label", group_keys=False)
            .apply(
                lambda grp: grp.sample(
                    min(len(grp), config.QUICK_TEST_SAMPLES // 3),
                    random_state=config.RANDOM_SEED,
                )
            )
            .sample(frac=1, random_state=config.RANDOM_SEED)
            .reset_index(drop=True)
        )
        logger.info("Subsampled to %d rows", len(merged_df))

    train_df, val_df, test_df = split_dataset(merged_df)
    save_splits(train_df, val_df, test_df)

    logger.info("=== Data preparation complete ===")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_data_prep()
