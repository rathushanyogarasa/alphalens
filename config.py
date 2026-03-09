"""AlphaLens project configuration.

This module defines all configurable parameters for the AlphaLens
NLP-driven sentiment analysis and portfolio management system.
All directory paths are created on import if they do not exist.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project identity
# ---------------------------------------------------------------------------

PROJECT_NAME: str = "AlphaLens"
"""Human-readable name of the project."""

VERSION: str = "1.0.0"
"""Semantic version string."""

# ---------------------------------------------------------------------------
# Universe of securities
# ---------------------------------------------------------------------------

TICKERS: list[str] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "JPM",
    "GS",
    "BAC",
    "TSLA",
    "NVDA",
    "META",
]
"""List of equity tickers used for data collection, training, and backtesting."""

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------

MODEL_NAME: str = "ProsusAI/finbert"
"""HuggingFace model identifier for the sentiment backbone."""

BATCH_SIZE: int = 16
"""Number of samples per forward/backward pass."""

EPOCHS: int = 3
"""Number of full passes over the training dataset."""

LEARNING_RATE: float = 2e-5
"""AdamW optimiser learning rate."""

MAX_LENGTH: int = 128
"""Maximum token length fed to the transformer."""

RANDOM_SEED: int = 42
"""Global random seed for reproducibility."""

# ---------------------------------------------------------------------------
# Dataset splits
# ---------------------------------------------------------------------------

TRAIN_SPLIT: float = 0.70
"""Fraction of data allocated to training."""

VAL_SPLIT: float = 0.15
"""Fraction of data allocated to validation."""

TEST_SPLIT: float = 0.15
"""Fraction of data allocated to held-out testing.

Note:
    TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT must equal 1.0.
"""

# ---------------------------------------------------------------------------
# Signal thresholds
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD: float = 0.70
"""Minimum model confidence required to act on a sentiment prediction."""

BUY_THRESHOLD: float = 0.3
"""Composite sentiment score above which a buy signal is generated."""

SELL_THRESHOLD: float = -0.3
"""Composite sentiment score below which a sell signal is generated."""

# ---------------------------------------------------------------------------
# Event-study and keyword parameters
# ---------------------------------------------------------------------------

EVENT_WINDOW: tuple[int, int] = (-1, 3)
"""(pre, post) trading-day window around each news event for return calculation."""

MIN_EVENTS_PER_KEYWORD: int = 5
"""Minimum number of events a keyword must appear in to be retained."""

TOP_N_KEYWORDS: int = 50
"""Number of top-scoring keywords kept after ranking."""

RECENCY_DECAY_LAMBDA: float = 0.3
"""Exponential decay rate applied to older events when computing keyword scores."""

# ---------------------------------------------------------------------------
# Portfolio / backtest parameters
# ---------------------------------------------------------------------------

RISK_FREE_RATE: float = 0.045
"""Annualised risk-free rate used for Sharpe and Sortino ratio calculations."""

MARKET_BENCHMARK: str = "SPY"
"""Ticker symbol of the market benchmark for alpha/beta computation."""

# ---------------------------------------------------------------------------
# Data-source reliability weights
# ---------------------------------------------------------------------------

SOURCE_WEIGHTS: dict[str, float] = {
    "sec_edgar": 1.0,
    "reuters": 0.8,
    "financial_times": 0.8,
    "newsapi": 0.6,
    "yfinance": 0.5,
    "reddit": 0.3,
}
"""Reliability weight assigned to each news/data source.

Scores range from 0.0 (untrusted) to 1.0 (fully trusted) and are used
to down-weight sentiment signals from lower-quality sources.
"""

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------

NEWS_API_KEY: str = ""
"""NewsAPI.org key. Set via environment variable or replace this default."""

# ---------------------------------------------------------------------------
# Development / quick-test flags
# ---------------------------------------------------------------------------

QUICK_TEST: bool = False
"""When ``True`` the pipeline uses a small data subset for fast iteration."""

QUICK_TEST_SAMPLES: int = 100
"""Number of samples to use when :data:`QUICK_TEST` is enabled."""

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).parent
"""Repository root directory (location of this file)."""

DATA_DIR: Path = BASE_DIR / "data"
"""Top-level data directory."""

RAW_DATA_DIR: Path = DATA_DIR / "raw"
"""Directory for raw, unprocessed data downloads."""

PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
"""Directory for cleaned and feature-engineered datasets."""

RESULTS_DIR: Path = BASE_DIR / "results"
"""Top-level results directory."""

PLOTS_DIR: Path = RESULTS_DIR / "plots"
"""Directory for saved matplotlib / seaborn figures."""

METRICS_DIR: Path = RESULTS_DIR / "metrics"
"""Directory for JSON / CSV evaluation metric files."""

MODEL_DIR: Path = RESULTS_DIR / "best_model"
"""Directory where the best fine-tuned model checkpoint is saved."""

# ---------------------------------------------------------------------------
# Ensure all directories exist on import
# ---------------------------------------------------------------------------

for _dir in (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    METRICS_DIR,
    MODEL_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)
