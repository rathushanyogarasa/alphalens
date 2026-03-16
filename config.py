"""AlphaLens project configuration.

This module defines all configurable parameters for the AlphaLens
NLP-driven sentiment analysis and portfolio management system.
All directory paths are created on import if they do not exist.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

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
    # Technology (20)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    "AVGO", "AMD", "ORCL", "CRM", "ADBE", "INTC", "QCOM", "TXN",
    "IBM", "NOW", "AMAT", "MU", "LRCX",
    # Financials (15)
    "JPM", "BAC", "GS", "WFC", "C", "MS", "BLK", "SCHW",
    "AXP", "USB", "PNC", "COF", "MET", "PRU", "TFC",
    # Healthcare (12)
    "UNH", "JNJ", "LLY", "ABBV", "PFE", "MRK", "TMO",
    "MDT", "AMGN", "BMY", "GILD", "CVS",
    # Consumer Staples (8)
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "SBUX", "PM",
    # Consumer Discretionary (7)
    "HD", "NKE", "TGT", "F", "GM", "LOW", "BKNG",
    # Energy (8)
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "OXY",
    # Industrials (10)
    "CAT", "HON", "UPS", "BA", "GE", "MMM", "LMT", "RTX", "DE", "EMR",
    # Communication Services (7)
    "T", "VZ", "CMCSA", "NFLX", "DIS", "CHTR", "FOXA",
    # Materials (5)
    "LIN", "SHW", "APD", "FCX", "NEM",
    # Real Estate (4)
    "PLD", "AMT", "CCI", "EQIX",
    # Utilities (4)
    "NEE", "DUK", "SO", "AEP",
]
"""List of equity tickers used for data collection, training, and backtesting.

Expanded from 10 to 100 stocks (S&P 500 large-cap diversified universe) covering
10 GICS sectors. Wider universe improves cross-sectional IC significance and
reduces survivorship bias concentrated in mega-cap tech.
"""

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

BUY_THRESHOLD: float = 0.60
"""Composite sentiment score above which a buy signal is generated.
Raised to 0.60 to reduce false positives and focus on high-conviction signals
(ablation study test_05 confirmed tighter thresholds improve Sharpe)."""

SELL_THRESHOLD: float = -0.60
"""Composite sentiment score below which a sell signal is generated.
Raised to -0.60 symmetrically with BUY_THRESHOLD."""

# ---------------------------------------------------------------------------
# Conviction requirements (multi-component agreement)
# ---------------------------------------------------------------------------

MIN_CONFIDENT_HEADLINES: int = 3
"""Minimum number of confident headlines required before issuing a BUY or SELL.
If fewer are available the engine defaults to HOLD."""

MIN_SOURCE_QUALITY: float = 0.65
"""Minimum average source-weight among confident headlines required for a
directional recommendation.  Prevents low-credibility-only signals from
triggering a trade."""

MIN_MODEL_QUALITY_FOR_TRADE: float = 5.5
"""Minimum overall model quality score (0–10) required to issue a BUY or SELL.
Below this threshold all recommendations are capped at HOLD.
Set to 0.0 to disable the quality gate."""

REQUIRE_MACRO_ALIGNMENT: bool = True
"""When True, the engine will reduce conviction (downgrade BUY/SELL to HOLD)
if the macro regime is hostile to the ticker's sector exposure."""

# ---------------------------------------------------------------------------
# Keyword quality filter
# ---------------------------------------------------------------------------

MIN_KEYWORD_CAR_MAGNITUDE: float = 0.003
"""Minimum |avg_CAR| required for a keyword to influence live scoring.
Keywords loaded from keyword_summary.csv below this threshold are ignored."""

KEYWORD_GRADE_FILTER: str = "B"
"""Minimum signal_grade ('A', 'B', or 'C') for keywords to be loaded into
the live recommendation engine.  Grade 'C' keywords are too noisy for
production use."""

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

TRANSACTION_COST_BPS: float = 10.0
"""Round-trip transaction cost in basis points applied per trade in backtest.
10 bps = 0.10% total round-trip (entry + exit combined), charged once per
position change.  Set to 0.0 to disable."""

SLIPPAGE_BPS: float = 5.0
"""Estimated slippage in basis points applied per trade round-trip.
5 bps is conservative for liquid large-caps; charged once at entry alongside
TRANSACTION_COST_BPS."""

MAX_POSITION_SIZE: float = 0.20
"""Maximum single-position weight in the portfolio (0.20 = 20%)."""

LONGSHORT_HOLD_DAYS: int = 2
"""Number of trading days to hold each cross-sectional long-short position.
2-day horizon matched empirical IC peak from test_05 horizon ablation."""

LONGSHORT_QUANTILE_CUTOFF: float = 0.20
"""Top/bottom fraction of the ranked universe to trade in long-short mode.
0.20 = top quintile long, bottom quintile short (trade only strongest signals)."""

MAX_SECTOR_WEIGHT: float = 0.40
"""Maximum total weight in any single GICS sector (0.40 = 40%)."""

MAX_PORTFOLIO_POSITIONS: int = 10
"""Maximum number of simultaneous long positions in ranked portfolio mode."""

# ---------------------------------------------------------------------------
# Data-source reliability weights
# ---------------------------------------------------------------------------

SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology",  "MSFT": "Technology",  "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology",   "AVGO": "Technology",
    "AMD":  "Technology",  "ORCL": "Technology",   "CRM":  "Technology",
    "ADBE": "Technology",  "INTC": "Technology",   "QCOM": "Technology",
    "TXN":  "Technology",  "IBM":  "Technology",   "NOW":  "Technology",
    "AMAT": "Technology",  "MU":   "Technology",   "LRCX": "Technology",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD":   "Consumer Discretionary", "NKE":  "Consumer Discretionary",
    "TGT":  "Consumer Discretionary", "F":    "Consumer Discretionary",
    "GM":   "Consumer Discretionary", "LOW":  "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    # Financials
    "JPM":  "Financials", "BAC":  "Financials", "GS":   "Financials",
    "WFC":  "Financials", "C":    "Financials", "MS":   "Financials",
    "BLK":  "Financials", "SCHW": "Financials", "AXP":  "Financials",
    "USB":  "Financials", "PNC":  "Financials", "COF":  "Financials",
    "MET":  "Financials", "PRU":  "Financials", "TFC":  "Financials",
    # Healthcare
    "UNH":  "Healthcare", "JNJ":  "Healthcare", "LLY":  "Healthcare",
    "ABBV": "Healthcare", "PFE":  "Healthcare", "MRK":  "Healthcare",
    "TMO":  "Healthcare", "MDT":  "Healthcare", "AMGN": "Healthcare",
    "BMY":  "Healthcare", "GILD": "Healthcare", "CVS":  "Healthcare",
    # Consumer Staples
    "PG":   "Consumer Staples", "KO":   "Consumer Staples",
    "PEP":  "Consumer Staples", "WMT":  "Consumer Staples",
    "COST": "Consumer Staples", "MCD":  "Consumer Staples",
    "SBUX": "Consumer Staples", "PM":   "Consumer Staples",
    # Energy
    "XOM":  "Energy", "CVX":  "Energy", "COP":  "Energy",
    "SLB":  "Energy", "EOG":  "Energy", "PSX":  "Energy",
    "VLO":  "Energy", "OXY":  "Energy",
    # Industrials
    "CAT":  "Industrials", "HON":  "Industrials", "UPS":  "Industrials",
    "BA":   "Industrials", "GE":   "Industrials", "MMM":  "Industrials",
    "LMT":  "Industrials", "RTX":  "Industrials", "DE":   "Industrials",
    "EMR":  "Industrials",
    # Communication Services
    "T":    "Communication Services", "VZ":    "Communication Services",
    "CMCSA":"Communication Services", "NFLX":  "Communication Services",
    "DIS":  "Communication Services", "CHTR":  "Communication Services",
    "FOXA": "Communication Services",
    # Materials
    "LIN":  "Materials", "SHW":  "Materials", "APD":  "Materials",
    "FCX":  "Materials", "NEM":  "Materials",
    # Real Estate
    "PLD":  "Real Estate", "AMT":  "Real Estate",
    "CCI":  "Real Estate", "EQIX": "Real Estate",
    # Utilities
    "NEE":  "Utilities", "DUK":  "Utilities",
    "SO":   "Utilities", "AEP":  "Utilities",
}
"""GICS sector classification for all tickers in TICKERS.
Used by the factor engine for sector-neutral signal normalisation."""

TICKER_ALIASES: dict[str, list[str]] = {
    "AAPL":  ["Apple"],
    "MSFT":  ["Microsoft"],
    "GOOGL": ["Google", "Alphabet"],
    "AMZN":  ["Amazon"],
    "JPM":   ["JPMorgan", "JP Morgan", "JPMorgan Chase"],
    "GS":    ["Goldman Sachs", "Goldman"],
    "BAC":   ["Bank of America", "BofA"],
    "TSLA":  ["Tesla"],
    "NVDA":  ["Nvidia", "NVIDIA"],
    "META":  ["Meta", "Meta Platforms", "Facebook"],
}
"""Maps each ticker to a list of company name strings used for RSS headline matching."""

RSS_FEEDS: dict[str, list[str]] = {
    # India — markets feed + tech feed (covers Apple, Google, Microsoft heavily)
    "economic_times": [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms",
    ],
    # China — China Daily general (300 entries, English)
    "china_daily": [
        "https://www.chinadaily.com.cn/rss/cndy_rss.xml",
    ],
    # Japan — Nikkei Asia (English, 50 entries, financial focus)
    "nikkei": [
        "https://asia.nikkei.com/rss/feed/nar",
    ],
}
"""RSS feed URLs for international news sources.  Each key maps to a list
of feed URLs that are all fetched and merged.  Update if a URL changes."""
"""RSS feed URLs for international news sources.  Update if a URL changes."""

SOURCE_WEIGHTS: dict[str, float] = {
    "sec_edgar":       1.0,
    "reuters":         0.9,
    "financial_times": 0.9,
    "nyt":             0.85,
    "guardian":        0.8,
    "economic_times":  0.75,
    "nikkei":          0.75,
    "caixin":          0.75,
    "japan_times":     0.70,
    "china_daily":     0.65,
    "yahoo_finance":   0.7,
    "newsapi":         0.6,
    "fnspid":          0.5,
    "reddit":          0.3,
}
"""Reliability weight assigned to each news/data source.

Scores range from 0.0 (untrusted) to 1.0 (fully trusted) and are used
to down-weight sentiment signals from lower-quality sources.
"""

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------

NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
"""NewsAPI.org key. Set in .env as NEWS_API_KEY=your-key"""

GUARDIAN_API_KEY: str = os.getenv("GUARDIAN_API_KEY", "")
"""The Guardian Open Platform key. Set in .env as GUARDIAN_API_KEY=your-key
Register free at https://open-platform.theguardian.com/access/"""

NYT_API_KEY: str = os.getenv("NYT_API_KEY", "")
"""New York Times Article Search API key. Set in .env as NYT_API_KEY=your-key
Register free at https://developer.nytimes.com/"""

FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
"""St. Louis FRED API key for macroeconomic data (CPI, unemployment, GDP, etc.).
Register free at https://fred.stlouisfed.org/docs/api/api_key.html
Set in .env as FRED_API_KEY=your-key.  Module gracefully degrades without it."""

# ---------------------------------------------------------------------------
# Macroeconomic data parameters
# ---------------------------------------------------------------------------

MACRO_LOOKBACK_DAYS: int = 730
"""Number of calendar days of macroeconomic history to retrieve."""

MACRO_REGIME_VIX_RISK_OFF: float = 28.0
"""VIX level above which the macro regime is classified as RISK_OFF."""

MACRO_REGIME_CPI_SHOCK: float = 0.04
"""CPI year-over-year rate above which INFLATION_SHOCK regime can be triggered."""

# ---------------------------------------------------------------------------
# Commodity data parameters
# ---------------------------------------------------------------------------

COMMODITY_LOOKBACK_DAYS: int = 365
"""Number of calendar days of commodity price history to retrieve."""

COMMODITY_SHOCK_THRESHOLD: float = 0.05
"""Minimum 5-day percentage price change to classify a commodity shock."""

# ---------------------------------------------------------------------------
# Model trust / quality scoring weights
# ---------------------------------------------------------------------------

TRUST_WEIGHT_PREDICTIVE: float = 0.40
"""Weight of predictive (classification / IC) sub-score in overall quality."""

TRUST_WEIGHT_RISK: float = 0.35
"""Weight of risk-adjusted performance sub-score in overall quality."""

TRUST_WEIGHT_ROBUSTNESS: float = 0.25
"""Weight of signal robustness / decay sub-score in overall quality."""

# ---------------------------------------------------------------------------
# Enhanced model parameters
# ---------------------------------------------------------------------------

ENHANCED_MODEL_FORWARD_DAYS: int = 5
"""Number of trading days ahead for forward-return label in enhanced model training."""

ENHANCED_SIGNAL_MAX_ADJUSTMENT: float = 0.25
"""Maximum absolute delta applied to the base FinBERT signal score by context adjustment."""

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

RESEARCH_DATA_DIR: Path = PROCESSED_DATA_DIR / "research_dataset_pack"
"""Deterministic offline dataset pack for reproducible research diagnostics."""

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
    RESEARCH_DATA_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)
