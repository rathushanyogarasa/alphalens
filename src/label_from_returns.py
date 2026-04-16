"""Build outcome-labelled training corpus from news headlines and market returns.

Joins ``combined_news.csv`` to actual forward market returns, replacing
sentiment-based labels with market-outcome-based labels (outperform /
neutral / underperform relative to SPY).  The output is written to
``combined_news_labelled.csv`` and is intended as the training corpus
for FinBERT fine-tuning.

Typical usage::

    python src/label_from_returns.py --horizon 2
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.backtest import fetch_price_data

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
# Labelling constants
# ---------------------------------------------------------------------------

LABEL_HORIZON_DAYS: int = 2
"""Number of forward trading days used to compute return windows."""

OUTPERFORM_THRESHOLD: float = 0.01
"""Excess return (stock minus SPY) at or above this value → label +1."""

UNDERPERFORM_THRESHOLD: float = -0.01
"""Excess return at or below this value → label -1."""

_FORWARD_CALENDAR_BUFFER: int = 30
"""Calendar-day buffer appended to date_max when fetching prices, to ensure
the horizon window has data even near the end of the news date range."""

_SPY_EDGE_BUFFER: int = 10
"""Calendar-day buffer prepended/appended to the SPY fetch range."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fwd_log_return(
    price_series: pd.Series,
    sorted_index: pd.DatetimeIndex,
    date: pd.Timestamp,
    horizon: int,
) -> float | None:
    """Compute log return from *date* to *date + horizon* trading days.

    Args:
        price_series: Adjusted close prices, dropna-cleaned and sort_index-ordered.
        sorted_index: Pre-built index of *price_series* (passed separately to
            avoid redundant attribute lookups in the inner loop).
        date: Start date of the return window (must be a trading day).
        horizon: Number of forward trading days.

    Returns:
        Log return as a float, or ``None`` if either price is unavailable.
    """
    loc = sorted_index.searchsorted(date)
    if loc >= len(sorted_index) or sorted_index[loc] != date:
        return None
    end_loc = loc + horizon
    if end_loc >= len(sorted_index):
        return None
    p0 = price_series.loc[sorted_index[loc]]
    p1 = price_series.loc[sorted_index[end_loc]]
    if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
        return None
    return float(np.log(p1 / p0))


def _assign_label(excess_return: float) -> int:
    """Map an excess return to a ternary outcome label.

    Args:
        excess_return: Stock log return minus SPY log return over the
            forward window.

    Returns:
        ``1`` (outperform), ``-1`` (underperform), or ``0`` (neutral).
    """
    if excess_return >= OUTPERFORM_THRESHOLD:
        return 1
    if excess_return <= UNDERPERFORM_THRESHOLD:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Main labelling function
# ---------------------------------------------------------------------------


def build_labelled_corpus(horizon: int = LABEL_HORIZON_DAYS) -> pd.DataFrame:
    """Construct a market-outcome-labelled corpus from news and price data.

    Loads ``combined_news.csv``, fetches adjusted close prices for all
    tickers present in the file plus SPY, then computes *horizon*-day
    forward log returns for each ``(date, ticker)`` pair.  Rows where
    price data is unavailable for the stock or SPY are dropped and
    counted separately.  The final labelled DataFrame is written to
    ``combined_news_labelled.csv``.

    Args:
        horizon: Number of forward trading days for the return window.
            Overrides the module-level :data:`LABEL_HORIZON_DAYS` constant.

    Returns:
        Labelled DataFrame with columns: ``date``, ``ticker``, ``headline``,
        ``label`` (int: −1 / 0 / 1), ``fwd_return_stock``, ``fwd_return_spy``,
        ``excess_return``, and any additional columns from the input file.

    Raises:
        FileNotFoundError: If ``combined_news.csv`` is absent.
        RuntimeError: If SPY price data cannot be fetched.
    """
    news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
    if not news_path.exists():
        raise FileNotFoundError(
            f"combined_news.csv not found at {news_path}. "
            "Run src/data_sources.py first."
        )

    # ------------------------------------------------------------------
    # 1. Load and normalise news
    # ------------------------------------------------------------------
    logger.info("Loading news from %s", news_path)
    news_df = pd.read_csv(news_path, parse_dates=["date"])
    total_input = len(news_df)
    logger.info("Loaded %d rows", total_input)

    news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
    if news_df["date"].dt.tz is not None:
        news_df["date"] = news_df["date"].dt.tz_localize(None)

    # ------------------------------------------------------------------
    # 2. Determine date range and fetch prices
    # ------------------------------------------------------------------
    tickers: list[str] = news_df["ticker"].dropna().unique().tolist()
    logger.info("Unique tickers in news file: %d", len(tickers))

    date_min: pd.Timestamp = news_df["date"].min()
    date_max: pd.Timestamp = news_df["date"].max()
    logger.info("News date range: %s → %s", date_min.date(), date_max.date())

    forward_buf = pd.Timedelta(days=_FORWARD_CALENDAR_BUFFER)
    spy_edge = pd.Timedelta(days=_SPY_EDGE_BUFFER)

    stock_start = (date_min - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    stock_end = (date_max + forward_buf).strftime("%Y-%m-%d")
    spy_start = (date_min - spy_edge).strftime("%Y-%m-%d")
    spy_end = (date_max + spy_edge + forward_buf).strftime("%Y-%m-%d")

    logger.info(
        "Fetching stock prices for %d tickers: %s → %s",
        len(tickers), stock_start, stock_end,
    )
    prices = fetch_price_data(tickers, start=stock_start, end=stock_end)

    logger.info(
        "Fetching %s prices: %s → %s",
        config.MARKET_BENCHMARK, spy_start, spy_end,
    )
    spy_raw = fetch_price_data(
        [config.MARKET_BENCHMARK], start=spy_start, end=spy_end
    )

    if spy_raw.empty or config.MARKET_BENCHMARK not in spy_raw.columns:
        raise RuntimeError(
            f"No price data returned for {config.MARKET_BENCHMARK}. "
            "Check network connectivity or yfinance availability."
        )

    spy_series = spy_raw[config.MARKET_BENCHMARK].dropna().sort_index()
    spy_idx: pd.DatetimeIndex = spy_series.index

    # Pre-build per-ticker sorted series cache to avoid redundant work
    ticker_cache: dict[str, tuple[pd.Series, pd.DatetimeIndex]] = {}
    for tkr in tickers:
        if tkr in prices.columns:
            s = prices[tkr].dropna().sort_index()
            ticker_cache[tkr] = (s, s.index)

    missing_tickers = [t for t in tickers if t not in ticker_cache]
    if missing_tickers:
        logger.warning(
            "No price data available for %d ticker(s): %s",
            len(missing_tickers),
            missing_tickers[:20],
        )

    # ------------------------------------------------------------------
    # 3. Compute forward returns and assign labels
    # ------------------------------------------------------------------
    rows: list[dict] = []
    dropped_stock = 0
    dropped_spy = 0

    for _, row in news_df.iterrows():
        date: pd.Timestamp = row["date"]
        ticker: str = str(row["ticker"])

        if ticker not in ticker_cache:
            logger.debug("No price data for %s on %s — dropping", ticker, date.date())
            dropped_stock += 1
            continue

        stock_series, stock_idx = ticker_cache[ticker]
        fwd_stock = _fwd_log_return(stock_series, stock_idx, date, horizon)
        if fwd_stock is None:
            logger.debug(
                "Stock forward return unavailable: %s @ %s — dropping",
                ticker, date.date(),
            )
            dropped_stock += 1
            continue

        fwd_spy = _fwd_log_return(spy_series, spy_idx, date, horizon)
        if fwd_spy is None:
            logger.debug("SPY forward return unavailable @ %s — dropping", date.date())
            dropped_spy += 1
            continue

        excess = fwd_stock - fwd_spy
        label = _assign_label(excess)

        row_dict = row.to_dict()
        row_dict.update(
            {
                "fwd_return_stock": round(fwd_stock, 6),
                "fwd_return_spy": round(fwd_spy, 6),
                "excess_return": round(excess, 6),
                "label": label,
            }
        )
        rows.append(row_dict)

    logger.info(
        "Labelling complete — total=%d  dropped_stock=%d  dropped_spy=%d  labelled=%d",
        total_input, dropped_stock, dropped_spy, len(rows),
    )

    if not rows:
        logger.warning("No labelled rows produced — returning empty DataFrame")
        return pd.DataFrame()

    labelled_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 4. Reorder columns and write output
    # ------------------------------------------------------------------
    priority_cols = [
        "date", "ticker", "headline", "label",
        "fwd_return_stock", "fwd_return_spy", "excess_return",
    ]
    extra_cols = [c for c in labelled_df.columns if c not in priority_cols]
    labelled_df = labelled_df[priority_cols + extra_cols]
    labelled_df["label"] = labelled_df["label"].astype(int)
    labelled_df = labelled_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    out_path = config.PROCESSED_DATA_DIR / "combined_news_labelled.csv"
    labelled_df.to_csv(out_path, index=False)
    logger.info("Labelled corpus written → %s", out_path)

    # ------------------------------------------------------------------
    # 5. Final console summary (bare print as required by spec)
    # ------------------------------------------------------------------
    label_counts = labelled_df["label"].value_counts().sort_index()
    total_labelled = len(labelled_df)
    label_names = {-1: "underperform", 0: "neutral", 1: "outperform"}

    print("\n=== label_from_returns summary ===")
    print(f"  Total input rows         : {total_input:>8,}")
    print(f"  Dropped (stock unavail)  : {dropped_stock:>8,}")
    print(f"  Dropped (SPY unavail)    : {dropped_spy:>8,}")
    print(f"  Final labelled rows      : {total_labelled:>8,}")
    print(f"\n  Label distribution (horizon={horizon}d):")
    for lbl in [-1, 0, 1]:
        count = int(label_counts.get(lbl, 0))
        pct = count / total_labelled * 100 if total_labelled > 0 else 0.0
        print(
            f"    {lbl:>+2}  ({label_names[lbl]:<12}) : {count:>7,}  ({pct:5.1f}%)"
        )
    print(f"\n  Output: {out_path}")
    print("==================================\n")

    return labelled_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build a market-outcome-labelled training corpus by joining "
            "combined_news.csv to forward stock and SPY returns."
        )
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=LABEL_HORIZON_DAYS,
        help=(
            "Number of forward trading days for the return window "
            f"(default: {LABEL_HORIZON_DAYS})."
        ),
    )
    args = parser.parse_args()

    if args.horizon != LABEL_HORIZON_DAYS:
        logger.info(
            "Horizon overridden via CLI: %d (module default: %d)",
            args.horizon, LABEL_HORIZON_DAYS,
        )

    build_labelled_corpus(horizon=args.horizon)
