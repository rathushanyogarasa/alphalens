"""expand_universe.py
=====================
Fetches Yahoo Finance news for all tickers in config.TICKERS that are NOT
already present in combined_news.csv, then appends the new rows.

This is a one-shot script: run it once after expanding the TICKERS list.
It is safe to re-run — duplicate headlines are dropped automatically.

Usage
-----
    python expand_universe.py

What it does
------------
1. Load existing combined_news.csv (preserve all historical rows)
2. Identify new tickers (not yet in the file)
3. Fetch Yahoo Finance news for each new ticker via yfinance
4. Also refresh Yahoo Finance news for existing tickers (picks up recents)
5. Deduplicate on (ticker, headline) — keep earliest date for duplicates
6. Save back to combined_news.csv

After running, rerun test_09 and test_10 to see universe-expansion effect.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("expand_universe")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import config


# ---------------------------------------------------------------------------
# Yahoo Finance fetch (mirrors src/data_sources.py but standalone)
# ---------------------------------------------------------------------------


def fetch_yahoo_news_for_tickers(tickers: list[str]) -> pd.DataFrame:
    """Fetch recent Yahoo Finance headlines for each ticker.

    Returns a DataFrame with columns: date, ticker, headline, source.
    """
    rows: list[dict] = []
    for i, ticker in enumerate(tickers):
        try:
            yt = yf.Ticker(ticker)
            articles = yt.news or []
            count = 0
            for art in articles:
                content = art.get("content", art)
                title = content.get("title", "").strip()
                if not title:
                    continue
                pub_date = content.get("pubDate") or content.get("displayTime")
                if pub_date:
                    date = pd.Timestamp(pub_date)
                else:
                    ts = art.get("providerPublishTime")
                    date = pd.Timestamp(ts, unit="s") if ts else pd.NaT
                rows.append({
                    "date":     date,
                    "ticker":   ticker,
                    "headline": title,
                    "source":   "yahoo_finance",
                })
                count += 1
            logger.info("[%3d/%d] %-6s  %d articles", i + 1, len(tickers), ticker, count)
        except Exception as exc:
            logger.warning("[%3d/%d] %-6s  FAILED: %s", i + 1, len(tickers), ticker, exc)

        # Gentle rate-limit: 0.3s between requests
        time.sleep(0.3)

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "headline", "source"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date", "headline"])
    df = df[df["headline"].str.strip() != ""]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"

    # 1. Load existing data
    if news_path.exists():
        existing = pd.read_csv(news_path, parse_dates=["date"])
        logger.info("Existing combined_news.csv: %d rows, %d tickers",
                    len(existing), existing["ticker"].nunique())
    else:
        existing = pd.DataFrame(columns=["date", "ticker", "headline", "source"])
        logger.info("No existing combined_news.csv — starting fresh")

    existing_tickers = set(existing["ticker"].unique())
    all_tickers      = config.TICKERS
    new_tickers      = [t for t in all_tickers if t not in existing_tickers]

    logger.info("Config universe : %d tickers", len(all_tickers))
    logger.info("Already in CSV  : %d tickers", len(existing_tickers))
    logger.info("New tickers     : %d  -> %s", len(new_tickers), new_tickers)

    # 2. Fetch new tickers
    if new_tickers:
        logger.info("Fetching Yahoo Finance news for %d new tickers ...", len(new_tickers))
        new_df = fetch_yahoo_news_for_tickers(new_tickers)
        logger.info("Fetched %d new headlines", len(new_df))
    else:
        new_df = pd.DataFrame(columns=["date", "ticker", "headline", "source"])
        logger.info("No new tickers to fetch")

    # 3. Also refresh Yahoo Finance for existing tickers (pick up recent articles)
    logger.info("Refreshing Yahoo Finance for existing %d tickers ...", len(existing_tickers))
    existing_refresh = fetch_yahoo_news_for_tickers(list(existing_tickers))
    logger.info("Refresh yielded %d headlines", len(existing_refresh))

    # 4. Combine all
    combined = pd.concat([existing, new_df, existing_refresh], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date", "headline"])
    combined["headline"] = combined["headline"].str.strip()

    # Deduplicate: keep one row per (ticker, headline), earliest date wins
    before = len(combined)
    combined = (
        combined
        .sort_values("date")
        .drop_duplicates(subset=["ticker", "headline"], keep="first")
        .reset_index(drop=True)
    )
    after = len(combined)
    logger.info("Deduplication: %d -> %d rows (removed %d duplicates)", before, after, before - after)

    # 5. Save
    combined.to_csv(news_path, index=False)
    logger.info("Saved combined_news.csv: %d rows, %d tickers",
                len(combined), combined["ticker"].nunique())

    # Summary
    print("\n" + "=" * 60)
    print("  Universe Expansion Complete")
    print("=" * 60)
    print(f"  Total headlines  : {len(combined):,}")
    print(f"  Total tickers    : {combined['ticker'].nunique()}")
    print(f"  Date range       : {combined['date'].min().date()} -> {combined['date'].max().date()}")
    print()
    per_ticker = combined.groupby("ticker").size().sort_values(ascending=False)
    print(f"  Median headlines/ticker : {per_ticker.median():.0f}")
    print(f"  Min headlines/ticker    : {per_ticker.min()}")
    print(f"  Max headlines/ticker    : {per_ticker.max()}")
    print()
    print("  Next steps:")
    print("    python test_09_multifactor.py")
    print("    python test_10_alpha_decay.py")
    print()

    # Warn about tickers with very few headlines
    sparse = per_ticker[per_ticker < 5]
    if not sparse.empty:
        print(f"  WARNING: {len(sparse)} tickers have <5 headlines (weak signal):")
        for t, n in sparse.items():
            print(f"    {t}: {n}")
        print()


if __name__ == "__main__":
    main()
