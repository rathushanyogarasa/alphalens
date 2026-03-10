"""fetch_gdelt_news.py
====================
Fetches financial headlines from the GDELT 2.0 Document API for all tickers
in config.TICKERS and appends them to combined_news.csv.

GDELT is completely free, requires no API key, and covers 2015→present
updated every 15 minutes.  Each query can return up to 250 articles per
ticker with title, URL, date, tone, and domain metadata.

Why GDELT beats NewsAPI for quant research
------------------------------------------
  • No API key, no rate limit (within reason)
  • Historical depth to 2015
  • 500–2000 headlines per ticker (vs ~10 on NewsAPI free plan)
  • Built-in tone scores (avg tone, positive score, negative score)
  • Entity metadata (companies, countries, themes)

Usage
-----
    python fetch_gdelt_news.py                         # last 90 days
    python fetch_gdelt_news.py --timespan 12months     # last 12 months
    python fetch_gdelt_news.py --timespan 1year        # last 1 year (max)
    python fetch_gdelt_news.py --max-records 250       # 250 per ticker (GDELT max)

GDELT timespan formats
----------------------
    "15min", "1h", "1d", "7d", "30d", "3months", "6months", "1year"
    (GDELT full-text search supports up to ~2 years of history)

Next step
---------
    python test_11_whites_reality_check.py --boot 2000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_gdelt")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import config

# ---------------------------------------------------------------------------
# GDELT 2.0 Document API
# ---------------------------------------------------------------------------

_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

_COMPANY_MAP: dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Alphabet",
    "META": "Meta Platforms", "AMZN": "Amazon", "TSLA": "Tesla",
    "AVGO": "Broadcom", "AMD": "AMD", "ORCL": "Oracle",
    "CRM": "Salesforce", "ADBE": "Adobe", "INTC": "Intel", "QCOM": "Qualcomm",
    "TXN": "Texas Instruments", "IBM": "IBM", "NOW": "ServiceNow",
    "AMAT": "Applied Materials", "MU": "Micron", "LRCX": "Lam Research",
    "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "WFC": "Wells Fargo", "C": "Citigroup", "MS": "Morgan Stanley",
    "BLK": "BlackRock", "SCHW": "Schwab", "AXP": "American Express",
    "USB": "US Bancorp", "PNC": "PNC Financial", "COF": "Capital One",
    "MET": "MetLife", "PRU": "Prudential", "TFC": "Truist",
    "UNH": "UnitedHealth", "JNJ": "Johnson Johnson", "LLY": "Eli Lilly",
    "ABBV": "AbbVie", "PFE": "Pfizer", "MRK": "Merck", "TMO": "Thermo Fisher",
    "MDT": "Medtronic", "AMGN": "Amgen", "BMY": "Bristol Myers",
    "GILD": "Gilead", "CVS": "CVS Health",
    "PG": "Procter Gamble", "KO": "Coca-Cola", "PEP": "PepsiCo",
    "WMT": "Walmart", "COST": "Costco", "MCD": "McDonald", "SBUX": "Starbucks",
    "PM": "Philip Morris", "HD": "Home Depot", "NKE": "Nike", "TGT": "Target",
    "F": "Ford Motor", "GM": "General Motors", "LOW": "Lowes", "BKNG": "Booking",
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "SLB": "Schlumberger", "EOG": "EOG Resources", "PSX": "Phillips 66",
    "VLO": "Valero", "OXY": "Occidental Petroleum",
    "CAT": "Caterpillar", "HON": "Honeywell", "UPS": "UPS", "BA": "Boeing",
    "GE": "GE Aerospace", "MMM": "3M", "LMT": "Lockheed Martin",
    "RTX": "Raytheon", "DE": "John Deere", "EMR": "Emerson Electric",
    "T": "AT&T", "VZ": "Verizon", "CMCSA": "Comcast", "NFLX": "Netflix",
    "DIS": "Disney", "CHTR": "Charter Communications", "FOXA": "Fox Corporation",
    "LIN": "Linde", "SHW": "Sherwin-Williams", "APD": "Air Products",
    "FCX": "Freeport McMoRan", "NEM": "Newmont",
    "PLD": "Prologis", "AMT": "American Tower", "CCI": "Crown Castle",
    "EQIX": "Equinix", "NEE": "NextEra Energy", "DUK": "Duke Energy",
    "SO": "Southern Company", "AEP": "American Electric Power",
}


def fetch_ticker_gdelt(
    ticker: str,
    timespan: str,
    max_records: int,
) -> list[dict]:
    """Fetch up to max_records articles for ticker from GDELT."""
    query = _COMPANY_MAP.get(ticker, ticker)
    params = {
        "query":      query,
        "mode":       "ArtList",
        "maxrecords": max_records,
        "timespan":   timespan,
        "sourcelang": "english",
        "format":     "json",
    }
    for attempt in range(3):
        try:
            resp = requests.get(_GDELT_URL, params=params, timeout=25)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)  # 15s, 30s, 45s
                logger.warning("GDELT 429 for %s (attempt %d) — sleeping %ds",
                               ticker, attempt + 1, wait)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.warning("GDELT HTTP %d for %s: %s",
                               resp.status_code, ticker, resp.text[:120])
                return []
            data = resp.json()
            articles = data.get("articles") or []
            break
        except Exception as exc:
            logger.warning("GDELT request failed for %s: %s", ticker, exc)
            return []
    else:
        logger.warning("GDELT: all retries exhausted for %s", ticker)
        return []

    rows: list[dict] = []
    for art in articles:
        title = (art.get("title") or "").strip()
        pub   = art.get("seendate", "")
        if not title or title.lower() == "[removed]":
            continue
        try:
            # GDELT format: "20240315T120000Z"
            date = pd.Timestamp(pub, tz="UTC").tz_localize(None)
        except Exception:
            try:
                date = pd.Timestamp(pub[:8])
            except Exception:
                continue
        rows.append({
            "date":     date,
            "ticker":   ticker,
            "headline": title,
            "source":   "gdelt",
            # GDELT extra metadata — useful for future feature engineering
            "url":      art.get("url", ""),
            "domain":   art.get("domain", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch financial headlines from GDELT 2.0")
    parser.add_argument(
        "--timespan", default="3months",
        help="GDELT timespan: e.g. 30d, 3months, 6months, 1year  (default: 3months)",
    )
    parser.add_argument(
        "--max-records", type=int, default=250,
        help="Max articles per ticker (GDELT hard cap: 250, default: 250)",
    )
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="Override tickers (default: all in config.TICKERS)",
    )
    parser.add_argument(
        "--sleep", type=float, default=15.0,
        help="Seconds between ticker requests (default: 15 — GDELT enforces ~1 req/10-15s)",
    )
    args = parser.parse_args()

    tickers    = args.tickers or config.TICKERS
    timespan   = args.timespan
    max_rec    = min(args.max_records, 250)
    sleep_sec  = args.sleep

    news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
    if news_path.exists():
        existing = pd.read_csv(news_path, parse_dates=["date"])
        logger.info("Loaded %d existing rows from combined_news.csv", len(existing))
    else:
        existing = pd.DataFrame(columns=["date", "ticker", "headline", "source"])

    est_mins = round(len(tickers) * sleep_sec / 60, 1)
    print(f"\n  GDELT Historical Fetch")
    print(f"  Timespan  : {timespan}")
    print(f"  Max/ticker: {max_rec}")
    print(f"  Tickers   : {len(tickers)}")
    print(f"  Sleep/req : {sleep_sec}s")
    print(f"  Est. time : ~{est_mins} minutes")
    print(f"  Est. total: up to {len(tickers) * max_rec:,} articles\n")

    all_rows: list[dict] = []
    for i, ticker in enumerate(tickers):
        rows = fetch_ticker_gdelt(ticker, timespan, max_rec)
        all_rows.extend(rows)
        logger.info("[%3d/%d] %-6s  %3d articles  (running: %d)",
                    i + 1, len(tickers), ticker, len(rows), len(all_rows))
        time.sleep(sleep_sec)

    if not all_rows:
        print("  No articles fetched — check network access to api.gdeltproject.org")
        return

    # Build DataFrame, drop URL/domain before merge (not in existing schema)
    new_df = pd.DataFrame(all_rows)
    new_df  = new_df[["date", "ticker", "headline", "source"]]

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.tz_localize(None)
    combined = combined.dropna(subset=["date", "headline"])
    combined["headline"] = combined["headline"].str.strip()

    before = len(combined)
    combined = (
        combined
        .sort_values("date")
        .drop_duplicates(subset=["ticker", "headline"], keep="first")
        .reset_index(drop=True)
    )
    after = len(combined)

    combined.to_csv(news_path, index=False)

    print("\n" + "=" * 60)
    print("  GDELT Fetch Complete")
    print("=" * 60)
    print(f"  New articles fetched     : {len(all_rows):,}")
    print(f"  Duplicates removed       : {before - after:,}")
    print(f"  Total in combined_news   : {len(combined):,}")
    print(f"  Tickers with data        : {combined['ticker'].nunique()}")
    print(f"  Date range               : {combined['date'].min().date()} -> {combined['date'].max().date()}")
    per_ticker = combined.groupby("ticker").size()
    print(f"  Median headlines/ticker  : {per_ticker.median():.0f}")
    print(f"  Min headlines/ticker     : {per_ticker.min()}")
    print(f"  Max headlines/ticker     : {per_ticker.max()}")
    print(f"\n  Source breakdown:")
    for src, cnt in combined["source"].value_counts().items():
        print(f"    {src:<20} {cnt:>6,}")
    print(f"\n  Next step: python test_11_whites_reality_check.py --boot 2000")
    print()


if __name__ == "__main__":
    main()
