"""fetch_historical_news.py
===========================
Fetches historical news headlines from NewsAPI for all tickers in
config.TICKERS and appends them to combined_news.csv.

Requires NEWS_API_KEY in .env (free plan: 1 month lookback, 100 req/day;
Developer plan: full history, 500 req/day).

Usage
-----
    python fetch_historical_news.py [--from 2022-01-01] [--to 2024-12-31]
    python fetch_historical_news.py --from 2022-01-01  # defaults to today

The script batches tickers into groups, rates itself to stay within the
API limit, and deduplicates before saving.  Safe to re-run.
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
logger = logging.getLogger("fetch_historical")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import config


# ---------------------------------------------------------------------------
# NewsAPI fetch
# ---------------------------------------------------------------------------

NEWSAPI_URL = "https://newsapi.org/v2/everything"


def _company_name(ticker: str) -> str:
    """Map ticker to a human-readable company name for the search query."""
    _MAP = {
        "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Alphabet Google",
        "META": "Meta Platforms Facebook", "AMZN": "Amazon", "TSLA": "Tesla",
        "AVGO": "Broadcom", "AMD": "Advanced Micro Devices AMD", "ORCL": "Oracle",
        "CRM": "Salesforce", "ADBE": "Adobe", "INTC": "Intel", "QCOM": "Qualcomm",
        "TXN": "Texas Instruments", "IBM": "IBM", "NOW": "ServiceNow", "AMAT": "Applied Materials",
        "MU": "Micron Technology", "LRCX": "Lam Research",
        "JPM": "JPMorgan Chase", "BAC": "Bank of America", "GS": "Goldman Sachs",
        "WFC": "Wells Fargo", "C": "Citigroup", "MS": "Morgan Stanley",
        "BLK": "BlackRock", "SCHW": "Charles Schwab", "AXP": "American Express",
        "USB": "US Bancorp", "PNC": "PNC Financial", "COF": "Capital One",
        "MET": "MetLife", "PRU": "Prudential Financial", "TFC": "Truist Financial",
        "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "LLY": "Eli Lilly",
        "ABBV": "AbbVie", "PFE": "Pfizer", "MRK": "Merck", "TMO": "Thermo Fisher",
        "MDT": "Medtronic", "AMGN": "Amgen", "BMY": "Bristol-Myers Squibb",
        "GILD": "Gilead Sciences", "CVS": "CVS Health",
        "PG": "Procter & Gamble", "KO": "Coca-Cola", "PEP": "PepsiCo",
        "WMT": "Walmart", "COST": "Costco", "MCD": "McDonald's", "SBUX": "Starbucks", "PM": "Philip Morris",
        "HD": "Home Depot", "NKE": "Nike", "TGT": "Target",
        "F": "Ford Motor", "GM": "General Motors", "LOW": "Lowe's", "BKNG": "Booking Holdings",
        "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
        "SLB": "Schlumberger SLB", "EOG": "EOG Resources", "PSX": "Phillips 66",
        "VLO": "Valero Energy", "OXY": "Occidental Petroleum",
        "CAT": "Caterpillar", "HON": "Honeywell", "UPS": "United Parcel Service",
        "BA": "Boeing", "GE": "GE Aerospace", "MMM": "3M", "LMT": "Lockheed Martin",
        "RTX": "RTX Raytheon", "DE": "John Deere", "EMR": "Emerson Electric",
        "T": "AT&T", "VZ": "Verizon", "CMCSA": "Comcast", "NFLX": "Netflix",
        "DIS": "Walt Disney", "CHTR": "Charter Communications", "FOXA": "Fox Corporation",
        "LIN": "Linde", "SHW": "Sherwin-Williams", "APD": "Air Products",
        "FCX": "Freeport-McMoRan", "NEM": "Newmont",
        "PLD": "Prologis", "AMT": "American Tower", "CCI": "Crown Castle", "EQIX": "Equinix",
        "NEE": "NextEra Energy", "DUK": "Duke Energy", "SO": "Southern Company", "AEP": "American Electric Power",
    }
    return _MAP.get(ticker, ticker)


def fetch_newsapi_ticker(
    ticker: str,
    api_key: str,
    from_date: str,
    to_date: str,
    page_size: int = 100,
    max_pages: int = 5,
) -> list[dict]:
    """Fetch up to page_size * max_pages articles for *ticker* from NewsAPI."""
    query = f'"{_company_name(ticker)}" OR "{ticker}"'
    rows: list[dict] = []

    for page in range(1, max_pages + 1):
        params = {
            "q":        query,
            "from":     from_date,
            "to":       to_date,
            "language": "en",
            "sortBy":   "relevancy",
            "pageSize": page_size,
            "page":     page,
            "apiKey":   api_key,
        }
        try:
            resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
            if resp.status_code == 429:
                logger.warning("Rate limited by NewsAPI — sleeping 60s")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                logger.warning("NewsAPI error %d for %s: %s",
                               resp.status_code, ticker, resp.text[:120])
                break
            data = resp.json()
            articles = data.get("articles", [])
            if not articles:
                break
            for art in articles:
                title = (art.get("title") or "").strip()
                pub   = art.get("publishedAt", "")
                if not title or title == "[Removed]":
                    continue
                try:
                    date = pd.Timestamp(pub)
                except Exception:
                    continue
                rows.append({
                    "date":     date,
                    "ticker":   ticker,
                    "headline": title,
                    "source":   "newsapi",
                })
            if len(articles) < page_size:
                break  # last page
        except Exception as exc:
            logger.warning("NewsAPI request failed for %s: %s", ticker, exc)
            break

        time.sleep(0.5)  # gentle rate limiting between pages

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical news via NewsAPI")
    # Free plan: max 30 days back.  Developer plan ($449/mo): full archive.
    default_from = (pd.Timestamp.today() - pd.Timedelta(days=29)).strftime("%Y-%m-%d")
    parser.add_argument("--from", dest="from_date", default=default_from,
                        help="Start date YYYY-MM-DD (free plan: last 30 days only)")
    parser.add_argument("--to",   dest="to_date",
                        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
                        help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    api_key = config.NEWS_API_KEY
    if not api_key:
        print("\n  ERROR: NEWS_API_KEY not set in .env")
        print("  Register free at https://newsapi.org and add:")
        print("    NEWS_API_KEY=your-key")
        print("  to your .env file, then re-run.\n")
        sys.exit(1)

    news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
    if news_path.exists():
        existing = pd.read_csv(news_path, parse_dates=["date"])
        logger.info("Loaded %d existing rows", len(existing))
    else:
        existing = pd.DataFrame(columns=["date", "ticker", "headline", "source"])

    print(f"\n  Fetching NewsAPI headlines {args.from_date} -> {args.to_date}")
    print(f"  Tickers: {len(config.TICKERS)}")
    print(f"  Estimated requests: {len(config.TICKERS) * 3}  (3 pages / ticker)")
    print(f"  (Free plan: 100 req/day — run in batches if needed)\n")

    all_new_rows: list[dict] = []
    for i, ticker in enumerate(config.TICKERS):
        rows = fetch_newsapi_ticker(
            ticker, api_key, args.from_date, args.to_date,
            page_size=100, max_pages=3,
        )
        all_new_rows.extend(rows)
        logger.info("[%3d/%d] %-6s  %d articles fetched  (running total: %d)",
                    i + 1, len(config.TICKERS), ticker, len(rows), len(all_new_rows))
        # Respect NewsAPI rate limits: ~1 req/sec on free plan
        time.sleep(1.2)

    if not all_new_rows:
        print("  No articles fetched. Check your API key and date range.")
        return

    new_df   = pd.DataFrame(all_new_rows)
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
    print("  Historical News Fetch Complete")
    print("=" * 60)
    print(f"  New headlines fetched    : {len(all_new_rows):,}")
    print(f"  Duplicates removed       : {before - after:,}")
    print(f"  Total in combined_news   : {len(combined):,}")
    print(f"  Tickers with data        : {combined['ticker'].nunique()}")
    print(f"  Date range               : {combined['date'].min().date()} -> {combined['date'].max().date()}")
    per_ticker = combined.groupby("ticker").size()
    print(f"  Median headlines/ticker  : {per_ticker.median():.0f}")
    print(f"\n  Next step: python test_11_whites_reality_check.py")
    print()


if __name__ == "__main__":
    main()
