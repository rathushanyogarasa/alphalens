"""Data source connectors for AlphaLens.

Provides nine news/filing connectors that all emit a unified schema:
    date (datetime64[ns]), ticker (str), headline (str), source (str)

Sources (API-key required):
    - The Guardian:   Guardian Open Platform API  (free key)
    - New York Times: NYT Article Search API      (free key)
    - NewsAPI:        newsapi.org                 (free key)

Sources (no key — RSS):
    - Yahoo Finance:    real headlines via yfinance
    - Economic Times:   India's #1 financial newspaper
    - China Daily:      English-language Chinese business news
    - Japan Times:      English-language Japanese business news

Sources (always available):
    - FNSPID:     local CSV or synthetic fallback
    - SEC EDGAR:  8-K filing full-text search API
"""

import logging
import random
import time
from pathlib import Path

import re

import feedparser
import pandas as pd
import requests
import yfinance as yf

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
# Schema helpers
# ---------------------------------------------------------------------------

_SCHEMA_COLS: list[str] = ["date", "ticker", "headline", "source"]


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the unified news schema.

    Returns:
        pd.DataFrame: Zero-row DataFrame with columns
            ``date``, ``ticker``, ``headline``, ``source``.
    """
    return pd.DataFrame(columns=_SCHEMA_COLS).astype(
        {"date": "datetime64[ns]", "ticker": str, "headline": str, "source": str}
    )


def _enforce_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Coerce *df* to the unified schema, tagging rows with *source*.

    Args:
        df: Raw DataFrame to normalise.
        source: Source tag to fill into the ``source`` column.

    Returns:
        pd.DataFrame: DataFrame containing only the four schema columns
        with correct dtypes.
    """
    df = df.copy()
    df["source"] = source
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    df["date"] = df["date"].dt.tz_localize(None)  # strip tz if present
    for col in ("ticker", "headline"):
        df[col] = df[col].astype(str)
    return df[_SCHEMA_COLS]


# ---------------------------------------------------------------------------
# Headline templates for synthetic data
# ---------------------------------------------------------------------------

_TEMPLATES: list[str] = [
    "{ticker} reports record quarterly earnings",
    "{ticker} misses revenue estimates by 8%",
    "{ticker} raises full year guidance",
    "{ticker} announces share buyback programme",
    "{ticker} faces SEC investigation over accounting",
    "{ticker} CEO resigns amid restructuring",
    "{ticker} beats earnings expectations",
    "{ticker} cuts dividend amid cash concerns",
    "{ticker} completes acquisition of rival firm",
    "{ticker} shares fall on weak guidance",
    "{ticker} profit margins under pressure",
    "{ticker} reports stronger than expected revenue",
    "{ticker} announces major cost cutting programme",
    "{ticker} stock upgraded by Goldman Sachs",
    "{ticker} faces antitrust investigation",
]


# ---------------------------------------------------------------------------
# 1. FNSPID connector
# ---------------------------------------------------------------------------


def load_fnspid(filepath: Path | None = None) -> pd.DataFrame:
    """Load the FNSPID financial news dataset or generate synthetic data.

    If *filepath* is provided and exists the function loads the CSV and
    standardises its columns to the unified schema.  Otherwise 500
    synthetic headlines are generated from :data:`_TEMPLATES` spread
    across ``config.TICKERS`` with random dates between
    2022-01-01 and 2024-01-01.

    Args:
        filepath: Optional path to a local FNSPID CSV file.  The CSV is
            expected to contain at least ``date``, ``ticker``, and
            ``headline`` columns (or common variants thereof).

    Returns:
        pd.DataFrame: Unified schema DataFrame with ``source="fnspid"``.
    """
    if filepath is not None and Path(filepath).exists():
        logger.info("Loading FNSPID from file: %s", filepath)
        raw = pd.read_csv(filepath)

        # Normalise common column name variants
        col_map: dict[str, str] = {}
        for col in raw.columns:
            low = col.lower()
            if low in {"date", "datetime", "published_at", "publishedat"}:
                col_map[col] = "date"
            elif low in {"ticker", "symbol", "stock"}:
                col_map[col] = "ticker"
            elif low in {"headline", "title", "text", "news"}:
                col_map[col] = "headline"
        raw = raw.rename(columns=col_map)

        missing = set(_SCHEMA_COLS) - {"source"} - set(raw.columns)
        if missing:
            logger.warning("FNSPID file missing columns %s — falling back to synthetic", missing)
        else:
            df = _enforce_schema(raw, source="fnspid")
            logger.info("FNSPID loaded from file: %d rows", len(df))
            return df

    # --- Synthetic fallback ---
    logger.info("FNSPID file not found — generating 500 synthetic headlines")
    rng = random.Random(config.RANDOM_SEED)
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2024-01-01")
    span_days = (end - start).days

    rows: list[dict] = []
    for _ in range(500):
        ticker = rng.choice(config.TICKERS)
        template = rng.choice(_TEMPLATES)
        offset = rng.randint(0, span_days)
        date = start + pd.Timedelta(days=offset)
        rows.append(
            {
                "date": date,
                "ticker": ticker,
                "headline": template.format(ticker=ticker),
            }
        )

    df = pd.DataFrame(rows)
    df = _enforce_schema(df, source="fnspid")
    logger.info("FNSPID synthetic dataset: %d rows across %d tickers", len(df), len(config.TICKERS))
    return df


# ---------------------------------------------------------------------------
# 2. Yahoo Finance connector
# ---------------------------------------------------------------------------


def fetch_yahoo_finance_news(tickers: list[str] | None = None) -> pd.DataFrame:
    """Fetch recent news headlines from Yahoo Finance via yfinance.

    Yahoo Finance aggregates real wire-service content from Reuters,
    Benzinga, AP, and other financial publishers.  No API key is required.
    Returns up to ~20 recent articles per ticker (typically covering the
    last 30 days depending on availability).

    Args:
        tickers: List of ticker symbols to query.  Defaults to
            ``config.TICKERS`` when ``None``.

    Returns:
        pd.DataFrame: Unified schema DataFrame with ``source="yahoo_finance"``,
            or an empty DataFrame if no articles are returned.
    """
    tickers = tickers or config.TICKERS
    rows: list[dict] = []

    for ticker in tickers:
        try:
            yt = yf.Ticker(ticker)
            articles = yt.news or []
            count = 0
            for art in articles:
                # Support both old schema (flat) and new schema (nested under "content")
                content = art.get("content", art)
                title = content.get("title", "").strip()
                if not title:
                    continue
                # New schema uses ISO pubDate; old schema uses providerPublishTime (Unix)
                pub_date = content.get("pubDate") or content.get("displayTime")
                if pub_date:
                    date = pd.Timestamp(pub_date)
                else:
                    publish_time = art.get("providerPublishTime")
                    date = pd.Timestamp(publish_time, unit="s") if publish_time else pd.NaT
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "headline": title,
                    }
                )
                count += 1
            logger.info("Yahoo Finance [%s]: %d articles", ticker, count)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Yahoo Finance fetch failed for %s: %s", ticker, exc)

    if not rows:
        logger.warning("Yahoo Finance returned no articles for any ticker")
        return _empty_df()

    df = pd.DataFrame(rows)
    df = _enforce_schema(df, source="yahoo_finance")
    # Drop duplicates — same story often appears under multiple tickers
    df = df.drop_duplicates(subset=["headline"])
    logger.info("Yahoo Finance total: %d rows (after dedup)", len(df))
    return df


# ---------------------------------------------------------------------------
# 3. The Guardian connector
# ---------------------------------------------------------------------------


def fetch_guardian_news(tickers: list[str] | None = None) -> pd.DataFrame:
    """Fetch business headlines from The Guardian Open Platform API.

    Searches the Guardian's ``/search`` endpoint restricted to the
    ``business`` section for each ticker symbol.  Returns up to 50
    articles per ticker from the last 90 days.

    Requires ``config.GUARDIAN_API_KEY``.  Register for a free key at
    https://open-platform.theguardian.com/access/

    Args:
        tickers: List of ticker symbols to query.  Defaults to
            ``config.TICKERS`` when ``None``.

    Returns:
        pd.DataFrame: Unified schema DataFrame with ``source="guardian"``,
            or an empty DataFrame if the key is absent or all requests fail.
    """
    if not config.GUARDIAN_API_KEY:
        logger.warning(
            "GUARDIAN_API_KEY is empty — skipping Guardian fetch. "
            "Register free at https://open-platform.theguardian.com/access/"
        )
        return _empty_df()

    tickers = tickers or config.TICKERS
    rows: list[dict] = []
    base_url = "https://content.guardianapis.com/search"
    from_date = (pd.Timestamp.utcnow() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")

    for ticker in tickers:
        params = {
            "q": ticker,
            "section": "business",
            "from-date": from_date,
            "show-fields": "headline,trailText",
            "page-size": 50,
            "order-by": "newest",
            "api-key": config.GUARDIAN_API_KEY,
        }
        try:
            resp = requests.get(base_url, params=params, timeout=10)
            if resp.status_code == 429:
                logger.warning("Guardian rate limit hit for %s — sleeping 5 s", ticker)
                time.sleep(5)
                resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("response", {}).get("results", [])
            for item in results:
                fields = item.get("fields", {})
                headline = fields.get("headline") or item.get("webTitle", "")
                pub_date = item.get("webPublicationDate", "")
                if headline.strip():
                    rows.append(
                        {
                            "date": pub_date,
                            "ticker": ticker,
                            "headline": headline.strip(),
                        }
                    )
            logger.info("Guardian [%s]: %d articles", ticker, len(results))
        except requests.RequestException as exc:
            logger.warning("Guardian request failed for %s: %s", ticker, exc)
        time.sleep(0.1)  # stay well within 12 req/s limit

    if not rows:
        logger.warning("Guardian returned no articles for any ticker")
        return _empty_df()

    df = pd.DataFrame(rows)
    df = _enforce_schema(df, source="guardian")
    df = df.drop_duplicates(subset=["headline"])
    logger.info("Guardian total: %d rows (after dedup)", len(df))
    return df


# ---------------------------------------------------------------------------
# 4. New York Times connector
# ---------------------------------------------------------------------------


def fetch_nyt_news(tickers: list[str] | None = None) -> pd.DataFrame:
    """Fetch business headlines from the New York Times Article Search API.

    Searches the NYT ``/articlesearch.json`` endpoint filtered to the
    Business section for each ticker symbol.  Returns up to 10 articles
    per ticker (1 page).  A 12-second sleep is inserted between requests
    to respect the 5 calls/minute rate limit.

    Requires ``config.NYT_API_KEY``.  Register for a free key at
    https://developer.nytimes.com/

    Args:
        tickers: List of ticker symbols to query.  Defaults to
            ``config.TICKERS`` when ``None``.

    Returns:
        pd.DataFrame: Unified schema DataFrame with ``source="nyt"``,
            or an empty DataFrame if the key is absent or all requests fail.
    """
    if not config.NYT_API_KEY:
        logger.warning(
            "NYT_API_KEY is empty — skipping NYT fetch. "
            "Register free at https://developer.nytimes.com/"
        )
        return _empty_df()

    tickers = tickers or config.TICKERS
    rows: list[dict] = []
    base_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

    for ticker in tickers:
        # Use company name + financial keywords to surface relevant articles
        aliases = config.TICKER_ALIASES.get(ticker, [ticker])
        company = aliases[0]
        query = f'"{company}" AND (stock OR earnings OR shares OR revenue OR profit OR CEO)'
        params = {
            "q": query,
            "sort": "newest",
            "api-key": config.NYT_API_KEY,
        }
        try:
            resp = requests.get(base_url, params=params, timeout=10)
            if resp.status_code == 429:
                logger.warning("NYT rate limit hit for %s — sleeping 60 s", ticker)
                time.sleep(60)
                resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            docs = resp.json().get("response", {}).get("docs") or []
            for doc in docs:
                headline = doc.get("headline", {}).get("main", "").strip()
                pub_date = doc.get("pub_date", "")
                if headline:
                    rows.append(
                        {
                            "date": pub_date,
                            "ticker": ticker,
                            "headline": headline,
                        }
                    )
            logger.info("NYT [%s]: %d articles", ticker, len(docs))
        except requests.RequestException as exc:
            logger.warning("NYT request failed for %s: %s", ticker, exc)
        time.sleep(12)  # NYT enforces 5 requests/minute

    if not rows:
        logger.warning("NYT returned no articles for any ticker")
        return _empty_df()

    df = pd.DataFrame(rows)
    df = _enforce_schema(df, source="nyt")
    df = df.drop_duplicates(subset=["headline"])
    logger.info("NYT total: %d rows (after dedup)", len(df))
    return df


# ---------------------------------------------------------------------------
# 5. RSS helper + international connectors
# ---------------------------------------------------------------------------


def _fetch_rss_feed(
    feed_urls: str | list[str],
    source_name: str,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch and filter one or more RSS feeds, matching entries to tracked tickers.

    Parses each RSS/Atom feed using ``feedparser``.  Each entry whose title
    or summary contains a ticker symbol or a company-name alias (from
    ``config.TICKER_ALIASES``) is retained.  Word-boundary regex matching
    prevents short tickers (e.g. ``GS``) from matching substrings.
    Entries mentioning multiple tickers are duplicated once per match.

    Args:
        feed_urls: A single URL string or a list of RSS/Atom feed URLs.
        source_name: Source tag written to the ``source`` column
            (e.g. ``"economic_times"``).
        tickers: Subset of tickers to match.  Defaults to
            ``config.TICKERS`` when ``None``.

    Returns:
        pd.DataFrame: Unified schema DataFrame or an empty DataFrame if
        all feeds are unreachable or yield no matching entries.
    """
    import calendar

    tickers = tickers or config.TICKERS
    if isinstance(feed_urls, str):
        feed_urls = [feed_urls]

    all_entries: list = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RSS fetch failed for %s (%s): %s", source_name, url, exc)
            continue
        if feed.bozo and not feed.entries:
            logger.warning("RSS feed %s (%s) returned no entries", source_name, url)
            continue
        all_entries.extend(feed.entries)
        logger.debug("RSS [%s] fetched %d entries from %s", source_name, len(feed.entries), url)

    if not all_entries:
        logger.info("RSS [%s]: all feeds empty or unreachable", source_name)
        return _empty_df()

    rows: list[dict] = []
    for entry in all_entries:
        title = (entry.get("title") or "").strip()
        summary = entry.get("summary") or entry.get("description") or ""

        published = entry.get("published_parsed") or entry.get("updated_parsed")
        if published:
            date = pd.Timestamp(calendar.timegm(published), unit="s")
        else:
            date = pd.Timestamp.utcnow().normalize()

        original_text = f"{title} {summary}"
        for ticker in tickers:
            aliases = config.TICKER_ALIASES.get(ticker, [])
            patterns = [rf"\b{re.escape(ticker)}\b"] + [
                rf"\b{re.escape(alias)}\b" for alias in aliases
            ]
            if any(re.search(p, original_text, re.IGNORECASE) for p in patterns):
                rows.append({"date": date, "ticker": ticker, "headline": title})

    if not rows:
        logger.info("RSS [%s]: no entries matched tracked tickers (%d entries scanned)", source_name, len(all_entries))
        return _empty_df()

    df = pd.DataFrame(rows)
    df = _enforce_schema(df, source=source_name)
    df = df.drop_duplicates(subset=["ticker", "headline"])
    logger.info("RSS [%s]: %d matched rows from %d feed entries", source_name, len(df), len(all_entries))
    return df


def fetch_economic_times_news(tickers: list[str] | None = None) -> pd.DataFrame:
    """Fetch business and tech headlines from The Economic Times (India).

    The Economic Times is India's largest financial newspaper.  Two RSS
    feeds are fetched: the markets feed and the technology feed (which
    covers Apple, Google, Microsoft, and other tracked tickers heavily).
    No authentication required.

    Feed URLs are read from ``config.RSS_FEEDS["economic_times"]``.

    Args:
        tickers: Ticker symbols to match.  Defaults to ``config.TICKERS``.

    Returns:
        pd.DataFrame: Unified schema DataFrame with
        ``source="economic_times"``.
    """
    urls = config.RSS_FEEDS.get("economic_times", [])
    if not urls:
        logger.warning("No RSS URLs configured for economic_times")
        return _empty_df()
    return _fetch_rss_feed(urls, source_name="economic_times", tickers=tickers)


def fetch_china_daily_news(tickers: list[str] | None = None) -> pd.DataFrame:
    """Fetch English-language headlines from China Daily RSS feed.

    China Daily is China's state-run English-language newspaper with ~300
    entries per feed refresh.  Covers Chinese and global corporate news.
    No authentication required.

    Feed URLs are read from ``config.RSS_FEEDS["china_daily"]``.

    Args:
        tickers: Ticker symbols to match.  Defaults to ``config.TICKERS``.

    Returns:
        pd.DataFrame: Unified schema DataFrame with
        ``source="china_daily"``.
    """
    urls = config.RSS_FEEDS.get("china_daily", [])
    if not urls:
        logger.warning("No RSS URLs configured for china_daily")
        return _empty_df()
    return _fetch_rss_feed(urls, source_name="china_daily", tickers=tickers)


def fetch_nikkei_news(tickers: list[str] | None = None) -> pd.DataFrame:
    """Fetch English-language headlines from Nikkei Asia RSS feed.

    Nikkei Asia is Japan's leading English-language financial newspaper.
    Its RSS feed covers Asian and global financial markets, tech, and
    corporate news.  No authentication required.

    Feed URLs are read from ``config.RSS_FEEDS["nikkei"]``.

    Args:
        tickers: Ticker symbols to match.  Defaults to ``config.TICKERS``.

    Returns:
        pd.DataFrame: Unified schema DataFrame with ``source="nikkei"``.
    """
    urls = config.RSS_FEEDS.get("nikkei", [])
    if not urls:
        logger.warning("No RSS URLs configured for nikkei")
        return _empty_df()
    return _fetch_rss_feed(urls, source_name="nikkei", tickers=tickers)


# ---------------------------------------------------------------------------
# 6. NewsAPI connector
# ---------------------------------------------------------------------------


def fetch_newsapi(tickers: list[str] | None = None) -> pd.DataFrame:
    """Fetch recent financial headlines from NewsAPI.

    Queries NewsAPI's ``/v2/everything`` endpoint for each ticker over
    the last 30 days.  A 1-second sleep is inserted between requests to
    respect rate limits.

    If ``config.NEWS_API_KEY`` is empty the function logs a warning and
    returns an empty DataFrame so the pipeline can continue without a key.

    Args:
        tickers: List of ticker symbols to query.  Defaults to
            ``config.TICKERS`` when ``None``.

    Returns:
        pd.DataFrame: Unified schema DataFrame with ``source="newsapi"``,
            or an empty DataFrame if the key is absent or all requests fail.
    """
    if not config.NEWS_API_KEY:
        logger.warning(
            "NEWS_API_KEY is empty in config — skipping NewsAPI fetch. "
            "Set NEWS_API_KEY to enable live news collection."
        )
        return _empty_df()

    tickers = tickers or config.TICKERS
    rows: list[dict] = []
    # Free tier only supports /v2/top-headlines (not /v2/everything)
    base_url = "https://newsapi.org/v2/top-headlines"

    for ticker in tickers:
        # Search by company name for better results than raw ticker symbol
        aliases = config.TICKER_ALIASES.get(ticker, [ticker])
        query = aliases[0]  # use primary company name (e.g. "Apple" not "AAPL")
        params = {
            "q": query,
            "language": "en",
            "pageSize": 20,
            "apiKey": config.NEWS_API_KEY,
        }
        try:
            resp = requests.get(base_url, params=params, timeout=10)
            if resp.status_code == 429:
                logger.warning("NewsAPI rate limit hit for %s — sleeping 60 s", ticker)
                time.sleep(60)
                resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            for art in articles:
                title = (art.get("title") or "").strip()
                if title and "[Removed]" not in title:
                    rows.append(
                        {
                            "date": art.get("publishedAt", ""),
                            "ticker": ticker,
                            "headline": title,
                        }
                    )
            logger.info("NewsAPI [%s]: %d articles", ticker, len(articles))
        except requests.RequestException as exc:
            logger.warning("NewsAPI request failed for %s: %s", ticker, exc)

        time.sleep(1)

    if not rows:
        logger.warning("NewsAPI returned no articles for any ticker")
        return _empty_df()

    df = pd.DataFrame(rows)
    df = _enforce_schema(df, source="newsapi")
    logger.info("NewsAPI total: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 7. GDELT connector
# ---------------------------------------------------------------------------

_GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Map ticker → primary search term for GDELT (company name gives better recall)
_GDELT_COMPANY_MAP: dict[str, str] = {
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


def fetch_gdelt_news(
    tickers: list[str] | None = None,
    max_records: int = 250,
    lookback_days: int = 90,
    timespan: str | None = None,
) -> pd.DataFrame:
    """Fetch financial headlines from the GDELT 2.0 Document API.

    GDELT is completely free with no API key required, updated every 15 minutes,
    and has historical coverage back to 2015.  Each query returns up to 250
    articles with title, URL, date, and domain.

    Args:
        tickers: Tickers to fetch.  Defaults to ``config.TICKERS``.
        max_records: Articles per ticker (max 250 per GDELT limit).
        lookback_days: Days of history to search.  GDELT supports up to ~2 years
            via the ``timespan`` parameter.  Values up to 30 use ``{N}d`` format;
            larger values use ``{N}months``.
        timespan: Override lookback_days with an explicit GDELT timespan string,
            e.g. ``"12months"``, ``"1year"``, ``"30d"``.

    Returns:
        pd.DataFrame: Unified schema with ``source="gdelt"``.
    """
    tickers = tickers or config.TICKERS
    rows: list[dict] = []

    if timespan is None:
        if lookback_days <= 30:
            timespan = f"{lookback_days}d"
        else:
            months = max(1, round(lookback_days / 30))
            timespan = f"{months}months"

    logger.info("GDELT: fetching %d tickers | timespan=%s | max_records=%d",
                len(tickers), timespan, max_records)

    for i, ticker in enumerate(tickers):
        query_term = _GDELT_COMPANY_MAP.get(ticker, ticker)
        params = {
            "query":      query_term,
            "mode":       "ArtList",
            "maxrecords": max_records,
            "timespan":   timespan,
            "sourcelang": "english",
            "format":     "json",
        }
        try:
            resp = requests.get(_GDELT_DOC_URL, params=params, timeout=20)
            if resp.status_code == 429:
                logger.warning("GDELT rate-limited — sleeping 10s")
                time.sleep(10)
                resp = requests.get(_GDELT_DOC_URL, params=params, timeout=20)
            if resp.status_code != 200:
                logger.warning("GDELT HTTP %d for %s", resp.status_code, ticker)
                time.sleep(1)
                continue
            data = resp.json()
            articles = data.get("articles") or []
            ticker_rows = 0
            for art in articles:
                title = (art.get("title") or "").strip()
                pub   = art.get("seendate", "")  # format: "20240315T120000Z"
                if not title or title.lower() in ("[removed]", ""):
                    continue
                try:
                    # GDELT seendate: "20240315T120000Z"
                    date = pd.Timestamp(pub, tz="UTC").tz_localize(None)
                except Exception:
                    try:
                        date = pd.Timestamp(pub[:8])  # fallback: just YYYYMMDD
                    except Exception:
                        continue
                rows.append({
                    "date":     date,
                    "ticker":   ticker,
                    "headline": title,
                    "source":   "gdelt",
                })
                ticker_rows += 1
            logger.info("GDELT [%3d/%d] %-6s  %d articles", i + 1, len(tickers), ticker, ticker_rows)
        except Exception as exc:
            logger.warning("GDELT request failed for %s: %s", ticker, exc)

        time.sleep(6.0)  # GDELT enforces 1 req/5s

    if not rows:
        logger.warning("GDELT returned no articles")
        return _empty_df()

    df = pd.DataFrame(rows)
    df = _enforce_schema(df, source="gdelt")
    logger.info("GDELT total: %d rows across %d tickers", len(df), df["ticker"].nunique())
    return df


# ---------------------------------------------------------------------------
# 8. SEC EDGAR connector
# ---------------------------------------------------------------------------

_EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_HEADERS = {"User-Agent": "AlphaLens research@alphalens.ai (research use)"}
_EDGAR_ITEMS = ["2.02", "8.01"]


def _edgar_backoff_get(url: str, params: dict, max_retries: int = 5) -> requests.Response | None:
    """GET *url* with exponential backoff on HTTP 429 responses.

    Args:
        url: The endpoint URL.
        params: Query parameters to include in the request.
        max_retries: Maximum number of retry attempts before giving up.

    Returns:
        requests.Response | None: The successful response, or ``None``
        if all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=_EDGAR_HEADERS, timeout=6)
            if resp.status_code == 429:
                wait = 2**attempt
                logger.warning("EDGAR rate limit (429) — backing off %d s (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as exc:
            wait = 2**attempt
            logger.warning("EDGAR request error: %s — retrying in %d s", exc, wait)
            time.sleep(wait)
    logger.error("EDGAR: all %d retries exhausted", max_retries)
    return None


def fetch_sec_edgar(tickers: list[str] | None = None) -> pd.DataFrame:
    """Fetch 8-K filings from the SEC EDGAR full-text search API.

    Queries ``https://efts.sec.gov/LATEST/search-index`` for each ticker
    and each of the item types in :data:`_EDGAR_ITEMS` (2.02 earnings
    releases and 8.01 other events) filed in the last two years.

    Exponential backoff is applied on HTTP 429 rate-limit responses.
    If EDGAR is unreachable the function logs a warning and returns an
    empty DataFrame so the pipeline degrades gracefully.

    Args:
        tickers: List of ticker symbols to query.  Defaults to
            ``config.TICKERS`` when ``None``.

    Returns:
        pd.DataFrame: Unified schema DataFrame with ``source="sec_edgar"``,
            or an empty DataFrame if the API is unreachable.
    """
    if config.QUICK_TEST:
        logger.info("QUICK_TEST enabled — skipping SEC EDGAR fetch")
        return _empty_df()

    tickers = tickers or config.TICKERS
    rows: list[dict] = []
    date_from = (pd.Timestamp.now() - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    date_to = pd.Timestamp.now().strftime("%Y-%m-%d")

    for ticker in tickers:
        for item in _EDGAR_ITEMS:
            params = {
                "q": f'"{ticker}"',
                "dateRange": "custom",
                "startdt": date_from,
                "enddt": date_to,
                "forms": "8-K",
                "hits.hits.total.value": 1,
                "_source": "period_of_report,file_date,display_names,form_type",
            }
            resp = _edgar_backoff_get(_EDGAR_BASE, params)
            if resp is None:
                logger.warning("EDGAR unreachable for %s item %s — skipping", ticker, item)
                continue
            if not resp.ok:
                logger.warning(
                    "EDGAR %s item %s → HTTP %d — skipping", ticker, item, resp.status_code
                )
                continue

            try:
                data = resp.json()
            except ValueError:
                logger.warning("EDGAR response for %s item %s is not valid JSON", ticker, item)
                continue

            hits = data.get("hits", {}).get("hits", [])
            for hit in hits:
                src = hit.get("_source", {})
                file_date = src.get("file_date") or src.get("period_of_report", "")
                display_names = src.get("display_names", ticker)
                # Build a descriptive headline from filing metadata
                headline = (
                    f"{ticker} 8-K item {item} filing: {display_names}"
                    if display_names
                    else f"{ticker} 8-K item {item} filing"
                )
                rows.append(
                    {
                        "date": file_date,
                        "ticker": ticker,
                        "headline": headline,
                    }
                )

            logger.info("EDGAR [%s item %s]: %d filings", ticker, item, len(hits))
            time.sleep(0.5)  # polite delay between EDGAR requests

    if not rows:
        logger.warning("SEC EDGAR returned no filings for any ticker")
        return _empty_df()

    df = pd.DataFrame(rows)
    df = _enforce_schema(df, source="sec_edgar")
    logger.info("SEC EDGAR total: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 8. Combiner
# ---------------------------------------------------------------------------


def combine_sources(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple source DataFrames into one unified dataset.

    Steps:

    1. Drop empty DataFrames.
    2. Concatenate all remaining DataFrames.
    3. Drop rows with null ``headline`` or ``date``.
    4. Sort by ``date`` descending.
    5. Log headline counts per source.
    6. Save to ``config.PROCESSED_DATA_DIR / "combined_news.csv"``.

    Args:
        dfs: List of DataFrames produced by the individual connectors.
            Each must conform to the unified schema.

    Returns:
        pd.DataFrame: The combined, cleaned DataFrame sorted by date.
    """
    non_empty = [df for df in dfs if not df.empty]
    if not non_empty:
        logger.warning("All source DataFrames are empty — returning empty combined DataFrame")
        return _empty_df()

    combined = pd.concat(non_empty, ignore_index=True)
    before = len(combined)
    combined = combined.dropna(subset=["headline", "date"])
    combined = combined[combined["headline"].str.strip() != ""]
    after = len(combined)
    if before != after:
        logger.info("Dropped %d rows with null/empty headline or date", before - after)

    combined = combined.sort_values("date", ascending=False).reset_index(drop=True)

    for src, grp in combined.groupby("source"):
        logger.info("  [%s] %d headlines", src, len(grp))
    logger.info("Combined news total: %d headlines", len(combined))

    out_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    logger.info("Saved combined news → %s", out_path)

    return combined


# ---------------------------------------------------------------------------
# 9. Orchestrator
# ---------------------------------------------------------------------------


def run_data_sources() -> pd.DataFrame:
    """Run all nine connectors, combine results, and return the full dataset.

    Execution order:

    1.  FNSPID (local / synthetic fallback)
    2.  Yahoo Finance (no key required)
    3.  Economic Times RSS — India (no key required)
    4.  China Daily RSS — China (no key required)
    5.  Japan Times RSS — Japan (no key required)
    6.  The Guardian API (free key — skipped if absent)
    7.  New York Times API (free key — skipped if absent)
    8.  NewsAPI (free key — skipped if absent)
    9.  GDELT 2.0 Document API (no key required — best free source)
    10. SEC EDGAR (no key required)

    Returns:
        pd.DataFrame: The combined news DataFrame saved to
        ``config.PROCESSED_DATA_DIR / "combined_news.csv"``.
    """
    logger.info("=== AlphaLens Data Sources Pipeline ===")

    fnspid_df      = load_fnspid()
    yahoo_df       = fetch_yahoo_finance_news()
    et_df          = fetch_economic_times_news()
    china_df       = fetch_china_daily_news()
    nikkei_df      = fetch_nikkei_news()
    guardian_df    = fetch_guardian_news()
    nyt_df         = fetch_nyt_news()
    newsapi_df     = fetch_newsapi()
    gdelt_df       = fetch_gdelt_news()
    edgar_df       = fetch_sec_edgar()

    combined = combine_sources([
        fnspid_df, yahoo_df, et_df, china_df, nikkei_df,
        guardian_df, nyt_df, newsapi_df, gdelt_df, edgar_df,
    ])

    logger.info("=== Data sources pipeline complete ===")
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_data_sources()
