"""Data source connectors for AlphaLens.

Provides three news/filing connectors that all emit a unified schema:
    date (datetime64[ns]), ticker (str), headline (str), source (str)

Sources:
    - FNSPID: local CSV or synthetic fallback
    - NewsAPI: live REST API (requires NEWS_API_KEY)
    - SEC EDGAR: 8-K filing full-text search API
"""

import logging
import random
import time
from pathlib import Path

import pandas as pd
import requests

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
# 2. NewsAPI connector
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
    base_url = "https://newsapi.org/v2/everything"
    from_date = (pd.Timestamp.utcnow() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    for ticker in tickers:
        params = {
            "q": ticker,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
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
                rows.append(
                    {
                        "date": art.get("publishedAt", ""),
                        "ticker": ticker,
                        "headline": art.get("title", ""),
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
# 3. SEC EDGAR connector
# ---------------------------------------------------------------------------

_EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_HEADERS = {"User-Agent": "AlphaLens research@alphalens.ai"}
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
# 4. Combiner
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
# 5. Orchestrator
# ---------------------------------------------------------------------------


def run_data_sources() -> pd.DataFrame:
    """Run all three connectors, combine results, and return the full dataset.

    Executes :func:`load_fnspid`, :func:`fetch_newsapi`, and
    :func:`fetch_sec_edgar` in sequence, then passes their outputs to
    :func:`combine_sources`.

    Returns:
        pd.DataFrame: The combined news DataFrame saved to
        ``config.PROCESSED_DATA_DIR / "combined_news.csv"``.
    """
    logger.info("=== AlphaLens Data Sources Pipeline ===")

    fnspid_df = load_fnspid()
    newsapi_df = fetch_newsapi()
    edgar_df = fetch_sec_edgar()

    combined = combine_sources([fnspid_df, newsapi_df, edgar_df])

    logger.info("=== Data sources pipeline complete ===")
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_data_sources()
