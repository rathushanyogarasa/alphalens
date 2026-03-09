"""Stock recommendation engine for AlphaLens.

Provides :class:`StockRecommendationEngine`, which fetches live headlines
from multiple sources, scores them with a sentiment model, applies
source-credibility, keyword-impact, and recency weights, and produces a
structured :class:`RecommendationResult` with plain-English reasoning.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.model import SIGNAL_MAP

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
# Recommendation thresholds
# ---------------------------------------------------------------------------

_BUY_THRESHOLD: float = config.BUY_THRESHOLD      # e.g. +0.3
_SELL_THRESHOLD: float = config.SELL_THRESHOLD     # e.g. -0.3

# Seconds per day — used for recency decay
_SECONDS_PER_DAY: float = 86_400.0

# EDGAR search endpoint
_EDGAR_URL: str = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_HEADERS: dict[str, str] = {"User-Agent": "AlphaLens research@alphalens.ai"}


# ---------------------------------------------------------------------------
# 1. RecommendationResult
# ---------------------------------------------------------------------------


@dataclass
class RecommendationResult:
    """Structured output from :class:`StockRecommendationEngine`.

    Attributes:
        ticker: Equity ticker symbol (e.g. ``"AAPL"``).
        recommendation: One of ``"BUY"``, ``"HOLD"``, or ``"SELL"``.
        confidence: Mean model confidence across confident headlines,
            in the range ``[0, 1]``.
        signal_score: Composite weighted sentiment score in
            approximately ``[-1, 1]``.
        headline_count: Total number of headlines fetched.
        confident_headline_count: Headlines that met
            ``config.CONFIDENCE_THRESHOLD``.
        top_positive_keywords: Up to five high-impact positive keywords
            found in the headlines.
        top_negative_keywords: Up to five high-impact negative keywords
            found in the headlines.
        source_breakdown: Mapping of source name → headline count.
        reasoning: Plain-English explanation of the recommendation.
        generated_at: UTC timestamp when the recommendation was produced.
    """

    ticker: str
    recommendation: str
    confidence: float
    signal_score: float
    headline_count: int
    confident_headline_count: int
    top_positive_keywords: list[str] = field(default_factory=list)
    top_negative_keywords: list[str] = field(default_factory=list)
    source_breakdown: dict[str, int] = field(default_factory=dict)
    reasoning: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"[{self.ticker}] {self.recommendation} "
            f"(score={self.signal_score:+.3f}, conf={self.confidence:.2f}) "
            f"| {self.reasoning}"
        )


# ---------------------------------------------------------------------------
# 2. StockRecommendationEngine
# ---------------------------------------------------------------------------


class StockRecommendationEngine:
    """Produces buy/hold/sell recommendations from live multi-source news.

    Combines source-credibility weights, keyword CAR-based impact weights,
    and exponential recency decay to score each headline.  The composite
    signal is then thresholded to produce a ``BUY``, ``HOLD``, or
    ``SELL`` recommendation with a plain-English rationale.

    Args:
        model: A trained sentiment model exposing
            ``predict(texts: list[str]) -> list[dict]``.  Compatible
            with :class:`~src.model.FinBERTClassifier` and
            :class:`~src.model.VADERBaseline`.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.keyword_weights: dict[str, float] = self._load_keyword_weights()
        logger.info(
            "StockRecommendationEngine initialised | %d keyword weights loaded",
            len(self.keyword_weights),
        )

    # ------------------------------------------------------------------
    # Keyword weight loading
    # ------------------------------------------------------------------

    def _load_keyword_weights(self) -> dict[str, float]:
        """Load keyword CAR weights from ``keyword_summary.csv``.

        Reads ``config.METRICS_DIR / "keyword_summary.csv"`` and builds a
        mapping of ``keyword → avg_CAR``.  If the file does not exist,
        returns an empty dict (uniform weights of 1.0 are applied at
        scoring time).

        Returns:
            dict[str, float]: Mapping of keyword string to its historical
            average cumulative abnormal return.
        """
        path = config.METRICS_DIR / "keyword_summary.csv"
        if not path.exists():
            logger.info(
                "keyword_summary.csv not found at %s — using uniform keyword weights", path
            )
            return {}
        try:
            df = pd.read_csv(path)
            weights = dict(zip(df["keyword"].astype(str), df["avg_CAR"].astype(float)))
            logger.info("Loaded %d keyword CAR weights from %s", len(weights), path)
            return weights
        except Exception as exc:
            logger.warning("Failed to load keyword weights: %s — using uniform", exc)
            return {}

    # ------------------------------------------------------------------
    # Headline fetching
    # ------------------------------------------------------------------

    def _fetch_yfinance_headlines(self, ticker: str) -> list[dict]:
        """Fetch recent headlines from yfinance ticker news.

        Args:
            ticker: Equity ticker symbol.

        Returns:
            list[dict]: Rows with ``headline``, ``date``, ``ticker``,
            ``source`` keys.
        """
        rows: list[dict] = []
        try:
            import yfinance as yf

            info = yf.Ticker(ticker)
            news_items = info.news or []
            for item in news_items:
                title = item.get("title", "")
                pub_ts = item.get("providerPublishTime", 0)
                if not title:
                    continue
                date = pd.Timestamp(pub_ts, unit="s", tz="UTC").tz_localize(None)
                rows.append(
                    {
                        "headline": title,
                        "date": date,
                        "ticker": ticker,
                        "source": "yfinance",
                    }
                )
            logger.info("yfinance [%s]: %d headlines", ticker, len(rows))
        except Exception as exc:
            logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return rows

    def _fetch_newsapi_headlines(self, ticker: str) -> list[dict]:
        """Fetch recent headlines from NewsAPI.

        Silently returns an empty list if ``config.NEWS_API_KEY`` is
        not set.

        Args:
            ticker: Equity ticker symbol used as query term.

        Returns:
            list[dict]: Rows with ``headline``, ``date``, ``ticker``,
            ``source`` keys.
        """
        if not config.NEWS_API_KEY:
            return []
        rows: list[dict] = []
        from_date = (pd.Timestamp.utcnow() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        params = {
            "q": ticker,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50,
            "apiKey": config.NEWS_API_KEY,
        }
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            for art in resp.json().get("articles", []):
                title = art.get("title", "")
                if not title:
                    continue
                rows.append(
                    {
                        "headline": title,
                        "date": pd.Timestamp(art.get("publishedAt", "")).tz_localize(None),
                        "ticker": ticker,
                        "source": "newsapi",
                    }
                )
            logger.info("NewsAPI [%s]: %d headlines", ticker, len(rows))
        except Exception as exc:
            logger.warning("NewsAPI fetch failed for %s: %s", ticker, exc)
        return rows

    def _fetch_edgar_headlines(self, ticker: str) -> list[dict]:
        """Fetch recent 8-K filing headlines from SEC EDGAR.

        Queries the EDGAR full-text search API for items 2.02 (earnings
        releases) and 8.01 (other material events) filed in the last
        two years.

        Args:
            ticker: Equity ticker symbol.

        Returns:
            list[dict]: Rows with ``headline``, ``date``, ``ticker``,
            ``source`` keys.
        """
        rows: list[dict] = []
        date_from = (pd.Timestamp.now() - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
        date_to = pd.Timestamp.now().strftime("%Y-%m-%d")

        for item in ("2.02", "8.01"):
            params = {
                "q": f'"{ticker}"',
                "dateRange": "custom",
                "startdt": date_from,
                "enddt": date_to,
                "forms": "8-K",
                "_source": "file_date,display_names",
            }
            for attempt in range(2):
                try:
                    resp = requests.get(
                        _EDGAR_URL,
                        params=params,
                        headers=_EDGAR_HEADERS,
                        timeout=6,
                    )
                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        logger.warning("EDGAR 429 — sleeping %ds", wait)
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    hits = resp.json().get("hits", {}).get("hits", [])
                    for hit in hits:
                        src = hit.get("_source", {})
                        file_date = src.get("file_date", "")
                        names = src.get("display_names", ticker)
                        headline = f"{ticker} 8-K item {item} filing: {names}"
                        rows.append(
                            {
                                "headline": headline,
                                "date": pd.Timestamp(file_date) if file_date else pd.NaT,
                                "ticker": ticker,
                                "source": "sec_edgar",
                            }
                        )
                    break
                except Exception as exc:
                    logger.warning("EDGAR [%s item %s] attempt %d failed: %s", ticker, item, attempt + 1, exc)
                    time.sleep(2 ** attempt)
            time.sleep(0.4)

        logger.info("EDGAR [%s]: %d headlines", ticker, len(rows))
        return rows

    def _fetch_headlines(self, ticker: str) -> pd.DataFrame:
        """Aggregate headlines for *ticker* from all configured sources.

        Calls yfinance, NewsAPI (if key present), and SEC EDGAR in
        sequence, combines all rows, deduplicates on headline text, and
        drops rows with null dates.

        Args:
            ticker: Equity ticker symbol.

        Returns:
            pd.DataFrame: Combined DataFrame with columns ``headline``,
            ``date``, ``ticker``, ``source``, sorted newest-first.
            Returns an empty DataFrame if no headlines are found.
        """
        combined_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
        if combined_path.exists():
            try:
                cached = pd.read_csv(combined_path, parse_dates=["date"])
                cached = cached[cached["ticker"].astype(str).str.upper() == ticker.upper()].copy()
                cached = cached.dropna(subset=["headline", "date"])
                cached = cached.drop_duplicates(subset=["headline"]).sort_values(
                    "date", ascending=False
                )
                if not cached.empty:
                    logger.info(
                        "[%s] Using %d cached headlines from %s",
                        ticker,
                        len(cached),
                        combined_path,
                    )
                    return cached[["headline", "date", "ticker", "source"]].reset_index(
                        drop=True
                    )
            except Exception as exc:
                logger.warning("Failed loading cached combined_news.csv: %s", exc)

        all_rows: list[dict] = []
        if not config.QUICK_TEST:
            all_rows.extend(self._fetch_yfinance_headlines(ticker))
            all_rows.extend(self._fetch_newsapi_headlines(ticker))
            all_rows.extend(self._fetch_edgar_headlines(ticker))

        if not all_rows:
            logger.warning("No headlines found for %s", ticker)
            return pd.DataFrame(columns=["headline", "date", "ticker", "source"])

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        before = len(df)
        df = df.dropna(subset=["date"])
        df = df.drop_duplicates(subset=["headline"])
        df = df[df["headline"].str.strip() != ""]
        df = df.sort_values("date", ascending=False).reset_index(drop=True)
        logger.info(
            "[%s] Headlines fetched: %d (after dedup/clean: %d) | sources: %s",
            ticker,
            before,
            len(df),
            df["source"].value_counts().to_dict(),
        )
        return df

    # ------------------------------------------------------------------
    # Signal scoring
    # ------------------------------------------------------------------

    def _calculate_signal_score(
        self,
        predictions: list[dict],
        news_df: pd.DataFrame,
    ) -> float:
        """Compute a composite weighted sentiment signal score.

        For each confident prediction (confidence ≥
        ``config.CONFIDENCE_THRESHOLD``) three multiplicative weights
        are applied to a base sentiment value ``{+1, 0, -1}``:

        1. **Source credibility** — from ``config.SOURCE_WEIGHTS``.
        2. **Keyword impact** — looks up each headline against
           ``self.keyword_weights`` (avg CAR from event studies);
           uses 1.0 if no keyword matches.
        3. **Recency decay** — ``exp(−λ × days_old)`` where λ is
           ``config.RECENCY_DECAY_LAMBDA``.

        The final score is the weighted mean of
        ``sentiment × source_w × keyword_w × recency_w``.

        Args:
            predictions: List of dicts from ``model.predict()``.
            news_df: DataFrame aligned row-for-row with *predictions*,
                providing ``source`` and ``date`` columns.

        Returns:
            float: Composite score in approximately ``[-1, 1]``.
                Returns 0.0 if no confident predictions exist.
        """
        now = pd.Timestamp.now()
        weighted_scores: list[float] = []
        weights: list[float] = []

        for i, pred in enumerate(predictions):
            if pred["confidence"] < config.CONFIDENCE_THRESHOLD:
                continue

            sentiment = float(SIGNAL_MAP.get(pred["label_name"], 0))
            if sentiment == 0:
                continue  # neutral contributes no directional signal

            row = news_df.iloc[i]

            # Weight 1: source credibility
            source = str(row.get("source", ""))
            source_w = config.SOURCE_WEIGHTS.get(source, 0.5)

            # Weight 2: keyword impact
            headline = str(row.get("headline", "")).lower()
            keyword_w = 1.0
            best_abs = 0.0
            for kw, avg_car in self.keyword_weights.items():
                if kw.lower() in headline and abs(avg_car) > best_abs:
                    best_abs = abs(avg_car)
                    # Scale: CAR of 0.02 → weight of 1 + 0.02*10 = 1.2 (capped at 2.0)
                    keyword_w = min(2.0, 1.0 + abs(avg_car) * 10)

            # Weight 3: recency decay
            date = row.get("date")
            try:
                days_old = max(0.0, (now - pd.Timestamp(date)).total_seconds() / _SECONDS_PER_DAY)
            except Exception:
                days_old = 30.0
            import math
            recency_w = math.exp(-config.RECENCY_DECAY_LAMBDA * days_old)

            combined_w = source_w * keyword_w * recency_w
            weighted_scores.append(sentiment * combined_w * pred["confidence"])
            weights.append(combined_w)

        if not weights or sum(weights) == 0:
            return 0.0

        score = sum(weighted_scores) / sum(weights)
        return round(float(score), 6)

    # ------------------------------------------------------------------
    # Reasoning generation
    # ------------------------------------------------------------------

    def _generate_reasoning(
        self,
        predictions: list[dict],
        news_df: pd.DataFrame,
        signal_score: float,
    ) -> str:
        """Build a plain-English explanation of the recommendation.

        Summarises:

        - The proportion and sentiment direction of confident headlines.
        - The single most credible source with a positive/negative signal.
        - The most impactful keyword found (with its historical avg CAR
          if available).
        - A note on contradicting signals if both positive and negative
          headlines exist.

        Args:
            predictions: List of dicts from ``model.predict()``.
            news_df: DataFrame aligned row-for-row with *predictions*.
            signal_score: Computed composite signal score.

        Returns:
            str: Multi-clause plain-English reasoning string.
        """
        confident = [
            (p, news_df.iloc[i])
            for i, p in enumerate(predictions)
            if p["confidence"] >= config.CONFIDENCE_THRESHOLD
        ]

        total_confident = len(confident)
        if total_confident == 0:
            return (
                f"No headlines met the confidence threshold of "
                f"{config.CONFIDENCE_THRESHOLD:.0%}. Defaulting to HOLD."
            )

        n_pos = sum(1 for p, _ in confident if p["label_name"] == "positive")
        n_neg = sum(1 for p, _ in confident if p["label_name"] == "negative")
        n_neu = total_confident - n_pos - n_neg

        parts: list[str] = []

        # Clause 1: overall headline sentiment
        direction = "positive" if n_pos > n_neg else ("negative" if n_neg > n_pos else "mixed")
        parts.append(
            f"{n_pos} of {total_confident} confident headlines classified positive, "
            f"{n_neg} negative, {n_neu} neutral (overall sentiment: {direction})."
        )

        # Clause 2: most credible source
        best_src, best_w = "", 0.0
        for _, row in confident:
            src = str(row.get("source", ""))
            w = config.SOURCE_WEIGHTS.get(src, 0.5)
            if w > best_w:
                best_w, best_src = w, src
        if best_src:
            parts.append(f"Highest-credibility source: {best_src} (weight={best_w:.1f}).")

        # Clause 3: most impactful keyword
        best_kw, best_car = "", 0.0
        for _, row in confident:
            headline = str(row.get("headline", "")).lower()
            for kw, avg_car in self.keyword_weights.items():
                if kw.lower() in headline and abs(avg_car) > abs(best_car):
                    best_kw, best_car = kw, avg_car
        if best_kw:
            direction_kw = "positive" if best_car > 0 else "negative"
            parts.append(
                f"Keyword '{best_kw}' detected "
                f"(historically {direction_kw} avg CAR={best_car:+.2%})."
            )

        # Clause 4: contradicting signals note
        if n_pos > 0 and n_neg > 0:
            parts.append(
                f"Signal partially dampened by {n_neg} negative headline(s) "
                f"countering {n_pos} positive one(s)."
            )

        # Clause 5: composite score
        parts.append(f"Composite signal score: {signal_score:+.3f}.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Main recommendation method
    # ------------------------------------------------------------------

    def recommend(self, ticker: str) -> RecommendationResult:
        """Generate a buy/hold/sell recommendation for *ticker*.

        Orchestrates headline fetching, sentiment prediction, signal
        scoring, keyword extraction, and reasoning generation.

        Args:
            ticker: Equity ticker symbol (e.g. ``"AAPL"``).

        Returns:
            RecommendationResult: Fully populated recommendation dataclass.
        """
        logger.info("Generating recommendation for %s …", ticker)

        # Fetch headlines
        news_df = self._fetch_headlines(ticker)
        headline_count = len(news_df)

        if news_df.empty:
            logger.warning("[%s] No headlines — returning HOLD by default", ticker)
            return RecommendationResult(
                ticker=ticker,
                recommendation="HOLD",
                confidence=0.0,
                signal_score=0.0,
                headline_count=0,
                confident_headline_count=0,
                reasoning="No headlines available. Defaulting to HOLD.",
            )

        # Predict sentiment
        texts = news_df["headline"].tolist()
        predictions: list[dict] = []
        for start in range(0, len(texts), config.BATCH_SIZE):
            batch = texts[start : start + config.BATCH_SIZE]
            predictions.extend(self.model.predict(batch))

        # Filter confident predictions
        confident_preds = [p for p in predictions if p["confidence"] >= config.CONFIDENCE_THRESHOLD]
        confident_headline_count = len(confident_preds)

        mean_confidence = (
            float(sum(p["confidence"] for p in confident_preds) / confident_headline_count)
            if confident_headline_count > 0
            else 0.0
        )

        # Signal score
        signal_score = self._calculate_signal_score(predictions, news_df)

        # Recommendation
        if signal_score > _BUY_THRESHOLD:
            recommendation = "BUY"
        elif signal_score < _SELL_THRESHOLD:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        # Keyword extraction: positive and negative
        top_positive_keywords = self._extract_sentiment_keywords(
            predictions, news_df, target_label="positive", top_n=5
        )
        top_negative_keywords = self._extract_sentiment_keywords(
            predictions, news_df, target_label="negative", top_n=5
        )

        # Source breakdown
        source_breakdown = news_df["source"].value_counts().to_dict()

        # Reasoning
        reasoning = self._generate_reasoning(predictions, news_df, signal_score)

        result = RecommendationResult(
            ticker=ticker,
            recommendation=recommendation,
            confidence=round(mean_confidence, 4),
            signal_score=round(signal_score, 4),
            headline_count=headline_count,
            confident_headline_count=confident_headline_count,
            top_positive_keywords=top_positive_keywords,
            top_negative_keywords=top_negative_keywords,
            source_breakdown=source_breakdown,
            reasoning=reasoning,
            generated_at=datetime.now(timezone.utc),
        )

        logger.info(
            "[%s] → %s | score=%+.3f | conf=%.2f | "
            "headlines=%d (confident=%d)",
            ticker,
            recommendation,
            signal_score,
            mean_confidence,
            headline_count,
            confident_headline_count,
        )
        return result

    # ------------------------------------------------------------------
    # Internal keyword helper
    # ------------------------------------------------------------------

    def _extract_sentiment_keywords(
        self,
        predictions: list[dict],
        news_df: pd.DataFrame,
        target_label: str,
        top_n: int = 5,
    ) -> list[str]:
        """Extract the most frequent known keywords from sentiment-labelled headlines.

        Filters headlines predicted as *target_label*, tokenises them,
        and returns the *top_n* tokens that appear in
        ``self.keyword_weights``.  Falls back to the most common
        unigrams if the keyword weight dict is empty.

        Args:
            predictions: Full list of prediction dicts.
            news_df: DataFrame aligned with *predictions*.
            target_label: ``"positive"`` or ``"negative"``.
            top_n: Maximum number of keywords to return.

        Returns:
            list[str]: Up to *top_n* keyword strings.
        """
        matching_headlines: list[str] = [
            str(news_df.iloc[i].get("headline", "")).lower()
            for i, p in enumerate(predictions)
            if p.get("label_name") == target_label
            and p["confidence"] >= config.CONFIDENCE_THRESHOLD
        ]

        if not matching_headlines:
            return []

        if self.keyword_weights:
            freq: dict[str, int] = {}
            for headline in matching_headlines:
                for kw in self.keyword_weights:
                    if kw.lower() in headline:
                        freq[kw] = freq.get(kw, 0) + 1
            return sorted(freq, key=freq.get, reverse=True)[:top_n]  # type: ignore[arg-type]

        # Fallback: simple unigram frequency
        from collections import Counter

        tokens: list[str] = []
        for h in matching_headlines:
            tokens.extend(h.split())
        common = Counter(tokens).most_common(top_n * 3)
        stopwords = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "was"}
        return [w for w, _ in common if w not in stopwords][:top_n]
