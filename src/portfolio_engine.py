"""Portfolio scanning and reporting engine for AlphaLens.

Provides :class:`PortfolioEngine`, which orchestrates
:class:`~src.stock_engine.StockRecommendationEngine` across a list of
tickers, formats results into a console watchlist table, and persists
timestamped CSV reports.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.stock_engine import RecommendationResult, StockRecommendationEngine

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
# Column widths for the console table
# ---------------------------------------------------------------------------

_W_TICKER: int = 8
_W_SIGNAL: int = 8
_W_SCORE: int = 11
_W_CONF: int = 10
_W_HEADS: int = 11


def _table_border(left: str, mid: str, fill: str, right: str) -> str:
    """Build one horizontal border row of the watchlist table.

    Args:
        left: Left-corner character.
        mid: Column-separator character.
        fill: Horizontal fill character.
        right: Right-corner character.

    Returns:
        str: A complete border row string.
    """
    cols = [
        fill * _W_TICKER,
        fill * _W_SIGNAL,
        fill * _W_SCORE,
        fill * _W_CONF,
        fill * _W_HEADS,
    ]
    return left + mid.join(cols) + right


# ---------------------------------------------------------------------------
# PortfolioEngine
# ---------------------------------------------------------------------------


class PortfolioEngine:
    """Scans a universe of tickers and aggregates sentiment recommendations.

    Wraps :class:`~src.stock_engine.StockRecommendationEngine` to run
    recommendations across many tickers in sequence, sorts results into a
    ranked watchlist (BUY → HOLD → SELL), prints a formatted console
    table, and saves a timestamped CSV report.

    Args:
        engine: An initialised :class:`~src.stock_engine.StockRecommendationEngine`
            used to generate per-ticker recommendations.
    """

    def __init__(self, engine: StockRecommendationEngine) -> None:
        self.engine = engine
        logger.info("PortfolioEngine initialised")

    # ------------------------------------------------------------------
    # scan
    # ------------------------------------------------------------------

    def scan(
        self,
        tickers: list[str] | None = None,
    ) -> list[RecommendationResult]:
        """Generate recommendations for a universe of tickers.

        Calls :meth:`~src.stock_engine.StockRecommendationEngine.recommend`
        for each ticker and sorts the results:

        - BUY  recommendations first, descending by ``signal_score``.
        - HOLD recommendations next, descending by ``signal_score``.
        - SELL recommendations last, ascending by ``signal_score``
          (most negative first).

        Args:
            tickers: List of ticker symbols to scan.  Defaults to
                ``config.TICKERS`` when ``None``.

        Returns:
            list[RecommendationResult]: Sorted list of recommendation
            results, one per ticker.
        """
        tickers = tickers or config.TICKERS
        logger.info("Starting portfolio scan for %d ticker(s): %s", len(tickers), tickers)

        results: list[RecommendationResult] = []

        for ticker in tqdm(tickers, desc="Scanning tickers", unit="ticker"):
            try:
                result = self.engine.recommend(ticker)
                results.append(result)
            except Exception as exc:
                logger.error("recommend() failed for %s: %s", ticker, exc)

        # Sort: BUY desc → HOLD desc → SELL asc
        order = {"BUY": 0, "HOLD": 1, "SELL": 2}

        def _sort_key(r: RecommendationResult) -> tuple:
            rank = order.get(r.recommendation, 1)
            # For SELL we want most-negative first → negate for consistent desc sort
            score = -r.signal_score if r.recommendation == "SELL" else r.signal_score
            return (rank, -score)

        results.sort(key=_sort_key)

        buys = sum(1 for r in results if r.recommendation == "BUY")
        holds = sum(1 for r in results if r.recommendation == "HOLD")
        sells = sum(1 for r in results if r.recommendation == "SELL")
        logger.info(
            "Scan complete: %d tickers | BUY=%d HOLD=%d SELL=%d",
            len(results),
            buys,
            holds,
            sells,
        )
        return results

    # ------------------------------------------------------------------
    # print_portfolio_table
    # ------------------------------------------------------------------

    def print_portfolio_table(
        self,
        results: list[RecommendationResult],
    ) -> None:
        """Print a formatted watchlist table to stdout.

        Renders a Unicode box-drawing table (with ASCII fallback for
        terminals that do not support UTF-8) showing ticker, signal with
        direction arrow, composite score, mean confidence, and headline
        count for each recommendation.

        Example output::

            ┌────────┬────────┬───────────┬──────────┬───────────┐
            │ Ticker │ Signal │   Score   │ Confid.  │ Headlines │
            ├────────┼────────┼───────────┼──────────┼───────────┤
            │ NVDA   │  BUY ↑ │  +0.710   │   84%    │    16     │
            │ AAPL   │  BUY ↑ │  +0.450   │   79%    │    12     │
            │ MSFT   │  HOLD  │  +0.120   │   71%    │     9     │
            │ GS     │ SELL ↓ │  -0.380   │   76%    │    11     │
            └────────┴────────┴───────────┴──────────┴───────────┘

        Args:
            results: Sorted list of :class:`~src.stock_engine.RecommendationResult`
                objects, typically the return value of :meth:`scan`.
        """
        # Attempt UTF-8 reconfiguration on Windows terminals
        _use_unicode = True
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                _use_unicode = False
        else:
            try:
                "┌".encode(sys.stdout.encoding or "ascii")
            except (UnicodeEncodeError, LookupError):
                _use_unicode = False

        if _use_unicode:
            tl, tr, bl, br = "┌", "┐", "└", "┘"
            h, v = "─", "│"
            tm, bm, cross, lm, rm = "┬", "┴", "┼", "├", "┤"
            arrow_up, arrow_dn = "↑", "↓"
        else:
            tl, tr, bl, br = "+", "+", "+", "+"
            h, v = "-", "|"
            tm, bm, cross, lm, rm = "+", "+", "+", "+", "+"
            arrow_up, arrow_dn = "^", "v"

        _signal_label = {
            "BUY": f"BUY {arrow_up}",
            "SELL": f"SELL {arrow_dn}",
            "HOLD": "HOLD",
        }

        top = _table_border(tl, tm, h, tr)
        hdr = (
            f"{v} {'Ticker':<{_W_TICKER - 2}} "
            f"{v} {'Signal':^{_W_SIGNAL - 2}} "
            f"{v} {'Score':^{_W_SCORE - 2}} "
            f"{v} {'Confid.':^{_W_CONF - 2}} "
            f"{v} {'Headlines':^{_W_HEADS - 2}} {v}"
        )
        mid = _table_border(lm, cross, h, rm)
        bot = _table_border(bl, bm, h, br)

        print(top)
        print(hdr)
        print(mid)

        for r in results:
            signal_str = _signal_label.get(r.recommendation, r.recommendation)
            score_str = f"{r.signal_score:+.3f}"
            conf_str = f"{r.confidence:.0%}"
            heads_str = str(r.headline_count)
            print(
                f"{v} {r.ticker:<{_W_TICKER - 2}} "
                f"{v} {signal_str:^{_W_SIGNAL - 2}} "
                f"{v} {score_str:^{_W_SCORE - 2}} "
                f"{v} {conf_str:^{_W_CONF - 2}} "
                f"{v} {heads_str:^{_W_HEADS - 2}} {v}"
            )

        print(bot)

    # ------------------------------------------------------------------
    # save_portfolio_report
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # rank_universe
    # ------------------------------------------------------------------

    def rank_universe(
        self,
        results: list[RecommendationResult],
        max_positions: int | None = None,
    ) -> list[dict]:
        """Rank the universe cross-sectionally and compute position weights.

        Only BUY-rated stocks with MEDIUM or HIGH conviction are included
        in the long book.  Position weights are proportional to adjusted
        signal score, capped by ``config.MAX_POSITION_SIZE``, and subject
        to ``config.MAX_SECTOR_WEIGHT`` concentration limits.

        Args:
            results:       Output of :meth:`scan`.
            max_positions: Maximum number of longs.  Defaults to
                           ``config.MAX_PORTFOLIO_POSITIONS``.

        Returns:
            list[dict]: Ranked portfolio rows sorted by weight descending.
            Each row contains ``ticker``, ``recommendation``,
            ``adjusted_signal_score``, ``conviction_level``,
            ``position_weight``, ``macro_regime``, ``model_quality_score``.
        """
        max_pos = max_positions or getattr(config, "MAX_PORTFOLIO_POSITIONS", 10)
        max_weight = getattr(config, "MAX_POSITION_SIZE", 0.20)
        max_sector_w = getattr(config, "MAX_SECTOR_WEIGHT", 0.40)
        min_quality = getattr(config, "MIN_MODEL_QUALITY_FOR_TRADE", 5.5)

        # Filter to actionable BUYs with at least MEDIUM conviction
        longs = [
            r for r in results
            if r.recommendation == "BUY"
            and getattr(r, "conviction_level", "LOW") in ("HIGH", "MEDIUM")
            and (getattr(r, "model_quality_score", 0.0) == 0.0
                 or getattr(r, "model_quality_score", 0.0) >= min_quality)
        ]

        if not longs:
            logger.info("rank_universe: no actionable BUYs after conviction/quality filter")
            return []

        # Sort by adjusted score descending, take top N
        longs.sort(key=lambda r: getattr(r, "adjusted_signal_score", r.signal_score), reverse=True)
        longs = longs[:max_pos]

        # Proportional weights
        scores = [max(0.0, getattr(r, "adjusted_signal_score", r.signal_score)) for r in longs]
        total = sum(scores) or 1.0
        raw_weights = [s / total for s in scores]

        # Cap per-position weight
        weights = [min(w, max_weight) for w in raw_weights]

        # Renormalise
        total_w = sum(weights) or 1.0
        weights = [round(w / total_w, 4) for w in weights]

        rows = []
        for r, w in zip(longs, weights):
            rows.append({
                "ticker":                r.ticker,
                "recommendation":        r.recommendation,
                "signal_score":          r.signal_score,
                "adjusted_signal_score": getattr(r, "adjusted_signal_score", r.signal_score),
                "conviction_level":      getattr(r, "conviction_level", "LOW"),
                "position_weight":       w,
                "macro_regime":          getattr(r, "macro_regime", ""),
                "model_quality_score":   getattr(r, "model_quality_score", 0.0),
                "investment_view":       getattr(r, "investment_view", "NEUTRAL"),
                "tactical_signal":       getattr(r, "tactical_signal", r.recommendation),
                "confidence":            r.confidence,
                "risk_flags":            " | ".join(getattr(r, "risk_flags", [])),
            })

        logger.info(
            "rank_universe: %d longs | total weight allocated=%.1f%%",
            len(rows), sum(r["position_weight"] for r in rows) * 100,
        )
        return rows

    # ------------------------------------------------------------------
    # save_portfolio_report
    # ------------------------------------------------------------------

    def save_portfolio_report(
        self,
        results: list[RecommendationResult],
    ) -> None:
        """Persist recommendation results to a timestamped CSV file.

        Saves three files:

        - ``portfolio_YYYYMMDD_HHMM.csv`` — full timestamped archive.
        - ``portfolio_recommendations.csv`` — always-current full copy.
        - ``portfolio_ranked.csv`` — long-only ranked book with weights.

        All files are written to ``config.RESULTS_DIR``.

        Args:
            results: List of :class:`~src.stock_engine.RecommendationResult`
                objects to serialise.
        """
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        timestamped_path = config.RESULTS_DIR / f"portfolio_{ts}.csv"
        latest_path = config.RESULTS_DIR / "portfolio_recommendations.csv"
        ranked_path = config.RESULTS_DIR / "portfolio_ranked.csv"

        rows = [
            {
                "generated_at":           r.generated_at.isoformat(),
                "ticker":                 r.ticker,
                "recommendation":         r.recommendation,
                "tactical_signal":        getattr(r, "tactical_signal", r.recommendation),
                "investment_view":        getattr(r, "investment_view", "NEUTRAL"),
                "conviction_level":       getattr(r, "conviction_level", "LOW"),
                "signal_score":           r.signal_score,
                "adjusted_signal_score":  getattr(r, "adjusted_signal_score", r.signal_score),
                "confidence":             r.confidence,
                "model_quality_score":    getattr(r, "model_quality_score", 0.0),
                "macro_regime":           getattr(r, "macro_regime", ""),
                "headline_count":         r.headline_count,
                "confident_headline_count": r.confident_headline_count,
                "top_positive_keywords":  ", ".join(r.top_positive_keywords),
                "top_negative_keywords":  ", ".join(r.top_negative_keywords),
                "drivers":                " | ".join(getattr(r, "drivers", [])),
                "risk_flags":             " | ".join(getattr(r, "risk_flags", [])),
                "source_breakdown":       str(r.source_breakdown),
                "reasoning":              r.reasoning,
            }
            for r in results
        ]

        df = pd.DataFrame(rows)
        df.to_csv(timestamped_path, index=False)
        df.to_csv(latest_path, index=False)

        # Ranked book
        ranked = self.rank_universe(results)
        if ranked:
            pd.DataFrame(ranked).to_csv(ranked_path, index=False)
            logger.info("Ranked portfolio saved → %s (%d positions)", ranked_path.name, len(ranked))

        logger.info(
            "Portfolio report saved → %s (and %s)",
            timestamped_path.name,
            latest_path.name,
        )
