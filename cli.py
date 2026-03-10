"""AlphaLens command-line interface.

Entry point for all user-facing operations:

- ``--ticker``      Single-stock recommendation with a formatted result box.
- ``--compare``     Side-by-side comparison table for multiple tickers.
- ``--portfolio``   Full portfolio scan across all configured tickers.
- ``--watch``       60-minute polling loop with change-alert notifications.
- ``--quick-test``  Rapid end-to-end pipeline validation with a small dataset.

Usage examples::

    python cli.py --ticker AAPL
    python cli.py --compare AAPL MSFT GOOGL
    python cli.py --portfolio
    python cli.py --watch AAPL
    python cli.py --quick-test
"""

import argparse
import logging
import sys
import time
from pathlib import Path

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
# ASCII logo
# ---------------------------------------------------------------------------

_LOGO = r"""
  ___  _                 _
 / _ \| |               | |
/ /_\ \ |   __ __   __ _| |     ___ _ __  ___
|  _  | |   | '_ \ / _` | |    / _ \ '_ \/ __|
| | | | |___| |_) | (_| | |___|  __/ | | \__ \
\_| |_/_____|_.__/ \__,_\_____/\___|_| |_|___/
            | |
            |_|

AI-Powered Stock Sentiment Engine v{version}
""".format(
    version=config.VERSION
)

# Result box width
_BOX_W: int = 45

# Watch interval (seconds)
_WATCH_INTERVAL: int = 3600


# ---------------------------------------------------------------------------
# Unicode / ASCII helpers (Windows CP1252 safe)
# ---------------------------------------------------------------------------


def _enable_unicode() -> bool:
    """Attempt to switch stdout to UTF-8 encoding.

    Returns:
        bool: ``True`` if UTF-8 output is available, ``False`` if the
        terminal only supports ASCII-safe characters.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            return True
        except Exception:
            return False
    try:
        "┌".encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_UNICODE = _enable_unicode()

# Stable ASCII characters (avoids Windows encoding issues in constrained shells)
_TL = "+"
_TR = "+"
_BL = "+"
_BR = "+"
_H = "-"
_V = "|"
_TM = "+"
_BM = "+"
_LM = "+"
_RM = "+"
_CR = "+"
_ARROW_UP = "^"
_ARROW_DN = "v"
_BULLET = "*"
_WARN = "!"

def _hline(left: str, right: str, width: int) -> str:
    """Return a horizontal box border of *width* inner characters.

    Args:
        left: Left corner character.
        right: Right corner character.
        width: Number of fill characters between corners.

    Returns:
        str: The complete border string.
    """
    return left + _H * width + right


def _row(content: str, width: int) -> str:
    """Return one padded content row inside the box.

    Args:
        content: The text to display (must fit within *width* characters).
        width: Total inner width (padding included).

    Returns:
        str: Formatted row with left/right border characters.
    """
    return f"{_V} {content:<{width - 2}} {_V}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model():
    """Load the best-checkpoint FinBERT model from ``config.MODEL_DIR``.

    Returns:
        FinBERTClassifier: The loaded model in eval mode.

    Raises:
        SystemExit: If the checkpoint directory does not contain a
            ``weights.pt`` file, prints a helpful instruction and exits.
    """
    from src.model import FinBERTClassifier

    weights = config.MODEL_DIR / "weights.pt"
    if not weights.exists():
        logger.error(
            "No trained model found at %s. Run:  python main.py  first.",
            config.MODEL_DIR,
        )
        print(
            f"\n  No trained model found in {config.MODEL_DIR}.\n"
            "  Run:  python main.py  to train and save a model first.\n"
        )
        sys.exit(1)

    logger.info("Loading FinBERT model from %s …", config.MODEL_DIR)
    return FinBERTClassifier.load(config.MODEL_DIR)


def _get_model_or_vader():
    """Return the best available model.

    Tries to load the trained FinBERT checkpoint.  If not found, falls
    back to :class:`~src.model.VADERBaseline` and logs a notice so the
    user is not left confused.

    Returns:
        FinBERTClassifier | VADERBaseline: Ready-to-use model instance.
    """
    from src.model import FinBERTClassifier, VADERBaseline

    weights = config.MODEL_DIR / "weights.pt"
    if weights.exists():
        try:
            return FinBERTClassifier.load(config.MODEL_DIR)
        except Exception as exc:
            logger.warning(
                "Found checkpoint marker but loading failed (%s). Falling back to VADERBaseline.",
                exc,
            )
            return VADERBaseline()

    logger.warning(
        "No trained FinBERT model found — falling back to VADERBaseline. "
        "Run  python main.py  to train FinBERT."
    )
    print(
        "\n  [INFO] No trained model found in "
        f"{config.MODEL_DIR}.\n"
        "  Using VADERBaseline instead.  "
        "Run  python main.py  to train FinBERT.\n"
    )
    return VADERBaseline()


# ---------------------------------------------------------------------------
# Result box printer
# ---------------------------------------------------------------------------


def _print_result_box(result) -> None:
    """Print a single-ticker recommendation in institutional output format.

    Displays ticker, recommendation, confidence, signal score, macro regime,
    model quality score, drivers, risk flags, keywords with CAR annotations,
    source breakdown, and reasoning.

    Args:
        result: A :class:`~src.stock_engine.RecommendationResult` instance.
    """
    from src.keyword_analysis import _FINANCIAL_STOPWORDS  # re-use for filter

    # Load keyword weights once for CAR annotation
    kw_path = config.METRICS_DIR / "keyword_summary.csv"
    kw_weights: dict[str, float] = {}
    try:
        import pandas as pd

        if kw_path.exists():
            df = pd.read_csv(kw_path)
            kw_weights = dict(zip(df["keyword"].astype(str), df["avg_CAR"].astype(float)))
    except Exception:
        pass

    inner = _BOX_W - 2  # space between border and outer padding
    rec = result.recommendation
    arrow = (
        f" {_ARROW_UP}" if rec == "BUY" else (f" {_ARROW_DN}" if rec == "SELL" else "  ")
    )

    def _kw_annotation(kw: str) -> str:
        car = kw_weights.get(kw)
        if car is not None:
            return f"{kw}  ({car:+.1%} avg CAR)"
        return kw

    # Source breakdown string
    src_parts = [f"{src}({cnt})" for src, cnt in result.source_breakdown.items()]
    src_lines: list[str] = []
    line_buf = "Sources: "
    for part in src_parts:
        if len(line_buf) + len(part) + 1 > inner - 2:
            src_lines.append(line_buf.rstrip())
            line_buf = "         " + part + " "
        else:
            line_buf += part + " "
    if line_buf.strip():
        src_lines.append(line_buf.rstrip())

    # Reasoning word-wrap
    def _wrap(text: str, width: int) -> list[str]:
        words = text.split()
        lines, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 > width:
                if cur:
                    lines.append(cur.rstrip())
                cur = w + " "
            else:
                cur += w + " "
        if cur.strip():
            lines.append(cur.rstrip())
        return lines or [""]

    reasoning_lines = _wrap(result.reasoning, inner - 2)
    generated = result.generated_at.strftime("%Y-%m-%d %H:%M UTC")

    # Phase-2 extended fields (safe defaults for backward compat)
    macro_regime = getattr(result, "macro_regime", "") or ""
    model_quality = getattr(result, "model_quality_score", 0.0)
    drivers = getattr(result, "drivers", []) or []
    risk_flags = getattr(result, "risk_flags", []) or []
    adj_score = getattr(result, "adjusted_signal_score", result.signal_score)

    rows: list[str] = [
        _hline(_TL, _TR, _BOX_W),
        _row("AlphaLens  Institutional Intelligence", inner),
        _row("", inner),
        _row(f"Ticker:           {result.ticker}", inner),
        _row(f"Recommendation:   {rec}{arrow}", inner),
        _row(f"Confidence:       {result.confidence:.0%}", inner),
        _row(f"Signal Score:     {result.signal_score:+.3f}  (adj: {adj_score:+.3f})", inner),
        _row(
            f"Headlines:        {result.headline_count} analysed "
            f"({result.confident_headline_count} conf.)",
            inner,
        ),
    ]

    # Extended Phase-2 fields
    tactical = getattr(result, "tactical_signal", "")
    inv_view = getattr(result, "investment_view", "")
    conviction = getattr(result, "conviction_level", "")
    if model_quality > 0:
        rows.append(_row(f"Model Quality:    {model_quality:.1f} / 10", inner))
    if macro_regime:
        rows.append(_row(f"Macro Regime:     {macro_regime.replace('_', ' ').title()}", inner))
    if conviction:
        rows.append(_row(f"Conviction:       {conviction}", inner))
    if tactical and inv_view:
        rows.append(_row(f"Tactical:         {tactical}   |   Inv. View: {inv_view}", inner))

    # Drivers section
    if drivers:
        rows.append(_row("", inner))
        rows.append(_row("Drivers:", inner))
        for d in drivers[:4]:
            for dl in _wrap(f"  {_BULLET} {d}", inner - 2):
                rows.append(_row(dl, inner))

    # Risk flags section
    if risk_flags:
        rows.append(_row("", inner))
        rows.append(_row(f"Risk Flags:  {_WARN}", inner))
        for rf in risk_flags[:4]:
            for rfl in _wrap(f"  {_BULLET} {rf}", inner - 2):
                rows.append(_row(rfl, inner))

    # Positive keywords
    if result.top_positive_keywords:
        rows.append(_row("", inner))
        rows.append(_row("Top positive keywords:", inner))
        for kw in result.top_positive_keywords[:3]:
            rows.append(_row(f"  {_BULLET} {_kw_annotation(kw)}", inner))
    # Negative keywords
    if result.top_negative_keywords:
        rows.append(_row("", inner))
        rows.append(_row("Top negative keywords:", inner))
        for kw in result.top_negative_keywords[:3]:
            rows.append(_row(f"  {_BULLET} {_kw_annotation(kw)}", inner))

    # Sources
    rows.append(_row("", inner))
    for sl in src_lines:
        rows.append(_row(sl, inner))

    # Reasoning
    rows.append(_row("", inner))
    rows.append(_row("Reasoning:", inner))
    for rl in reasoning_lines:
        rows.append(_row(f"  {rl}", inner))

    rows.append(_row("", inner))
    rows.append(_row(f"Generated: {generated}", inner))
    rows.append(_hline(_BL, _BR, _BOX_W))

    print()
    for row in rows:
        print(row)
    print()


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------


def _print_comparison_table(results: list) -> None:
    """Print a side-by-side comparison table for multiple tickers.

    Args:
        results: List of :class:`~src.stock_engine.RecommendationResult`
            objects to compare.
    """
    col_w = [8, 8, 9, 9, 10]  # ticker, rec, score, conf, headlines
    headers = ["Ticker", "Signal", "Score", "Confid.", "Headlines"]

    def _border(l: str, m: str, r: str) -> str:
        return l + m.join(_H * w for w in col_w) + r

    def _data_row(cells: list[str]) -> str:
        parts = [f"{c:^{w}}" for c, w in zip(cells, col_w)]
        return _V + _V.join(parts) + _V

    _signal_str = {"BUY": f"BUY {_ARROW_UP}", "SELL": f"SELL {_ARROW_DN}", "HOLD": "HOLD"}

    print()
    print(_border(_TL, _TM, _TR))
    print(_data_row(headers))
    print(_border(_LM, _CR, _RM))
    for r in results:
        print(
            _data_row(
                [
                    r.ticker,
                    _signal_str.get(r.recommendation, r.recommendation),
                    f"{r.signal_score:+.3f}",
                    f"{r.confidence:.0%}",
                    str(r.headline_count),
                ]
            )
        )
    print(_border(_BL, _BM, _BR))
    print()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_ticker(ticker: str) -> None:
    """Handle ``--ticker TICKER``.

    Loads the best available model, generates a recommendation for
    *ticker*, and prints the formatted result box.

    Args:
        ticker: Equity ticker symbol to analyse.
    """
    from src.stock_engine import StockRecommendationEngine

    model = _get_model_or_vader()
    engine = StockRecommendationEngine(model=model)
    result = engine.recommend(ticker.upper())
    _print_result_box(result)


def cmd_compare(tickers: list[str]) -> None:
    """Handle ``--compare TICKER [TICKER ...]``.

    Generates recommendations for all *tickers* and prints a side-by-side
    comparison table.

    Args:
        tickers: List of equity ticker symbols to compare.
    """
    from src.stock_engine import StockRecommendationEngine

    model = _get_model_or_vader()
    engine = StockRecommendationEngine(model=model)
    results = []
    for ticker in tickers:
        logger.info("Analysing %s …", ticker.upper())
        results.append(engine.recommend(ticker.upper()))
    _print_comparison_table(results)


def cmd_portfolio() -> None:
    """Handle ``--portfolio``.

    Runs :class:`~src.portfolio_engine.PortfolioEngine` across all
    tickers in ``config.TICKERS``, prints the watchlist table, and saves
    a timestamped CSV report.
    """
    from src.portfolio_engine import PortfolioEngine
    from src.stock_engine import StockRecommendationEngine

    model = _get_model_or_vader()
    engine = StockRecommendationEngine(model=model)
    portfolio = PortfolioEngine(engine=engine)
    results = portfolio.scan()
    portfolio.print_portfolio_table(results)
    portfolio.save_portfolio_report(results)


def cmd_watch(ticker: str) -> None:
    """Handle ``--watch TICKER``.

    Polls :meth:`~src.stock_engine.StockRecommendationEngine.recommend`
    every 60 minutes and prints each result with its UTC timestamp.
    Emits a highlighted alert line when the recommendation changes from
    the previous cycle.  Press Ctrl+C to exit cleanly.

    Args:
        ticker: Equity ticker symbol to monitor.
    """
    from src.stock_engine import StockRecommendationEngine

    model = _get_model_or_vader()
    engine = StockRecommendationEngine(model=model)

    ticker = ticker.upper()
    previous_rec: str | None = None

    logger.info("Watching %s — polling every %d minutes. Ctrl+C to stop.", ticker, _WATCH_INTERVAL // 60)
    print(f"\n  Watching {ticker} — updating every {_WATCH_INTERVAL // 60} min.  Ctrl+C to stop.\n")

    try:
        while True:
            from datetime import datetime, timezone

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            result = engine.recommend(ticker)
            current_rec = result.recommendation

            print(f"[{ts}]  {ticker}  {current_rec}  score={result.signal_score:+.3f}  conf={result.confidence:.0%}")

            if previous_rec is not None and current_rec != previous_rec:
                alert = f"  {_WARN} RECOMMENDATION CHANGED: {previous_rec} → {current_rec}"
                print(alert)
                logger.warning("Watch alert for %s: %s → %s", ticker, previous_rec, current_rec)

            previous_rec = current_rec
            time.sleep(_WATCH_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n  Watch stopped for {ticker}.\n")
        logger.info("Watch loop terminated by user for %s", ticker)


def cmd_quick_test() -> None:
    """Handle ``--quick-test``.

    Temporarily sets ``config.QUICK_TEST = True`` and runs the full
    data preparation → training → evaluation pipeline with a small
    dataset subset for rapid development iteration.
    """
    import config as _cfg

    _cfg.QUICK_TEST = True
    logger.info("QUICK_TEST mode enabled — running abbreviated pipeline")
    print("\n  QUICK_TEST mode: running pipeline on reduced dataset …\n")

    from src.data_prep import run_data_prep
    from src.model import VADERBaseline
    from src.evaluate import run_evaluation

    run_data_prep()

    logger.info("Skipping FinBERT training in quick-test (using VADERBaseline)")
    vader = VADERBaseline()
    metrics = run_evaluation(finbert_model=vader)

    print("\n  Quick-test complete.")
    print(f"  VADER accuracy:   {metrics['vader']['accuracy']:.3f}")
    print(f"  FinBERT accuracy: {metrics['finbert']['accuracy']:.3f}  (VADERBaseline used as proxy)\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser with all sub-commands.
    """
    parser = argparse.ArgumentParser(
        prog="alphalens",
        description="AlphaLens — AI-Powered Stock Sentiment Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cli.py --ticker AAPL\n"
            "  python cli.py --compare AAPL MSFT GOOGL\n"
            "  python cli.py --portfolio\n"
            "  python cli.py --watch AAPL\n"
            "  python cli.py --quick-test\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ticker",
        metavar="SYMBOL",
        type=str,
        help="Generate a recommendation for a single ticker.",
    )
    group.add_argument(
        "--compare",
        metavar="SYMBOL",
        nargs="+",
        help="Compare recommendations across multiple tickers.",
    )
    group.add_argument(
        "--portfolio",
        action="store_true",
        help="Scan all configured tickers and print the portfolio watchlist.",
    )
    group.add_argument(
        "--watch",
        metavar="SYMBOL",
        type=str,
        help="Poll recommendations for a ticker every 60 minutes.",
    )
    group.add_argument(
        "--quick-test",
        action="store_true",
        dest="quick_test",
        help="Run an abbreviated end-to-end pipeline for development testing.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command handler.

    Displays the ASCII logo on every invocation, then routes to one of
    :func:`cmd_ticker`, :func:`cmd_compare`, :func:`cmd_portfolio`,
    :func:`cmd_watch`, or :func:`cmd_quick_test`.
    """
    print(_LOGO)

    parser = _build_parser()
    args = parser.parse_args()

    if args.ticker:
        cmd_ticker(args.ticker)
    elif args.compare:
        cmd_compare(args.compare)
    elif args.portfolio:
        cmd_portfolio()
    elif args.watch:
        cmd_watch(args.watch)
    elif args.quick_test:
        cmd_quick_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

