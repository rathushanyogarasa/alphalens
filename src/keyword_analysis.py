"""Keyword-driven event study and source comparison for AlphaLens.

Extracts financially meaningful keywords from news headlines using
TF-IDF and KeyBERT, runs event-study analysis to compute cumulative
abnormal returns (CARs) around news events, compares signal quality
across data sources, and produces a suite of diagnostic plots.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.plot_style import GOLD, NAVY, FIG_DPI, apply_plot_style

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
# Custom financial stop-words
# ---------------------------------------------------------------------------

_FINANCIAL_STOPWORDS: list[str] = [
    "said",
    "company",
    "year",
    "quarter",
    "per",
    "share",
    "inc",
    "corp",
    "ltd",
    "plc",
    "reported",
    "according",
]


# ---------------------------------------------------------------------------
# 1. extract_tfidf_keywords
# ---------------------------------------------------------------------------


def extract_tfidf_keywords(
    texts: list[str],
    n: int = config.TOP_N_KEYWORDS,
) -> list[str]:
    """Extract the top *n* keywords from a corpus using TF-IDF scoring.

    Builds a :class:`~sklearn.feature_extraction.text.TfidfVectorizer`
    with English stop-words extended by a custom list of domain-specific
    financial terms that carry little discriminative value.  Returns the
    *n* tokens with the highest mean TF-IDF score across the corpus.

    Args:
        texts: List of raw text strings (news headlines or sentences).
        n: Number of top keywords to return.  Defaults to
            ``config.TOP_N_KEYWORDS``.

    Returns:
        list[str]: Ordered list of top *n* keywords, highest-scoring first.
    """
    logger.info("Extracting TF-IDF keywords from %d texts (top %d) …", len(texts), n)

    stop_words: list[str] = list(
        TfidfVectorizer(stop_words="english").get_stop_words()
    ) + _FINANCIAL_STOPWORDS

    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError as exc:
        logger.warning("TF-IDF vectoriser failed (%s) — returning empty list", exc)
        return []

    feature_names = vectorizer.get_feature_names_out()
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[::-1][:n]
    keywords = [feature_names[i] for i in top_indices]

    logger.info("TF-IDF extracted %d keywords; top 5: %s", len(keywords), keywords[:5])
    return keywords


# ---------------------------------------------------------------------------
# 2. extract_keybert_keywords
# ---------------------------------------------------------------------------


def extract_keybert_keywords(
    texts: list[str],
    n_per_doc: int = 5,
) -> dict[str, int]:
    """Extract keywords using KeyBERT and aggregate corpus-level frequencies.

    Runs KeyBERT on each text, collects up to *n_per_doc* keyword
    candidates per document, and returns a dict mapping each unique
    keyword to the number of documents it appeared in.

    Args:
        texts: List of raw text strings.
        n_per_doc: Maximum number of keywords to extract per document.
            Defaults to 5.

    Returns:
        dict[str, int]: Mapping of keyword → document frequency, sorted
        descending by frequency.
    """
    from keybert import KeyBERT  # lazy import — slow to load

    logger.info("Extracting KeyBERT keywords from %d texts …", len(texts))
    kw_model = KeyBERT()
    freq: dict[str, int] = {}

    for i, text in enumerate(texts):
        if not text.strip():
            continue
        try:
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=n_per_doc,
            )
            for kw, _ in keywords:
                kw_lower = kw.lower().strip()
                freq[kw_lower] = freq.get(kw_lower, 0) + 1
        except Exception as exc:
            logger.debug("KeyBERT failed on text %d: %s", i, exc)

        if (i + 1) % 100 == 0:
            logger.info("  KeyBERT: %d / %d texts processed", i + 1, len(texts))

    sorted_freq = dict(sorted(freq.items(), key=lambda kv: kv[1], reverse=True))
    logger.info(
        "KeyBERT extracted %d unique keywords; top 5: %s",
        len(sorted_freq),
        list(sorted_freq.items())[:5],
    )
    return sorted_freq


# ---------------------------------------------------------------------------
# Internal helper: fetch SPY for market-model estimation
# ---------------------------------------------------------------------------


def _get_spy_returns(start: str, end: str) -> pd.Series:
    """Download SPY daily returns for a date range.

    Args:
        start: Start date string ``"YYYY-MM-DD"``.
        end: End date string ``"YYYY-MM-DD"``.

    Returns:
        pd.Series: Daily percentage returns indexed by date, or an empty
        Series on download failure.
    """
    try:
        import yfinance as yf

        spy = yf.download(
            config.MARKET_BENCHMARK,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if spy.empty:
            return pd.Series(dtype=float)
        close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        returns = close.pct_change().dropna()
        returns.index = pd.to_datetime(returns.index).tz_localize(None)
        return returns
    except Exception as exc:
        logger.warning("SPY download failed: %s", exc)
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# 3. run_event_study
# ---------------------------------------------------------------------------


def run_event_study(
    keyword: str,
    news_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict:
    """Run a market-model event study for a single keyword.

    Identifies all headlines that contain *keyword*, extracts the
    ``EVENT_WINDOW`` price window around each event date, fits an OLS
    market model on a 60-trading-day pre-event window to estimate
    expected returns, and computes the cumulative abnormal return (CAR)
    for each event.

    Args:
        keyword: The keyword to search for (case-insensitive substring
            match against the ``headline`` column).
        news_df: Combined news DataFrame with ``date``, ``ticker``, and
            ``headline`` columns.
        prices: Wide adjusted-close price DataFrame from
            :func:`~src.backtest.fetch_price_data`.

    Returns:
        dict: Event study results with keys:

            - ``keyword`` (str)
            - ``event_count`` (int): Number of qualifying events.
            - ``avg_CAR`` (float): Mean CAR across all events.
            - ``std_CAR`` (float): Standard deviation of CARs.
            - ``positive_pct`` (float): Fraction of events with CAR > 0.
            - ``all_CARs`` (list[float]): Raw CAR values.
            - ``daily_avg_ARs`` (dict): Mean abnormal return per event-window day.

        Returns an empty dict ``{}`` if fewer than
        ``config.MIN_EVENTS_PER_KEYWORD`` events are found.
    """
    pre, post = config.EVENT_WINDOW  # e.g. (-1, 3)
    window_days = list(range(pre, post + 1))

    # Find all events containing keyword (case-insensitive)
    mask = news_df["headline"].str.contains(keyword, case=False, na=False, regex=False)
    events = news_df[mask].copy()
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()

    n_events = len(events)
    if n_events < config.MIN_EVENTS_PER_KEYWORD:
        logger.debug(
            "Keyword '%s': only %d events (min=%d) — skipping",
            keyword,
            n_events,
            config.MIN_EVENTS_PER_KEYWORD,
        )
        return {}

    logger.info("Keyword '%s': %d events found", keyword, n_events)

    all_CARs: list[float] = []
    daily_ARs: dict[int, list[float]] = {d: [] for d in window_days}

    prices_sorted = prices.sort_index()
    price_dates = prices_sorted.index
    if config.MARKET_BENCHMARK in prices_sorted.columns:
        benchmark_returns = prices_sorted[config.MARKET_BENCHMARK].pct_change().dropna()
    else:
        benchmark_returns = prices_sorted.mean(axis=1).pct_change().dropna()

    for _, event_row in events.iterrows():
        event_date = pd.Timestamp(event_row["date"])
        ticker = str(event_row["ticker"])

        if ticker not in prices_sorted.columns:
            continue

        # Find position of event date (or nearest available)
        date_pos_arr = price_dates.searchsorted(event_date)
        if date_pos_arr >= len(price_dates):
            continue
        event_idx = int(date_pos_arr)

        # Estimation window: 60 trading days before pre-event
        est_start = max(0, event_idx + pre - 60)
        est_end = max(0, event_idx + pre)
        if est_end - est_start < 10:  # need minimum history
            continue

        # Stock returns in estimation window
        est_prices = prices_sorted[ticker].iloc[est_start:est_end]
        stock_est_ret = est_prices.pct_change().dropna()

        # Benchmark returns in estimation window
        aligned = stock_est_ret.align(benchmark_returns, join="inner")
        stock_aligned, spy_aligned = aligned[0], aligned[1]

        if len(stock_aligned) < 5:
            continue

        # OLS market model: stock_return ~ SPY_return
        X = spy_aligned.values.reshape(-1, 1)
        y = stock_aligned.values
        try:
            reg = LinearRegression().fit(X, y)
            alpha_est = float(reg.intercept_)
            beta_est = float(reg.coef_[0])
        except Exception:
            alpha_est, beta_est = 0.0, 1.0

        # Event window abnormal returns
        CAR = 0.0
        valid_event = True
        day_ARs: dict[int, float] = {}

        for d in window_days:
            win_idx = event_idx + d
            if win_idx <= 0 or win_idx >= len(price_dates):
                valid_event = False
                break

            p0 = prices_sorted[ticker].iloc[win_idx - 1]
            p1 = prices_sorted[ticker].iloc[win_idx]
            if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                valid_event = False
                break

            actual_ret = (p1 - p0) / p0

            # Benchmark return on same day for expected return
            win_date = price_dates[win_idx]
            spy_ret = float(benchmark_returns.get(win_date, 0.0))
            expected_ret = alpha_est + beta_est * spy_ret
            AR = float(actual_ret) - expected_ret
            CAR += AR
            day_ARs[d] = AR

        if valid_event:
            all_CARs.append(CAR)
            for d, ar in day_ARs.items():
                daily_ARs[d].append(ar)

    if len(all_CARs) < config.MIN_EVENTS_PER_KEYWORD:
        logger.debug(
            "Keyword '%s': only %d valid CARs after filtering — skipping",
            keyword,
            len(all_CARs),
        )
        return {}

    avg_CAR = float(np.mean(all_CARs))
    std_CAR = float(np.std(all_CARs))
    positive_pct = float(np.mean([c > 0 for c in all_CARs]))
    daily_avg = {d: float(np.mean(ars)) if ars else 0.0 for d, ars in daily_ARs.items()}

    result = {
        "keyword": keyword,
        "event_count": len(all_CARs),
        "avg_CAR": round(avg_CAR, 6),
        "std_CAR": round(std_CAR, 6),
        "positive_pct": round(positive_pct, 4),
        "all_CARs": all_CARs,
        "daily_avg_ARs": daily_avg,
    }
    logger.info(
        "  '%s': events=%d avg_CAR=%.4f std_CAR=%.4f positive_pct=%.1f%%",
        keyword,
        len(all_CARs),
        avg_CAR,
        std_CAR,
        positive_pct * 100,
    )
    return result


# ---------------------------------------------------------------------------
# 4. run_source_comparison
# ---------------------------------------------------------------------------


def run_source_comparison(
    news_df: pd.DataFrame,
    prices: pd.DataFrame,
    top_keywords: list[str],
) -> dict:
    """Compare event-study CARs across data sources for the top keywords.

    Runs :func:`run_event_study` for each source separately and reports
    the mean CAR per source.  Performs a two-sample Welch's t-test
    comparing SEC EDGAR CARs against all other sources combined.

    Args:
        news_df: Combined news DataFrame with a ``source`` column.
        prices: Wide price DataFrame from
            :func:`~src.backtest.fetch_price_data`.
        top_keywords: List of keywords (from TF-IDF or KeyBERT) to
            include in the comparison.

    Returns:
        dict: Mapping ``source_name → avg_CAR`` for each source, plus
        a ``"t_test"`` key containing ``{"t_stat", "p_value",
        "significant"}`` for the EDGAR vs others comparison.
    """
    sources = news_df["source"].unique().tolist()
    logger.info("Running source comparison across %d sources …", len(sources))

    source_cars: dict[str, list[float]] = {src: [] for src in sources}

    for src in sources:
        src_df = news_df[news_df["source"] == src]
        if src_df.empty:
            continue
        for kw in top_keywords:
            result = run_event_study(kw, src_df, prices)
            if result:
                source_cars[src].extend(result["all_CARs"])
        avg = float(np.mean(source_cars[src])) if source_cars[src] else 0.0
        logger.info("  [%s] %d events | avg_CAR=%.4f", src, len(source_cars[src]), avg)

    source_avg: dict[str, float] = {
        src: round(float(np.mean(cars)) if cars else 0.0, 6)
        for src, cars in source_cars.items()
    }

    # t-test: SEC EDGAR vs rest
    edgar_cars = source_cars.get("sec_edgar", [])
    other_cars: list[float] = []
    for src, cars in source_cars.items():
        if src != "sec_edgar":
            other_cars.extend(cars)

    t_result: dict = {"t_stat": 0.0, "p_value": 1.0, "significant": False}
    if len(edgar_cars) >= 2 and len(other_cars) >= 2:
        t_stat, p_value = stats.ttest_ind(edgar_cars, other_cars, equal_var=False)
        significant = bool(p_value < 0.05)
        t_result = {
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_value), 4),
            "significant": significant,
        }
        verdict = "SIGNIFICANT" if significant else "not significant"
        logger.info(
            "t-test SEC EDGAR vs others: t=%.4f p=%.4f → %s",
            t_stat,
            p_value,
            verdict,
        )
    else:
        logger.warning(
            "Insufficient data for t-test (edgar=%d other=%d)",
            len(edgar_cars),
            len(other_cars),
        )

    source_avg["t_test"] = t_result  # type: ignore[assignment]
    return source_avg


# ---------------------------------------------------------------------------
# 5. plot_keyword_bar_chart
# ---------------------------------------------------------------------------


def plot_keyword_bar_chart(keyword_results: list[dict]) -> None:
    """Plot a horizontal bar chart of the top 20 keywords by average CAR.

    Bars are coloured green for positive CAR and red for negative.
    Event count is annotated at the end of each bar.

    Args:
        keyword_results: List of result dicts from :func:`run_event_study`,
            each containing ``"keyword"``, ``"avg_CAR"``, and
            ``"event_count"`` keys.
    """
    if not keyword_results:
        logger.warning("No keyword results — skipping bar chart")
        return

    apply_plot_style()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "keyword_car_chart.png"

    sorted_results = sorted(keyword_results, key=lambda r: r["avg_CAR"], reverse=True)
    top20 = sorted_results[:20]

    keywords = [r["keyword"] for r in top20]
    cars = [r["avg_CAR"] for r in top20]
    counts = [r["event_count"] for r in top20]
    colours = [NAVY if c >= 0 else GOLD for c in cars]

    fig, ax = plt.subplots(figsize=(10, max(5, len(top20) * 0.55)))
    bars = ax.barh(keywords[::-1], cars[::-1], color=colours[::-1], edgecolor="white", height=0.7)

    for bar, count in zip(bars, counts[::-1]):
        x = bar.get_width()
        offset = 0.0002 if x >= 0 else -0.0002
        ha = "left" if x >= 0 else "right"
        ax.text(
            x + offset,
            bar.get_y() + bar.get_height() / 2,
            f"n={count}",
            va="center",
            ha=ha,
            fontsize=8,
            color="black",
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Average CAR")
    ax.set_title("Top 20 keywords by average Cumulative Abnormal Return (CAR)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Keyword CAR chart saved → %s", out_path)


# ---------------------------------------------------------------------------
# 6. plot_car_heatmap
# ---------------------------------------------------------------------------


def plot_car_heatmap(keyword_results: list[dict]) -> None:
    """Plot a heatmap of mean daily abnormal returns for the top 10 keywords.

    Rows are the top 10 keywords by absolute average CAR; columns are
    the event-window days from ``config.EVENT_WINDOW``.

    Args:
        keyword_results: List of result dicts from :func:`run_event_study`,
            each containing ``"keyword"`` and ``"daily_avg_ARs"`` keys.
    """
    if not keyword_results:
        logger.warning("No keyword results — skipping CAR heatmap")
        return

    apply_plot_style()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "car_heatmap.png"

    top10 = sorted(keyword_results, key=lambda r: abs(r["avg_CAR"]), reverse=True)[:10]
    pre, post = config.EVENT_WINDOW
    window_days = list(range(pre, post + 1))

    matrix = []
    row_labels = []
    for r in top10:
        daily = r.get("daily_avg_ARs", {})
        row = [daily.get(d, 0.0) for d in window_days]
        matrix.append(row)
        row_labels.append(r["keyword"])

    df_heat = pd.DataFrame(matrix, index=row_labels, columns=[str(d) for d in window_days])

    fig, ax = plt.subplots(figsize=(max(8, len(window_days) * 1.2), max(5, len(top10) * 0.7)))
    sns.heatmap(
        df_heat,
        annot=True,
        fmt=".4f",
        cmap=sns.diverging_palette(220, 40, as_cmap=True),
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Event window day (0 = event date)")
    ax.set_ylabel("Keyword")
    ax.set_title("Mean daily abnormal return by keyword and event-window day")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("CAR heatmap saved → %s", out_path)


# ---------------------------------------------------------------------------
# 7. plot_source_comparison
# ---------------------------------------------------------------------------


def plot_source_comparison(source_results: dict) -> None:
    """Plot a grouped bar chart of average CAR per data source.

    Args:
        source_results: Dict mapping source name → average CAR (float),
            as returned by :func:`run_source_comparison`.  The
            ``"t_test"`` key is excluded from the chart.
    """
    plot_data = {k: v for k, v in source_results.items() if k != "t_test" and isinstance(v, float)}
    if not plot_data:
        logger.warning("No source CAR data — skipping source comparison plot")
        return

    apply_plot_style()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "source_comparison.png"

    sources = list(plot_data.keys())
    avg_cars = [plot_data[s] for s in sources]
    colours = [NAVY if c >= 0 else GOLD for c in avg_cars]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(sources, avg_cars, color=colours, edgecolor="white", width=0.5)

    for bar in bars:
        h = bar.get_height()
        offset = 0.00005 if h >= 0 else -0.00005
        va = "bottom" if h >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            f"{h:.4f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Average CAR")
    ax.set_title("Average CAR by news source")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Source comparison plot saved → %s", out_path)


# ---------------------------------------------------------------------------
# 8. plot_frequency_vs_car
# ---------------------------------------------------------------------------


def plot_frequency_vs_car(keyword_results: list[dict]) -> None:
    """Scatter plot of keyword event frequency vs absolute average CAR.

    The top 5 keywords by absolute CAR are annotated with their names.

    Args:
        keyword_results: List of result dicts from :func:`run_event_study`.
    """
    if not keyword_results:
        logger.warning("No keyword results — skipping frequency vs CAR scatter")
        return

    apply_plot_style()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "frequency_vs_car.png"

    counts = [r["event_count"] for r in keyword_results]
    abs_cars = [abs(r["avg_CAR"]) for r in keyword_results]
    labels = [r["keyword"] for r in keyword_results]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(counts, abs_cars, alpha=0.7, color=NAVY, edgecolors="white", s=60)

    # Annotate top 5 by abs CAR
    top5_idx = sorted(range(len(abs_cars)), key=lambda i: abs_cars[i], reverse=True)[:5]
    for i in top5_idx:
        ax.annotate(
            labels[i],
            (counts[i], abs_cars[i]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=GOLD,
        )

    ax.set_xlabel("Event count (keyword frequency)")
    ax.set_ylabel("|Average CAR|")
    ax.set_title("Keyword frequency vs absolute average CAR")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Frequency vs CAR scatter saved → %s", out_path)


# ---------------------------------------------------------------------------
# 9. save_keyword_summary
# ---------------------------------------------------------------------------


def save_keyword_summary(results: list[dict]) -> None:
    """Persist keyword event-study results to a CSV file.

    Args:
        results: List of result dicts from :func:`run_event_study`,
            each containing ``"keyword"``, ``"event_count"``,
            ``"avg_CAR"``, ``"std_CAR"``, and ``"positive_pct"``.
    """
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.METRICS_DIR / "keyword_summary.csv"

    rows = [
        {
            "keyword": r["keyword"],
            "event_count": r["event_count"],
            "avg_CAR": r["avg_CAR"],
            "std_CAR": r["std_CAR"],
            "positive_pct": r["positive_pct"],
        }
        for r in results
    ]
    pd.DataFrame(rows).sort_values("avg_CAR", ascending=False).to_csv(out_path, index=False)
    logger.info("Keyword summary saved → %s (%d keywords)", out_path, len(rows))


# ---------------------------------------------------------------------------
# 10. run_keyword_analysis
# ---------------------------------------------------------------------------


def run_keyword_analysis(
    news_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> list[dict]:
    """Run the full keyword analysis pipeline end-to-end.

    Steps:

    1. Extract TF-IDF keywords from all headlines.
    2. Run :func:`run_event_study` for each keyword.
    3. Run :func:`run_source_comparison` across all sources.
    4. Produce all four diagnostic plots.
    5. Save the keyword summary CSV.

    Args:
        news_df: Combined news DataFrame with ``date``, ``ticker``,
            ``headline``, and ``source`` columns.
        prices: Wide adjusted-close price DataFrame indexed by date.

    Returns:
        list[dict]: List of event-study result dicts for every keyword
        that passed the ``config.MIN_EVENTS_PER_KEYWORD`` threshold.
    """
    logger.info("=== AlphaLens Keyword Analysis Pipeline ===")

    texts = news_df["headline"].dropna().tolist()

    # Step 1: Extract keywords
    tfidf_keywords = extract_tfidf_keywords(texts)
    logger.info("TF-IDF keywords: %d", len(tfidf_keywords))

    # Step 2: Run event studies
    keyword_results: list[dict] = []
    for kw in tfidf_keywords:
        result = run_event_study(kw, news_df, prices)
        if result:
            keyword_results.append(result)

    logger.info(
        "Event studies complete: %d / %d keywords passed threshold",
        len(keyword_results),
        len(tfidf_keywords),
    )

    # Step 3: Source comparison
    top_for_source = [r["keyword"] for r in keyword_results[: config.TOP_N_KEYWORDS]]
    source_results = run_source_comparison(news_df, prices, top_for_source)

    # Step 4: Plots
    plot_keyword_bar_chart(keyword_results)
    plot_car_heatmap(keyword_results)
    plot_source_comparison(source_results)
    plot_frequency_vs_car(keyword_results)

    # Step 5: Save
    if keyword_results:
        save_keyword_summary(keyword_results)

    logger.info("=== Keyword analysis complete: %d results ===", len(keyword_results))
    return keyword_results
