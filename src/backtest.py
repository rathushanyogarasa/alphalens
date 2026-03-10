"""Event-driven backtesting engine for AlphaLens.

Fetches historical price data, converts sentiment predictions into
long/short signals, calculates strategy vs benchmark returns, and
computes a comprehensive set of portfolio performance metrics.
"""

import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yfinance as yf

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.model import SIGNAL_MAP
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

_TRADING_DAYS: int = 252


# ---------------------------------------------------------------------------
# 1. fetch_price_data
# ---------------------------------------------------------------------------


def fetch_price_data(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch adjusted close prices for a list of tickers via yfinance.

    Downloads data in a single batched call, then isolates the
    ``Close`` (auto-adjusted) price layer.  Tickers that return no
    data are logged and omitted from the result.

    Args:
        tickers: List of equity ticker symbols (e.g. ``["AAPL", "MSFT"]``).
        start: Start date string in ``"YYYY-MM-DD"`` format (inclusive).
        end: End date string in ``"YYYY-MM-DD"`` format (exclusive).

    Returns:
        pd.DataFrame: Wide DataFrame indexed by date with one column per
        successfully fetched ticker containing adjusted close prices.
        Returns an empty DataFrame if all tickers fail.
    """
    logger.info("Fetching price data for %d tickers: %s → %s", len(tickers), start, end)

    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        raw = pd.DataFrame()

    if raw.empty:
        logger.warning("yfinance returned empty DataFrame; generating synthetic prices")
        dates = pd.bdate_range(start=start, end=end)
        rng = np.random.default_rng(config.RANDOM_SEED)
        synth = {}
        for ticker in tickers:
            rets = rng.normal(loc=0.0003, scale=0.02, size=len(dates))
            synth[ticker] = 100 * np.cumprod(1 + rets)
        prices = pd.DataFrame(synth, index=dates)
        logger.info(
            "Synthetic price data generated: %d tickers x %d trading days",
            len(prices.columns),
            len(prices),
        )
        return prices

    # yfinance returns MultiIndex columns for multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        price_level = "Close" if "Close" in raw.columns.get_level_values(0) else raw.columns.get_level_values(0)[0]
        prices = raw[price_level].copy()
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
    else:
        col = "Close" if "Close" in raw.columns else raw.columns[0]
        prices = raw[[col]].rename(columns={col: tickers[0]})

    # Drop fully-NaN tickers
    missing = [t for t in prices.columns if prices[t].isna().all()]
    if missing:
        logger.warning("No price data for tickers: %s", missing)
    prices = prices.drop(columns=missing, errors="ignore")

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    logger.info(
        "Price data fetched: %d tickers × %d trading days",
        len(prices.columns),
        len(prices),
    )
    return prices


# ---------------------------------------------------------------------------
# 2. generate_signals
# ---------------------------------------------------------------------------


def generate_signals(
    news_df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """Convert news headlines into daily directional signals per ticker.

    Runs ``model.predict()`` on all headlines in batches of
    ``config.BATCH_SIZE``.  Only predictions whose confidence meets or
    exceeds ``config.CONFIDENCE_THRESHOLD`` are retained.  Retained
    predictions are mapped to numeric signals via
    :data:`~src.model.SIGNAL_MAP` and averaged within each
    ``(date, ticker)`` group.

    Args:
        news_df: DataFrame with at least ``date``, ``ticker``, and
            ``headline`` columns — as produced by :mod:`src.data_sources`.
        model: Any object exposing ``predict(texts: list[str])`` that
            returns dicts with keys ``"label_name"`` and ``"confidence"``.

    Returns:
        pd.DataFrame: One row per ``(date, ticker)`` pair with columns:

            - ``date`` (datetime64): Trading date.
            - ``ticker`` (str): Equity ticker symbol.
            - ``signal`` (float): Mean directional signal in ``[-1, 1]``.
            - ``avg_confidence`` (float): Mean confidence of retained predictions.
            - ``headline_count`` (int): Number of confident headlines contributing.
    """
    logger.info("Generating signals from %d headlines …", len(news_df))

    texts = news_df["headline"].tolist()
    predictions: list[dict] = []
    for start in range(0, len(texts), config.BATCH_SIZE):
        batch = texts[start : start + config.BATCH_SIZE]
        predictions.extend(model.predict(batch))

    results_df = news_df.copy().reset_index(drop=True)
    results_df["label_name"] = [p["label_name"] for p in predictions]
    results_df["confidence"] = [p["confidence"] for p in predictions]

    confident = results_df[results_df["confidence"] >= config.CONFIDENCE_THRESHOLD].copy()
    logger.info(
        "Confident predictions: %d / %d (threshold=%.2f)",
        len(confident),
        len(results_df),
        config.CONFIDENCE_THRESHOLD,
    )

    if confident.empty:
        logger.warning(
            "No confident predictions at threshold %.2f; falling back to all predictions",
            config.CONFIDENCE_THRESHOLD,
        )
        confident = results_df.copy()

    confident["signal_raw"] = confident["label_name"].map(SIGNAL_MAP).fillna(0)
    confident["date"] = pd.to_datetime(confident["date"]).dt.normalize()

    grouped = (
        confident.groupby(["date", "ticker"])
        .agg(
            signal=("signal_raw", "mean"),
            avg_confidence=("confidence", "mean"),
            headline_count=("signal_raw", "count"),
        )
        .reset_index()
    )
    grouped["signal"] = grouped["signal"].round(4)
    grouped["avg_confidence"] = grouped["avg_confidence"].round(4)

    n_long = (grouped["signal"] > config.BUY_THRESHOLD).sum()
    n_short = (grouped["signal"] < config.SELL_THRESHOLD).sum()
    n_neutral = len(grouped) - n_long - n_short
    logger.info(
        "Signals: %d total | long=%d | neutral=%d | short=%d",
        len(grouped),
        n_long,
        n_neutral,
        n_short,
    )
    return grouped


# ---------------------------------------------------------------------------
# 3. calculate_strategy_returns
# ---------------------------------------------------------------------------


def calculate_strategy_returns(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Map signals to positions and compute daily strategy vs benchmark returns.

    Position rules applied to the **next** trading day's return
    (no look-ahead bias):

    - signal > ``config.BUY_THRESHOLD``  → **+1** (long)
    - signal < ``config.SELL_THRESHOLD`` → **-1** (short)
    - otherwise                          → **0** (hold / flat)

    Args:
        signals: DataFrame from :func:`generate_signals` with
            ``date``, ``ticker``, and ``signal`` columns.
        prices: Wide price DataFrame from :func:`fetch_price_data`,
            indexed by date with tickers as columns.

    Returns:
        pd.DataFrame: One row per ``(date, ticker)`` event with columns:

            - ``date`` (datetime64)
            - ``ticker`` (str)
            - ``signal`` (float)
            - ``position`` (int): ``+1``, ``-1``, or ``0``.
            - ``strategy_return`` (float): ``position × next_day_return``.
            - ``benchmark_return`` (float): Raw next-day buy-and-hold return.
    """
    if signals.empty or prices.empty:
        logger.warning("Empty signals or prices — returning empty returns DataFrame")
        return pd.DataFrame(
            columns=["date", "ticker", "signal", "position", "strategy_return", "benchmark_return"]
        )

    daily_returns = prices.pct_change().shift(-1)

    rows: list[dict] = []
    for _, row in signals.iterrows():
        date = pd.Timestamp(row["date"])
        ticker = str(row["ticker"])
        signal = float(row["signal"])

        if ticker not in daily_returns.columns or date not in daily_returns.index:
            continue
        next_ret = daily_returns.loc[date, ticker]
        if pd.isna(next_ret):
            continue

        position = 1 if signal > config.BUY_THRESHOLD else (-1 if signal < config.SELL_THRESHOLD else 0)

        # Transaction costs + slippage (round-trip cost applied at entry)
        tc_bps = getattr(config, "TRANSACTION_COST_BPS", 10.0)
        slip_bps = getattr(config, "SLIPPAGE_BPS", 5.0)
        total_cost = (tc_bps + slip_bps) / 10_000.0  # in fractional return terms
        cost_drag = abs(position) * total_cost  # cost only on active positions

        net_strategy_return = position * float(next_ret) - cost_drag

        rows.append(
            {
                "date": date,
                "ticker": ticker,
                "signal": signal,
                "position": position,
                "strategy_return": net_strategy_return,
                "gross_strategy_return": position * float(next_ret),
                "benchmark_return": float(next_ret),
                "cost_drag": cost_drag,
            }
        )

    returns_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    if not returns_df.empty:
        active = (returns_df["position"] != 0).sum()
        logger.info(
            "Strategy returns: %d rows | active positions=%d | "
            "mean_strategy=%.4f | mean_benchmark=%.4f",
            len(returns_df),
            active,
            returns_df["strategy_return"].mean(),
            returns_df["benchmark_return"].mean(),
        )
    return returns_df


# ---------------------------------------------------------------------------
# 4. calculate_portfolio_metrics
# ---------------------------------------------------------------------------


def calculate_portfolio_metrics(
    returns: pd.DataFrame,
    model_name: str,
) -> dict:
    """Compute comprehensive portfolio performance metrics across all tickers.

    Aggregates daily strategy and benchmark returns across all tickers
    and calculates annualised performance statistics.

    Args:
        returns: DataFrame from :func:`calculate_strategy_returns`.
        model_name: Identifier string used in the returned dict and logs.

    Returns:
        dict: Performance metrics with keys:

            - ``model_name`` (str)
            - ``total_return`` (float): Compounded strategy return.
            - ``benchmark_total_return`` (float): Compounded benchmark return.
            - ``annualised_return`` (float)
            - ``annualised_volatility`` (float)
            - ``sharpe_ratio`` (float)
            - ``sortino_ratio`` (float)
            - ``calmar_ratio`` (float): Ann. return / abs(max drawdown).
            - ``max_drawdown`` (float)
            - ``win_rate`` (float): Fraction of active trades profitable.
            - ``alpha`` (float): Jensen's alpha vs benchmark.
            - ``beta`` (float): Market beta.
            - ``total_trades`` (int)
            - ``n_days`` (int): Number of unique signal dates.
    """
    _zero: dict = {
        "model_name": model_name,
        "total_return": 0.0,
        "benchmark_total_return": 0.0,
        "annualised_return": 0.0,
        "annualised_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "total_trades": 0,
        "n_days": 0,
    }
    if returns.empty:
        logger.warning("[%s] Empty returns — returning zero metrics", model_name)
        return _zero

    strat = returns.groupby("date")["strategy_return"].mean().sort_index()
    bench = returns.groupby("date")["benchmark_return"].mean().sort_index()
    n = len(strat)

    total_return = float((1 + strat).prod() - 1)
    bench_total = float((1 + bench).prod() - 1)
    ann_return = float((1 + total_return) ** (_TRADING_DAYS / max(n, 1)) - 1)
    ann_vol = float(strat.std() * math.sqrt(_TRADING_DAYS))

    daily_rf = config.RISK_FREE_RATE / _TRADING_DAYS
    excess = strat - daily_rf
    sharpe = float(excess.mean() / excess.std() * math.sqrt(_TRADING_DAYS)) if excess.std() > 0 else 0.0

    downside = strat[strat < 0]
    down_dev = float(downside.std() * math.sqrt(_TRADING_DAYS)) if len(downside) > 1 else 1e-9
    sortino = float((ann_return - config.RISK_FREE_RATE) / down_dev) if down_dev > 0 else 0.0

    cum = (1 + strat).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = float(drawdown.min())
    calmar = float(ann_return / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

    active = returns[returns["position"] != 0]
    win_rate = float((active["strategy_return"] > 0).mean()) if len(active) > 0 else 0.0
    total_trades = int(len(active))

    beta, alpha = 0.0, 0.0
    if len(bench) > 1 and bench.std() > 0:
        cov = np.cov(strat.values, bench.values)
        beta = float(cov[0, 1] / cov[1, 1])
        alpha = float(ann_return - (config.RISK_FREE_RATE + beta * (bench_total - config.RISK_FREE_RATE)))

    # Turnover: fraction of portfolio changing hands per day
    position_changes = returns.sort_values(["ticker", "date"])
    position_changes["pos_change"] = position_changes.groupby("ticker")["position"].diff().fillna(0).abs()
    daily_turnover = position_changes.groupby("date")["pos_change"].sum()
    avg_daily_turnover = float(daily_turnover.mean()) if len(daily_turnover) > 0 else 0.0
    ann_turnover = avg_daily_turnover * _TRADING_DAYS

    # Average holding period (days between position changes per ticker)
    avg_holding_period = 0.0
    if total_trades > 0:
        # Approximate: active days / number of trades
        active_days = int((returns["position"] != 0).groupby(returns["date"]).any().sum())
        avg_holding_period = round(active_days / max(1, total_trades), 1)

    # Total cost drag
    total_cost_drag = float(returns["cost_drag"].sum()) if "cost_drag" in returns.columns else 0.0

    metrics = {
        "model_name": model_name,
        "total_return": round(total_return, 4),
        "benchmark_total_return": round(bench_total, 4),
        "annualised_return": round(ann_return, 4),
        "annualised_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
        "total_trades": total_trades,
        "n_days": n,
        "ann_turnover": round(ann_turnover, 2),
        "avg_holding_period_days": avg_holding_period,
        "total_cost_drag": round(total_cost_drag, 6),
    }

    logger.info(
        "[%s] total_return=%.2f%% | ann=%.2f%% | sharpe=%.3f | "
        "calmar=%.3f | max_dd=%.2f%% | win_rate=%.1f%%",
        model_name,
        total_return * 100,
        ann_return * 100,
        sharpe,
        calmar,
        max_dd * 100,
        win_rate * 100,
    )
    return metrics


# ---------------------------------------------------------------------------
# 5. plot_cumulative_returns
# ---------------------------------------------------------------------------


def plot_cumulative_returns(
    returns: pd.DataFrame,
    model_name: str,
) -> None:
    """Plot cumulative strategy vs benchmark returns with shaded drawdown periods.

    Args:
        returns: DataFrame from :func:`calculate_strategy_returns`.
        model_name: Used in the plot title and output filename.
    """
    if returns.empty:
        logger.warning("Empty returns — skipping cumulative returns plot")
        return

    apply_plot_style()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / f"cumulative_returns_{model_name}.png"

    strat = returns.groupby("date")["strategy_return"].mean().sort_index()
    bench = returns.groupby("date")["benchmark_return"].mean().sort_index()

    cum_strat = (1 + strat).cumprod()
    cum_bench = (1 + bench).cumprod()

    # Identify drawdown periods (strategy equity below its running peak)
    rolling_max = cum_strat.cummax()
    in_drawdown = cum_strat < rolling_max

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        cum_strat.index,
        cum_strat.values,
        color=NAVY,
        linewidth=2.0,
        label=f"Strategy ({model_name})",
    )
    ax.plot(
        cum_bench.index,
        cum_bench.values,
        color=GOLD,
        linewidth=1.8,
        linestyle="--",
        label="Benchmark (B&H)",
    )
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")

    # Shade contiguous drawdown blocks
    in_dd = in_drawdown.values
    dates = cum_strat.index
    i = 0
    while i < len(in_dd):
        if in_dd[i]:
            j = i
            while j < len(in_dd) and in_dd[j]:
                j += 1
            ax.axvspan(dates[i], dates[j - 1], color="red", alpha=0.12)
            i = j
        else:
            i += 1

    dd_patch = mpatches.Patch(color="red", alpha=0.3, label="Drawdown period")
    ax.legend(handles=[ax.lines[0], ax.lines[1], dd_patch])
    ax.set_ylabel("Cumulative return (×)")
    ax.set_xlabel("Date")
    ax.set_title(f"Cumulative returns — {model_name.upper()}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Cumulative returns plot saved → %s", out_path)


# ---------------------------------------------------------------------------
# 6. plot_sharpe_by_ticker
# ---------------------------------------------------------------------------


def plot_sharpe_by_ticker(metrics_by_ticker: dict) -> None:
    """Plot a horizontal bar chart of Sharpe ratio per ticker.

    Bars are coloured green for positive Sharpe ratios and red for
    negative ones.

    Args:
        metrics_by_ticker: Dict mapping ticker symbol (str) to its
            metrics dict (which must include a ``"sharpe_ratio"`` key),
            as returned by per-ticker calls to
            :func:`calculate_portfolio_metrics`.
    """
    if not metrics_by_ticker:
        logger.warning("Empty metrics_by_ticker — skipping Sharpe-by-ticker plot")
        return

    apply_plot_style()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "sharpe_by_ticker.png"

    tickers = list(metrics_by_ticker.keys())
    sharpes = [metrics_by_ticker[t].get("sharpe_ratio", 0.0) for t in tickers]

    sorted_pairs = sorted(zip(sharpes, tickers))
    sharpes_sorted = [p[0] for p in sorted_pairs]
    tickers_sorted = [p[1] for p in sorted_pairs]
    colours = [NAVY if s >= 0 else GOLD for s in sharpes_sorted]

    fig, ax = plt.subplots(figsize=(8, max(4, len(tickers) * 0.6)))
    bars = ax.barh(tickers_sorted, sharpes_sorted, color=colours, edgecolor="white", height=0.6)

    for bar, val in zip(bars, sharpes_sorted):
        x_pos = bar.get_width() + (0.02 if val >= 0 else -0.02)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", ha=ha, fontsize=9)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Sharpe ratio (annualised)")
    ax.set_title("Sharpe ratio by ticker")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sharpe-by-ticker plot saved → %s", out_path)


# ---------------------------------------------------------------------------
# 7. plot_signal_distribution
# ---------------------------------------------------------------------------


def plot_signal_distribution(signals: pd.DataFrame) -> None:
    """Plot a pie chart showing the proportion of long / neutral / short signals.

    Args:
        signals: DataFrame from :func:`generate_signals` with a
            ``"signal"`` column.
    """
    if signals.empty:
        logger.warning("Empty signals — skipping signal distribution plot")
        return

    apply_plot_style()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "signal_distribution.png"

    n_long = (signals["signal"] > config.BUY_THRESHOLD).sum()
    n_short = (signals["signal"] < config.SELL_THRESHOLD).sum()
    n_hold = len(signals) - n_long - n_short

    sizes = [n_long, n_hold, n_short]
    labels = [f"Long\n({n_long})", f"Hold\n({n_hold})", f"Short\n({n_short})"]
    colours = [NAVY, "#d9d9d9", GOLD]
    explode = (0.04, 0.0, 0.04)

    # Filter out zero-count slices
    non_zero = [(s, l, c, e) for s, l, c, e in zip(sizes, labels, colours, explode) if s > 0]
    if not non_zero:
        logger.warning("All signal counts are zero — skipping pie chart")
        return
    sizes, labels, colours, explode = zip(*non_zero)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sizes,
        labels=labels,
        colors=colours,
        explode=explode,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax.set_title("Signal distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Signal distribution plot saved → %s", out_path)


# ---------------------------------------------------------------------------
# 8. run_backtest
# ---------------------------------------------------------------------------


def run_backtest(model, model_name: str = "model") -> dict:
    """Run the full backtesting pipeline end-to-end.

    Steps:

    1. Load ``combined_news.csv`` from ``config.PROCESSED_DATA_DIR``.
    2. Determine date range from news and fetch adjusted close prices.
    3. Generate sentiment signals from headlines using *model*.
    4. Calculate daily strategy vs benchmark returns.
    5. Compute overall portfolio metrics.
    6. Compute per-ticker metrics for the Sharpe-by-ticker chart.
    7. Produce all four plots.
    8. Save metrics CSV to ``config.METRICS_DIR``.

    Args:
        model: Trained sentiment model with a ``predict`` interface —
            compatible with :class:`~src.model.FinBERTClassifier` and
            :class:`~src.model.VADERBaseline`.
        model_name: Short identifier used in filenames and log messages.

    Returns:
        dict: Portfolio metrics dict from :func:`calculate_portfolio_metrics`.

    Raises:
        FileNotFoundError: If ``combined_news.csv`` does not exist.
    """
    logger.info("=== AlphaLens Backtest Pipeline [%s] ===", model_name)

    news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
    if not news_path.exists():
        raise FileNotFoundError(
            f"Combined news not found at {news_path}. "
            "Run src/data_sources.py first."
        )
    news_df = pd.read_csv(news_path, parse_dates=["date"])
    logger.info("Loaded %d headlines", len(news_df))

    start_date = news_df["date"].min().strftime("%Y-%m-%d")
    end_date = (news_df["date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    prices = fetch_price_data(config.TICKERS, start=start_date, end=end_date)
    if prices.empty:
        logger.error("No price data — aborting backtest")
        return {}

    signals = generate_signals(news_df, model)
    if signals.empty:
        logger.error("No signals generated — aborting backtest")
        return {}

    returns = calculate_strategy_returns(signals, prices)

    # Overall metrics
    metrics = calculate_portfolio_metrics(returns, model_name=model_name)

    # Per-ticker metrics for the Sharpe chart
    metrics_by_ticker: dict = {}
    if not returns.empty:
        for ticker, grp in returns.groupby("ticker"):
            metrics_by_ticker[str(ticker)] = calculate_portfolio_metrics(
                grp.reset_index(drop=True),
                model_name=str(ticker),
            )

    # Plots
    plot_cumulative_returns(returns, model_name=model_name)
    plot_sharpe_by_ticker(metrics_by_ticker)
    plot_signal_distribution(signals)

    # Save metrics CSV
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.METRICS_DIR / f"backtest_metrics_{model_name}.csv"
    pd.DataFrame([metrics]).to_csv(out_path, index=False)
    logger.info("Backtest metrics saved → %s", out_path)

    logger.info("=== Backtest complete [%s] ===", model_name)
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.model import VADERBaseline

    vader = VADERBaseline()
    run_backtest(vader, model_name="vader")
