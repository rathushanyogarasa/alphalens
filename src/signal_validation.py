"""Signal validation and predictive accuracy testing for AlphaLens.

Tests whether sentiment signals actually predict stock price movements
using a battery of statistical methods from quantitative finance:

- **Directional accuracy (hit rate)**: Does signal direction match
  return direction?
- **Information Coefficient (IC)**: Spearman rank correlation between
  signal score and subsequent returns.
- **Quantile portfolio analysis**: Do high-signal stocks outperform
  low-signal stocks?
- **Signal decay analysis**: How many days does the signal stay
  predictive?
- **Conditional return t-test**: Are BUY-signal returns statistically
  significantly positive?
- **Confidence calibration**: Does higher model confidence correlate
  with higher directional accuracy?
"""

import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_TRADING_DAYS = 252
_LAGS = [1, 2, 3, 5, 10]          # days to test IC decay over
_N_QUANTILES = 5                    # quintiles for portfolio analysis


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def build_signal_return_df(
    news_df: pd.DataFrame,
    prices: pd.DataFrame,
    model,
    max_lag: int = 10,
    confidence_threshold: float | None = None,
) -> pd.DataFrame:
    """Pair each confident headline with forward returns at multiple lags.

    Runs ``model.predict()`` on all headlines, filters to those meeting
    ``config.CONFIDENCE_THRESHOLD``, and joins each prediction to the
    corresponding stock's forward return at each lag day.

    Args:
        news_df: Combined news DataFrame with ``date``, ``ticker``,
            ``headline``, and ``source`` columns.
        prices: Wide adjusted-close price DataFrame indexed by date.
        model: Sentiment model with a ``predict(texts)`` interface.
        max_lag: Maximum forward-return lag in trading days to compute.

    Returns:
        pd.DataFrame: One row per confident (date, ticker) pair with
        columns ``date``, ``ticker``, ``signal``, ``confidence``,
        ``label_name``, and ``return_lag_{n}`` for each lag in
        ``range(1, max_lag + 1)``.
    """
    from src.model import SIGNAL_MAP

    logger.info("Building signal-return dataset from %d headlines …", len(news_df))

    texts = news_df["headline"].tolist()
    predictions: list[dict] = []
    for start in range(0, len(texts), config.BATCH_SIZE):
        predictions.extend(model.predict(texts[start : start + config.BATCH_SIZE]))

    news_df = news_df.copy().reset_index(drop=True)
    news_df["label_name"] = [p["label_name"] for p in predictions]
    news_df["confidence"] = [p["confidence"] for p in predictions]
    news_df["signal"] = news_df["label_name"].map(SIGNAL_MAP).fillna(0)
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()

    # Keep only non-neutral, non-uncertain predictions; apply confidence filter
    # Use a lower threshold for validation than live trading (more samples = better stats)
    thresh = confidence_threshold if confidence_threshold is not None else 0.50
    confident = news_df[
        (news_df["confidence"] >= thresh)
        & (news_df["label_name"] != "uncertain")
        & (news_df["label_name"] != "neutral")
    ].copy()
    logger.info(
        "Non-neutral confident headlines (threshold=%.2f): %d / %d",
        thresh, len(confident), len(news_df),
    )

    prices_sorted = prices.sort_index()
    price_dates = prices_sorted.index

    rows: list[dict] = []
    for _, row in confident.iterrows():
        date = pd.Timestamp(row["date"])
        ticker = str(row["ticker"])

        if ticker not in prices_sorted.columns:
            continue

        pos = price_dates.searchsorted(date)
        if pos >= len(price_dates):
            continue

        p0 = prices_sorted[ticker].iloc[pos]
        if pd.isna(p0) or p0 == 0:
            continue

        lag_returns: dict[str, float] = {}
        for lag in range(1, max_lag + 1):
            lag_pos = pos + lag
            if lag_pos < len(price_dates):
                p_lag = prices_sorted[ticker].iloc[lag_pos]
                lag_returns[f"return_lag_{lag}"] = (
                    float((p_lag - p0) / p0) if not pd.isna(p_lag) else np.nan
                )
            else:
                lag_returns[f"return_lag_{lag}"] = np.nan

        rows.append(
            {
                "date": date,
                "ticker": ticker,
                "signal": float(row["signal"]),
                "confidence": float(row["confidence"]),
                "label_name": str(row["label_name"]),
                "source": str(row.get("source", "")),
                **lag_returns,
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["return_lag_1"])
    logger.info("Signal-return dataset: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 1. Directional accuracy (hit rate)
# ---------------------------------------------------------------------------


def compute_hit_rate(sr_df: pd.DataFrame, lag: int = 1) -> dict:
    """Compute directional accuracy of sentiment signals.

    For each signal direction (BUY/SELL) counts the fraction of cases
    where the forward return matched the predicted direction.

    Args:
        sr_df: Signal-return DataFrame from :func:`build_signal_return_df`.
        lag: Forward-return lag in days to evaluate.

    Returns:
        dict: Keys ``overall``, ``buy``, ``sell``, ``n_buy``, ``n_sell``,
        ``n_total`` — all hit-rate fractions or counts.
    """
    col = f"return_lag_{lag}"
    df = sr_df.dropna(subset=[col])

    buy_mask = df["signal"] > config.BUY_THRESHOLD
    sell_mask = df["signal"] < config.SELL_THRESHOLD

    buy_hits = (df.loc[buy_mask, col] > 0).sum()
    sell_hits = (df.loc[sell_mask, col] < 0).sum()
    n_buy = buy_mask.sum()
    n_sell = sell_mask.sum()
    n_total = n_buy + n_sell

    overall = (buy_hits + sell_hits) / n_total if n_total > 0 else 0.0
    buy_hr = buy_hits / n_buy if n_buy > 0 else 0.0
    sell_hr = sell_hits / n_sell if n_sell > 0 else 0.0

    result = {
        "lag": lag,
        "overall": round(float(overall), 4),
        "buy": round(float(buy_hr), 4),
        "sell": round(float(sell_hr), 4),
        "n_buy": int(n_buy),
        "n_sell": int(n_sell),
        "n_total": int(n_total),
    }
    logger.info(
        "Hit rate (lag=%d): overall=%.1f%% | BUY=%.1f%% (n=%d) | SELL=%.1f%% (n=%d)",
        lag,
        overall * 100,
        buy_hr * 100,
        n_buy,
        sell_hr * 100,
        n_sell,
    )
    return result


# ---------------------------------------------------------------------------
# 2. Information Coefficient
# ---------------------------------------------------------------------------


def compute_ic(sr_df: pd.DataFrame, lag: int = 1) -> dict:
    """Compute the Information Coefficient (Spearman rank correlation).

    IC measures whether higher signal scores rank-correlate with higher
    subsequent returns across the cross-section.

    Args:
        sr_df: Signal-return DataFrame.
        lag: Forward-return lag in days.

    Returns:
        dict: ``ic`` (float), ``p_value`` (float), ``significant`` (bool),
        ``n`` (int), ``lag`` (int).
    """
    col = f"return_lag_{lag}"
    df = sr_df.dropna(subset=[col, "signal"])
    if len(df) < 5:
        return {"lag": lag, "ic": 0.0, "p_value": 1.0, "significant": False, "n": len(df)}

    ic, p_value = stats.spearmanr(df["signal"], df[col])
    result = {
        "lag": lag,
        "ic": round(float(ic), 4),
        "p_value": round(float(p_value), 4),
        "significant": bool(p_value < 0.05),
        "n": len(df),
    }
    sig_str = "SIGNIFICANT" if result["significant"] else "not significant"
    logger.info(
        "IC (lag=%d): IC=%.4f  p=%.4f  %s  (n=%d)",
        lag, ic, p_value, sig_str, len(df),
    )
    return result


# ---------------------------------------------------------------------------
# 3. Quantile portfolio analysis
# ---------------------------------------------------------------------------


def compute_quantile_returns(
    sr_df: pd.DataFrame,
    lag: int = 1,
    n_quantiles: int = _N_QUANTILES,
) -> pd.DataFrame:
    """Divide signals into quantile buckets and compute mean forward returns.

    The spread between the top and bottom quantile (long-short return)
    measures the signal's alpha-generating potential.

    Args:
        sr_df: Signal-return DataFrame.
        lag: Forward-return lag in days.
        n_quantiles: Number of equal-width quantile buckets.

    Returns:
        pd.DataFrame: One row per quantile with columns ``quantile``,
        ``mean_signal``, ``mean_return``, ``count``.
    """
    col = f"return_lag_{lag}"
    df = sr_df.dropna(subset=[col, "signal"]).copy()
    if len(df) < n_quantiles * 2:
        logger.warning("Too few rows (%d) for quantile analysis", len(df))
        return pd.DataFrame()

    df["quantile"] = pd.qcut(df["signal"], q=n_quantiles, labels=False, duplicates="drop")
    result = (
        df.groupby("quantile")
        .agg(
            mean_signal=("signal", "mean"),
            mean_return=(col, "mean"),
            count=(col, "count"),
        )
        .reset_index()
    )
    result["quantile"] = result["quantile"] + 1  # 1-indexed

    if result.empty or len(result) < 2:
        logger.warning("Quantile analysis produced fewer than 2 groups — skipping spread")
        return result

    spread = float(result["mean_return"].iloc[-1] - result["mean_return"].iloc[0])
    logger.info(
        "Quantile returns (lag=%d): Q%d=%.4f Q1=%.4f spread=%.4f",
        lag, n_quantiles,
        result["mean_return"].iloc[-1],
        result["mean_return"].iloc[0],
        spread,
    )
    return result


# ---------------------------------------------------------------------------
# 4. Signal decay (IC at multiple lags)
# ---------------------------------------------------------------------------


def compute_ic_decay(
    sr_df: pd.DataFrame,
    lags: list[int] = _LAGS,
) -> pd.DataFrame:
    """Compute IC at multiple forward lags to measure signal decay.

    A quickly decaying IC suggests the signal is priced in rapidly
    (consistent with semi-strong market efficiency); a slowly decaying IC
    suggests a persistent edge.

    Args:
        sr_df: Signal-return DataFrame with all lag columns present.
        lags: List of forward-return lag days to test.

    Returns:
        pd.DataFrame: One row per lag with columns ``lag``, ``ic``,
        ``p_value``, ``significant``, ``n``.
    """
    logger.info("Computing IC decay over lags: %s", lags)
    rows = [compute_ic(sr_df, lag=lag) for lag in lags]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Conditional return t-test
# ---------------------------------------------------------------------------


def compute_conditional_ttest(
    sr_df: pd.DataFrame,
    lag: int = 1,
) -> dict:
    """Test whether BUY/SELL signal returns are statistically non-zero.

    Runs a one-sample Welch t-test: ``H0: mean(return | signal) == 0``.
    Rejection of H0 at p < 0.05 indicates the signal predicts returns
    better than random.

    Args:
        sr_df: Signal-return DataFrame.
        lag: Forward-return lag in days.

    Returns:
        dict: Results for ``buy`` and ``sell`` groups, each containing
        ``mean_return``, ``t_stat``, ``p_value``, ``significant``, ``n``.
    """
    col = f"return_lag_{lag}"
    df = sr_df.dropna(subset=[col])

    buy_returns = df.loc[df["signal"] > config.BUY_THRESHOLD, col].values
    sell_returns = df.loc[df["signal"] < config.SELL_THRESHOLD, col].values

    def _ttest(arr: np.ndarray, direction: str) -> dict:
        if len(arr) < 3:
            return {"mean_return": 0.0, "t_stat": 0.0, "p_value": 1.0, "significant": False, "n": len(arr)}
        t, p = stats.ttest_1samp(arr, popmean=0.0)
        sig = bool(p < 0.05 and ((direction == "buy" and np.mean(arr) > 0) or
                                   (direction == "sell" and np.mean(arr) < 0)))
        return {
            "mean_return": round(float(np.mean(arr)), 6),
            "t_stat": round(float(t), 4),
            "p_value": round(float(p), 4),
            "significant": sig,
            "n": len(arr),
        }

    buy_result = _ttest(buy_returns, "buy")
    sell_result = _ttest(sell_returns, "sell")

    logger.info(
        "t-test (lag=%d): BUY mean=%.4f t=%.3f p=%.4f %s | SELL mean=%.4f t=%.3f p=%.4f %s",
        lag,
        buy_result["mean_return"], buy_result["t_stat"], buy_result["p_value"],
        "SIG" if buy_result["significant"] else "n.s.",
        sell_result["mean_return"], sell_result["t_stat"], sell_result["p_value"],
        "SIG" if sell_result["significant"] else "n.s.",
    )
    return {"buy": buy_result, "sell": sell_result, "lag": lag}


# ---------------------------------------------------------------------------
# 6. Confidence calibration
# ---------------------------------------------------------------------------


def compute_calibration(
    sr_df: pd.DataFrame,
    lag: int = 1,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Test whether model confidence predicts directional accuracy.

    Bins predictions by confidence score and computes the directional
    hit rate within each bin.  A well-calibrated model shows monotonically
    increasing hit rate as confidence increases.

    Args:
        sr_df: Signal-return DataFrame.
        lag: Forward-return lag in days.
        n_bins: Number of confidence bins.

    Returns:
        pd.DataFrame: One row per bin with columns ``conf_bin``,
        ``mean_confidence``, ``hit_rate``, ``count``.
    """
    col = f"return_lag_{lag}"
    df = sr_df.dropna(subset=[col]).copy()

    # Directional correctness
    df["correct"] = (
        ((df["signal"] > config.BUY_THRESHOLD) & (df[col] > 0)) |
        ((df["signal"] < config.SELL_THRESHOLD) & (df[col] < 0))
    )
    df = df[df["signal"].abs() > 0]  # drop neutral signals

    if len(df) < n_bins * 2:
        logger.warning("Too few rows (%d) for calibration analysis", len(df))
        return pd.DataFrame()

    df["conf_bin"] = pd.qcut(df["confidence"], q=n_bins, duplicates="drop")
    result = (
        df.groupby("conf_bin", observed=True)
        .agg(
            mean_confidence=("confidence", "mean"),
            hit_rate=("correct", "mean"),
            count=("correct", "count"),
        )
        .reset_index()
    )
    result["conf_bin"] = result["conf_bin"].astype(str)
    logger.info("Calibration (lag=%d): %s", lag, result[["mean_confidence", "hit_rate"]].to_dict("records"))
    return result


# ---------------------------------------------------------------------------
# 7. Per-ticker IC breakdown
# ---------------------------------------------------------------------------


def compute_ic_by_ticker(
    sr_df: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
    """Compute IC separately for each ticker.

    Identifies which stocks the signal works best and worst for,
    useful for debugging model failure modes.

    Args:
        sr_df: Signal-return DataFrame.
        lag: Forward-return lag in days.

    Returns:
        pd.DataFrame: One row per ticker with columns ``ticker``,
        ``ic``, ``p_value``, ``significant``, ``n``.
    """
    col = f"return_lag_{lag}"
    rows: list[dict] = []
    for ticker, grp in sr_df.groupby("ticker"):
        sub = grp.dropna(subset=[col, "signal"])
        if len(sub) < 5:
            continue
        ic, p = stats.spearmanr(sub["signal"], sub[col])
        rows.append({
            "ticker": ticker,
            "ic": round(float(ic), 4),
            "p_value": round(float(p), 4),
            "significant": bool(p < 0.05),
            "n": len(sub),
        })
    df = pd.DataFrame(rows).sort_values("ic", ascending=False)
    logger.info("IC by ticker (lag=%d):\n%s", lag, df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_ic_decay(ic_decay_df: pd.DataFrame, model_name: str) -> None:
    """Plot IC vs. lag day (signal decay curve).

    Args:
        ic_decay_df: DataFrame from :func:`compute_ic_decay`.
        model_name: Used in title and filename.
    """
    if ic_decay_df.empty:
        return
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PLOTS_DIR / f"ic_decay_{model_name}.png"

    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["green" if ic >= 0 else "red" for ic in ic_decay_df["ic"]]
    ax.bar(ic_decay_df["lag"].astype(str), ic_decay_df["ic"], color=colours, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)

    # Mark significant bars
    for _, row in ic_decay_df.iterrows():
        if row["significant"]:
            ax.text(
                str(int(row["lag"])), row["ic"] + (0.001 if row["ic"] >= 0 else -0.003),
                "*", ha="center", fontsize=12, color="navy",
            )

    ax.set_xlabel("Forward lag (days)")
    ax.set_ylabel("IC (Spearman rank correlation)")
    ax.set_title(f"Signal IC decay — {model_name}  (* = p < 0.05)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("IC decay plot saved → %s", out)


def plot_quantile_returns(quantile_df: pd.DataFrame, model_name: str, lag: int = 1) -> None:
    """Plot mean return per signal quantile (quintile bar chart).

    Args:
        quantile_df: DataFrame from :func:`compute_quantile_returns`.
        model_name: Used in title and filename.
        lag: Lag day used (for title).
    """
    if quantile_df.empty:
        return
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PLOTS_DIR / f"quantile_returns_{model_name}_lag{lag}.png"

    colours = ["green" if r >= 0 else "red" for r in quantile_df["mean_return"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(quantile_df["quantile"].astype(str), quantile_df["mean_return"],
           color=colours, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Signal quantile (1=most negative, 5=most positive)")
    ax.set_ylabel(f"Mean {lag}-day forward return")
    ax.set_title(f"Quantile portfolio returns — {model_name}  (lag={lag}d)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Quantile returns plot saved → %s", out)


def plot_calibration(calib_df: pd.DataFrame, model_name: str) -> None:
    """Plot confidence calibration: confidence bin vs hit rate.

    Args:
        calib_df: DataFrame from :func:`compute_calibration`.
        model_name: Used in title and filename.
    """
    if calib_df.empty:
        return
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PLOTS_DIR / f"calibration_{model_name}.png"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(calib_df["mean_confidence"], calib_df["hit_rate"],
            marker="o", color="navy", linewidth=2, label="Actual hit rate")
    ax.axhline(0.5, color="gray", linewidth=1, linestyle="--", label="Random baseline (50%)")
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Directional hit rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Confidence calibration — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Calibration plot saved → %s", out)


def plot_hit_rate_by_ticker(ic_ticker_df: pd.DataFrame, model_name: str) -> None:
    """Horizontal bar chart of per-ticker IC.

    Args:
        ic_ticker_df: DataFrame from :func:`compute_ic_by_ticker`.
        model_name: Used in title and filename.
    """
    if ic_ticker_df.empty:
        return
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PLOTS_DIR / f"ic_by_ticker_{model_name}.png"

    colours = ["green" if ic >= 0 else "red" for ic in ic_ticker_df["ic"]]
    fig, ax = plt.subplots(figsize=(8, max(4, len(ic_ticker_df) * 0.55)))
    ax.barh(ic_ticker_df["ticker"], ic_ticker_df["ic"], color=colours, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Information Coefficient (Spearman)")
    ax.set_title(f"IC by ticker — {model_name}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("IC by ticker plot saved → %s", out)


def plot_return_distributions(sr_df: pd.DataFrame, model_name: str, lag: int = 1) -> None:
    """KDE plot of return distributions conditioned on signal direction.

    Shows whether BUY signals produce a positively shifted return
    distribution compared to SELL signals.

    Args:
        sr_df: Signal-return DataFrame.
        model_name: Used in title and filename.
        lag: Forward-return lag in days.
    """
    col = f"return_lag_{lag}"
    df = sr_df.dropna(subset=[col])
    buy_rets = df.loc[df["signal"] > config.BUY_THRESHOLD, col]
    sell_rets = df.loc[df["signal"] < config.SELL_THRESHOLD, col]
    hold_rets = df.loc[
        (df["signal"] >= config.SELL_THRESHOLD) & (df["signal"] <= config.BUY_THRESHOLD), col
    ]

    if len(buy_rets) < 3 and len(sell_rets) < 3:
        logger.warning("Too few BUY/SELL signals for return distribution plot")
        return

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PLOTS_DIR / f"return_distributions_{model_name}_lag{lag}.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    for rets, label, colour in [
        (buy_rets, "BUY signal", "green"),
        (hold_rets, "HOLD signal", "gray"),
        (sell_rets, "SELL signal", "red"),
    ]:
        if len(rets) >= 3:
            rets.plot.kde(ax=ax, label=f"{label} (n={len(rets)})", color=colour, linewidth=2)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"{lag}-day forward return")
    ax.set_ylabel("Density")
    ax.set_title(f"Return distributions by signal — {model_name}  (lag={lag}d)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Return distribution plot saved → %s", out)


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------


def print_validation_report(results: dict) -> None:
    """Print a structured validation summary to the console.

    Args:
        results: Dict returned by :func:`run_signal_validation`.
    """
    # Use ASCII to avoid encoding issues
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  AlphaLens Signal Validation Report — {results['model_name']}")
    print(sep)

    # Hit rate
    hr = results.get("hit_rate", {})
    print(f"\n  [1] Directional Accuracy (lag={hr.get('lag', 1)}d)")
    print(f"      Overall hit rate : {hr.get('overall', 0):.1%}  (random = 50%)")
    print(f"      BUY  signals     : {hr.get('buy', 0):.1%}  (n={hr.get('n_buy', 0)})")
    print(f"      SELL signals     : {hr.get('sell', 0):.1%}  (n={hr.get('n_sell', 0)})")
    above = hr.get("overall", 0) >= 0.55
    print(f"      Verdict          : {'PREDICTIVE (>55%)' if above else 'WEAK (<55%)'}")

    # IC
    ic_df = results.get("ic_decay")
    if ic_df is not None and not ic_df.empty:
        ic1 = ic_df[ic_df["lag"] == 1].iloc[0] if 1 in ic_df["lag"].values else ic_df.iloc[0]
        print(f"\n  [2] Information Coefficient  (lag=1d)")
        print(f"      IC               : {ic1['ic']:+.4f}")
        print(f"      p-value          : {ic1['p_value']:.4f}")
        sig = ic1["significant"]
        print(f"      Verdict          : {'SIGNIFICANT (p<0.05)' if sig else 'NOT significant'}")
        print(f"      Interpretation   : IC>0.05 is considered useful in practice")

    # t-test
    tt = results.get("ttest", {})
    buy_tt = tt.get("buy", {})
    sell_tt = tt.get("sell", {})
    print(f"\n  [3] Conditional Return t-test  (lag={tt.get('lag', 1)}d)")
    print(f"      BUY  mean return  : {buy_tt.get('mean_return', 0):+.4f}"
          f"  t={buy_tt.get('t_stat', 0):+.2f}"
          f"  {'SIG *' if buy_tt.get('significant') else 'n.s.'}")
    print(f"      SELL mean return  : {sell_tt.get('mean_return', 0):+.4f}"
          f"  t={sell_tt.get('t_stat', 0):+.2f}"
          f"  {'SIG *' if sell_tt.get('significant') else 'n.s.'}")

    # Quantile spread
    q_df = results.get("quantile_returns")
    if q_df is not None and not q_df.empty and len(q_df) >= 2:
        spread = float(q_df["mean_return"].iloc[-1] - q_df["mean_return"].iloc[0])
        print(f"\n  [4] Quantile Spread  (Q{len(q_df)} - Q1 return, lag=1d)")
        print(f"      Q1  (low signal) : {q_df['mean_return'].iloc[0]:+.4f}")
        print(f"      Q{len(q_df)}  (high signal): {q_df['mean_return'].iloc[-1]:+.4f}")
        print(f"      Spread           : {spread:+.4f}")
        print(f"      Verdict          : {'POSITIVE spread (good)' if spread > 0 else 'NEGATIVE spread (reversed!)'}")

    # IC by ticker
    ic_tick = results.get("ic_by_ticker")
    if ic_tick is not None and not ic_tick.empty:
        print(f"\n  [5] IC by Ticker  (lag=1d)")
        for _, row in ic_tick.head(5).iterrows():
            sig_mark = "*" if row["significant"] else " "
            print(f"      {row['ticker']:<6} IC={row['ic']:+.4f}  p={row['p_value']:.3f} {sig_mark}")

    # Overall verdict
    print(f"\n  {'=' * 53}")
    n_green = sum([
        hr.get("overall", 0) >= 0.55,
        (ic_df is not None and not ic_df.empty and
         float(ic_df[ic_df["lag"] == 1]["ic"].values[0]) > 0.02
         if len(ic_df[ic_df["lag"] == 1]) > 0 else False),
        buy_tt.get("significant", False),
    ])
    if n_green >= 2:
        verdict = "SIGNAL HAS PREDICTIVE POWER  -- consider live testing"
    elif n_green == 1:
        verdict = "WEAK SIGNAL -- more data or model improvement needed"
    else:
        verdict = "NO SIGNIFICANT PREDICTIVE POWER detected"
    print(f"  Overall verdict: {verdict}")
    print(f"  {'=' * 53}\n")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_validation_metrics(results: dict) -> None:
    """Save numerical validation results to CSV files.

    Args:
        results: Dict returned by :func:`run_signal_validation`.
    """
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = results.get("model_name", "model")

    # Hit rate
    hr = results.get("hit_rate", {})
    if hr:
        pd.DataFrame([hr]).to_csv(
            config.METRICS_DIR / f"hit_rate_{model_name}.csv", index=False
        )

    # IC decay
    ic_df = results.get("ic_decay")
    if ic_df is not None and not ic_df.empty:
        ic_df.to_csv(
            config.METRICS_DIR / f"ic_decay_{model_name}.csv", index=False
        )

    # Quantile returns
    q_df = results.get("quantile_returns")
    if q_df is not None and not q_df.empty:
        q_df.to_csv(
            config.METRICS_DIR / f"quantile_returns_{model_name}.csv", index=False
        )

    # IC by ticker
    ic_tick = results.get("ic_by_ticker")
    if ic_tick is not None and not ic_tick.empty:
        ic_tick.to_csv(
            config.METRICS_DIR / f"ic_by_ticker_{model_name}.csv", index=False
        )

    logger.info("Validation metrics saved to %s", config.METRICS_DIR)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_signal_validation(
    model,
    model_name: str = "model",
    news_df: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> dict:
    """Run the full signal validation battery.

    Loads ``combined_news.csv`` and fetches prices if not provided,
    then runs all six validation tests and produces plots and CSVs.

    Args:
        model: Trained sentiment model with a ``predict`` interface.
        model_name: Short identifier used in filenames and log messages.
        news_df: Optional pre-loaded news DataFrame.  Loaded from
            ``config.PROCESSED_DATA_DIR / "combined_news.csv"`` if ``None``.
        prices: Optional pre-loaded price DataFrame.  Fetched via
            yfinance if ``None``.

    Returns:
        dict: All validation results keyed by test name, plus
        ``"model_name"`` and ``"sr_df"`` (the raw signal-return table).
    """
    from src.backtest import fetch_price_data

    logger.info("=== AlphaLens Signal Validation [%s] ===", model_name)

    # Load news
    if news_df is None:
        news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
        if not news_path.exists():
            raise FileNotFoundError(
                f"combined_news.csv not found at {news_path}. "
                "Run src/data_sources.py first."
            )
        news_df = pd.read_csv(news_path, parse_dates=["date"])
        logger.info("Loaded %d headlines", len(news_df))

    # Fetch prices
    if prices is None:
        start = pd.to_datetime(news_df["date"]).min().strftime("%Y-%m-%d")
        end = (pd.to_datetime(news_df["date"]).max() + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
        prices = fetch_price_data(config.TICKERS, start=start, end=end)

    if prices.empty:
        logger.error("No price data — aborting validation")
        return {"model_name": model_name}

    # Build paired dataset
    sr_df = build_signal_return_df(news_df, prices, model, max_lag=max(_LAGS))

    if sr_df.empty:
        logger.error("No signal-return pairs built — aborting")
        return {"model_name": model_name, "sr_df": sr_df}

    # Run all tests
    hit_rate = compute_hit_rate(sr_df, lag=1)
    ic_decay = compute_ic_decay(sr_df, lags=_LAGS)
    ttest = compute_conditional_ttest(sr_df, lag=1)
    quantile_returns = compute_quantile_returns(sr_df, lag=1)
    calibration = compute_calibration(sr_df, lag=1)
    ic_by_ticker = compute_ic_by_ticker(sr_df, lag=1)

    results = {
        "model_name": model_name,
        "sr_df": sr_df,
        "hit_rate": hit_rate,
        "ic_decay": ic_decay,
        "ttest": ttest,
        "quantile_returns": quantile_returns,
        "calibration": calibration,
        "ic_by_ticker": ic_by_ticker,
    }

    # Plots
    plot_ic_decay(ic_decay, model_name)
    plot_quantile_returns(quantile_returns, model_name, lag=1)
    plot_calibration(calibration, model_name)
    plot_hit_rate_by_ticker(ic_by_ticker, model_name)
    plot_return_distributions(sr_df, model_name, lag=1)

    # Save metrics
    save_validation_metrics(results)

    # Console report
    print_validation_report(results)

    logger.info("=== Signal validation complete [%s] ===", model_name)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.model import VADERBaseline

    vader = VADERBaseline()
    run_signal_validation(vader, model_name="vader")
