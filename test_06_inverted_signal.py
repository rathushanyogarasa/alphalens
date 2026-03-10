"""test_06_inverted_signal.py
==============================
Tests whether inverting the sentiment signal (positive -> SELL, negative -> BUY)
improves predictive performance.

Hypothesis: if the model's positive-sentiment label consistently precedes
price declines (mean-reversion after good news) and negative-sentiment
precedes recoveries, IC will turn positive under inversion.

What this script does
---------------------
1. Runs model inference once and caches predictions.
2. Builds NORMAL signal-return pairs and INVERTED signal-return pairs.
3. For both variants, computes:
   - IC at lags 1, 2, 3, 5, 10 days
   - Directional hit rate at each lag
   - Sharpe, Calmar, win rate (with 10+5 bps cost)
   - Quantile spread (Q5 - Q1 mean return)
4. Prints a side-by-side comparison table.
5. Saves full results to results/metrics/ablation_inversion_*.csv

Usage
-----
    python test_06_inverted_signal.py [--vader]

    --vader   Use VADER baseline (faster).  Default: fine-tuned FinBERT.
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inversion")
logger.setLevel(logging.INFO)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.backtest import fetch_price_data, calculate_portfolio_metrics
from src.model import SIGNAL_MAP

# Inverted map: positive sentiment -> short signal, negative -> long
SIGNAL_MAP_INVERTED: dict[str, int] = {
    "negative":  1,   # flipped
    "neutral":   0,
    "positive": -1,   # flipped
    "uncertain": 0,
}


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def load_model(force_vader: bool = False):
    if force_vader:
        from src.model import VADERBaseline
        logger.info("Using VADERBaseline (forced via --vader)")
        return VADERBaseline(), "vader"

    checkpoint = config.MODEL_DIR / "weights.pt"
    if checkpoint.exists():
        try:
            import torch
            from src.model import FinBERTClassifier
            model = FinBERTClassifier()
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            model.eval()
            logger.info("Loaded fine-tuned FinBERT from %s", checkpoint)
            return model, "finbert"
        except Exception as exc:
            logger.warning("FinBERT load failed (%s) — falling back to VADER", exc)

    from src.model import VADERBaseline
    logger.info("Using VADERBaseline (no checkpoint found)")
    return VADERBaseline(), "vader"


# ---------------------------------------------------------------------------
# Inference (single pass)
# ---------------------------------------------------------------------------


def load_and_predict(model) -> pd.DataFrame:
    news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
    if not news_path.exists():
        raise FileNotFoundError(
            f"combined_news.csv not found at {news_path}. "
            "Run src/data_sources.py first."
        )
    news_df = pd.read_csv(news_path, parse_dates=["date"])
    logger.info("Loaded %d headlines — running inference ...", len(news_df))

    texts = news_df["headline"].tolist()
    predictions: list[dict] = []
    for i in range(0, len(texts), config.BATCH_SIZE):
        predictions.extend(model.predict(texts[i : i + config.BATCH_SIZE]))

    news_df = news_df.copy().reset_index(drop=True)
    news_df["label_name"]    = [p["label_name"] for p in predictions]
    news_df["confidence"]    = [p["confidence"]  for p in predictions]
    news_df["date"]          = pd.to_datetime(news_df["date"]).dt.normalize()
    # Both signal columns derived here; inversion applied below
    news_df["signal_normal"] = news_df["label_name"].map(SIGNAL_MAP).fillna(0)
    news_df["signal_inv"]    = news_df["label_name"].map(SIGNAL_MAP_INVERTED).fillna(0)

    dist = news_df["label_name"].value_counts().to_dict()
    logger.info("Label distribution: %s", dist)
    return news_df


# ---------------------------------------------------------------------------
# Signal aggregation
# ---------------------------------------------------------------------------


def aggregate(
    pred_df: pd.DataFrame,
    signal_col: str,
    confidence_threshold: float,
) -> pd.DataFrame:
    """Filter by confidence and group to (date, ticker) mean signal."""
    confident = pred_df[pred_df["confidence"] >= confidence_threshold].copy()
    if confident.empty:
        return pd.DataFrame()
    return (
        confident.groupby(["date", "ticker"])
        .agg(
            signal=(signal_col, "mean"),
            avg_confidence=("confidence", "mean"),
            headline_count=(signal_col, "count"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_ic(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    lag: int,
) -> tuple[float, float, int]:
    """Spearman IC between aggregated signal and lag-day forward return."""
    price_sorted = prices.sort_index()
    rows: list[dict] = []
    for _, row in signals.iterrows():
        date   = pd.Timestamp(row["date"])
        ticker = str(row["ticker"])
        if ticker not in price_sorted.columns:
            continue
        pos = price_sorted.index.searchsorted(date)
        if pos + lag >= len(price_sorted):
            continue
        p0    = price_sorted[ticker].iloc[pos]
        p_lag = price_sorted[ticker].iloc[pos + lag]
        if pd.isna(p0) or pd.isna(p_lag) or p0 == 0:
            continue
        rows.append({"signal": float(row["signal"]), "fwd": float((p_lag - p0) / p0)})
    if len(rows) < 10:
        return np.nan, np.nan, len(rows)
    df = pd.DataFrame(rows)
    ic, p = stats.spearmanr(df["signal"], df["fwd"])
    return float(ic), float(p), len(rows)


def compute_hit_rate(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    lag: int,
    buy_thresh: float,
    sell_thresh: float,
) -> tuple[float, int]:
    """Directional accuracy (active positions only) at *lag* days."""
    price_sorted = prices.sort_index()
    correct = total = 0
    for _, row in signals.iterrows():
        signal   = float(row["signal"])
        position = 1 if signal > buy_thresh else (-1 if signal < sell_thresh else 0)
        if position == 0:
            continue
        date   = pd.Timestamp(row["date"])
        ticker = str(row["ticker"])
        if ticker not in price_sorted.columns:
            continue
        pos = price_sorted.index.searchsorted(date)
        if pos + lag >= len(price_sorted):
            continue
        p0    = price_sorted[ticker].iloc[pos]
        p_lag = price_sorted[ticker].iloc[pos + lag]
        if pd.isna(p0) or pd.isna(p_lag) or p0 == 0:
            continue
        fwd = (p_lag - p0) / p0
        total += 1
        if (position > 0 and fwd > 0) or (position < 0 and fwd < 0):
            correct += 1
    return (float(correct / total) if total > 0 else np.nan), total


def compute_quantile_spread(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    lag: int = 1,
    n_quantiles: int = 5,
) -> float:
    """Q5 - Q1 mean forward return (higher = better signal monotonicity)."""
    price_sorted = prices.sort_index()
    rows: list[dict] = []
    for _, row in signals.iterrows():
        date   = pd.Timestamp(row["date"])
        ticker = str(row["ticker"])
        if ticker not in price_sorted.columns:
            continue
        pos = price_sorted.index.searchsorted(date)
        if pos + lag >= len(price_sorted):
            continue
        p0    = price_sorted[ticker].iloc[pos]
        p_lag = price_sorted[ticker].iloc[pos + lag]
        if pd.isna(p0) or pd.isna(p_lag) or p0 == 0:
            continue
        rows.append({"signal": float(row["signal"]), "fwd": float((p_lag - p0) / p0)})
    if len(rows) < n_quantiles * 4:
        return np.nan
    df = pd.DataFrame(rows)
    df["q"] = pd.qcut(df["signal"], q=n_quantiles, labels=False, duplicates="drop")
    qret = df.groupby("q")["fwd"].mean()
    if len(qret) < 2:
        return np.nan
    return float(qret.iloc[-1] - qret.iloc[0])


def signals_to_returns(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    buy_thresh: float,
    sell_thresh: float,
    tc_bps: float = 10.0,
    slip_bps: float = 5.0,
) -> pd.DataFrame:
    if signals.empty or prices.empty:
        return pd.DataFrame()
    daily_ret = prices.pct_change().shift(-1)
    rows: list[dict] = []
    for _, row in signals.iterrows():
        date   = pd.Timestamp(row["date"])
        ticker = str(row["ticker"])
        signal = float(row["signal"])
        if ticker not in daily_ret.columns or date not in daily_ret.index:
            continue
        next_ret = daily_ret.loc[date, ticker]
        if pd.isna(next_ret):
            continue
        position  = 1 if signal > buy_thresh else (-1 if signal < sell_thresh else 0)
        cost_drag = abs(position) * (tc_bps + slip_bps) / 10_000.0
        rows.append({
            "date":             date,
            "ticker":           ticker,
            "signal":           signal,
            "position":         position,
            "strategy_return":  position * float(next_ret) - cost_drag,
            "benchmark_return": float(next_ret),
            "cost_drag":        cost_drag,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def safe_fmt(val, fmt=".4f", na="  n/a  "):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    return format(val, fmt)


# ---------------------------------------------------------------------------
# Full evaluation for one variant
# ---------------------------------------------------------------------------

LAGS = [1, 2, 3, 5, 10]


def evaluate_variant(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    buy_thresh: float,
    sell_thresh: float,
    label: str,
) -> dict:
    """Compute all metrics for one signal variant. Returns a result dict."""
    ic_rows: list[dict] = []
    for lag in LAGS:
        ic, p, n = compute_ic(signals, prices, lag)
        hr, n_act = compute_hit_rate(signals, prices, lag, buy_thresh, sell_thresh)
        ic_rows.append({"lag": lag, "ic": ic, "p": p, "n": n, "hit_rate": hr, "n_active": n_act})
    ic_df = pd.DataFrame(ic_rows)

    q_spread = compute_quantile_spread(signals, prices, lag=1)

    returns_df = signals_to_returns(signals, prices, buy_thresh, sell_thresh)
    n_active   = int((returns_df["position"] != 0).sum()) if not returns_df.empty else 0
    metrics    = (
        calculate_portfolio_metrics(returns_df, label)
        if not returns_df.empty and n_active > 0
        else {}
    )

    return {
        "label":     label,
        "n_signals": len(signals),
        "ic_df":     ic_df,
        "q_spread":  q_spread,
        "n_active":  n_active,
        "sharpe":    metrics.get("sharpe_ratio", np.nan),
        "calmar":    metrics.get("calmar_ratio", np.nan),
        "win_rate":  metrics.get("win_rate", np.nan),
        "ann_ret":   metrics.get("annualised_return", np.nan),
        "max_dd":    metrics.get("max_drawdown", np.nan),
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def print_ic_table(normal: dict, inverted: dict) -> None:
    print("\n" + "=" * 78)
    print("  IC & Hit Rate by Lag — Normal vs Inverted Signal")
    print("=" * 78)
    header = (
        f"  {'Lag':>5}  "
        f"{'IC (norm)':>10} {'hit (norm)':>11}  "
        f"{'IC (inv)':>10} {'hit (inv)':>11}  "
        f"{'IC delta':>10}"
    )
    print(header)
    print("  " + "-" * 75)

    ic_n = normal["ic_df"].set_index("lag")
    ic_i = inverted["ic_df"].set_index("lag")

    for lag in LAGS:
        row_n = ic_n.loc[lag] if lag in ic_n.index else {}
        row_i = ic_i.loc[lag] if lag in ic_i.index else {}
        ic_n_val  = row_n.get("ic", np.nan)
        ic_i_val  = row_i.get("ic", np.nan)
        hr_n_val  = row_n.get("hit_rate", np.nan)
        hr_i_val  = row_i.get("hit_rate", np.nan)
        delta     = (ic_i_val - ic_n_val) if not (np.isnan(ic_n_val) or np.isnan(ic_i_val)) else np.nan
        sig_n = "*" if (not np.isnan(row_n.get("p", np.nan)) and row_n.get("p", 1) < 0.05) else " "
        sig_i = "*" if (not np.isnan(row_i.get("p", np.nan)) and row_i.get("p", 1) < 0.05) else " "

        print(
            f"  {lag:>4}d  "
            f"{safe_fmt(ic_n_val, '+.4f'):>10}{sig_n} {safe_fmt(hr_n_val, '.3f'):>11}  "
            f"{safe_fmt(ic_i_val, '+.4f'):>10}{sig_i} {safe_fmt(hr_i_val, '.3f'):>11}  "
            f"{safe_fmt(delta, '+.4f'):>10}"
        )
    print("  (* = p < 0.05)")


def print_backtest_table(normal: dict, inverted: dict) -> None:
    print("\n" + "=" * 60)
    print("  Backtest Metrics — Normal vs Inverted (10+5 bps cost)")
    print("=" * 60)
    metrics = [
        ("n_active signals", "n_active", "d"),
        ("Quantile spread (Q5-Q1)", "q_spread", "+.4f"),
        ("Annualised return", "ann_ret", "+.2%"),
        ("Sharpe ratio", "sharpe", "+.3f"),
        ("Calmar ratio", "calmar", "+.3f"),
        ("Win rate", "win_rate", ".1%"),
        ("Max drawdown", "max_dd", ".2%"),
    ]
    print(f"  {'Metric':<28} {'Normal':>12} {'Inverted':>12}")
    print("  " + "-" * 57)
    for name, key, fmt in metrics:
        v_n = normal.get(key, np.nan)
        v_i = inverted.get(key, np.nan)
        s_n = format(v_n, fmt) if not (isinstance(v_n, float) and np.isnan(v_n)) else "n/a"
        s_i = format(v_i, fmt) if not (isinstance(v_i, float) and np.isnan(v_i)) else "n/a"
        print(f"  {name:<28} {s_n:>12} {s_i:>12}")


def print_verdict(normal: dict, inverted: dict) -> None:
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)

    ic_n_vals  = normal["ic_df"]["ic"].dropna()
    ic_i_vals  = inverted["ic_df"]["ic"].dropna()
    mean_ic_n  = float(ic_n_vals.mean()) if len(ic_n_vals) > 0 else np.nan
    mean_ic_i  = float(ic_i_vals.mean()) if len(ic_i_vals) > 0 else np.nan
    n_pos_n    = int((ic_n_vals > 0).sum())
    n_pos_i    = int((ic_i_vals > 0).sum())

    print(f"\n  Normal   — mean IC across lags: {safe_fmt(mean_ic_n, '+.4f')}  ({n_pos_n}/{len(LAGS)} lags positive)")
    print(f"  Inverted — mean IC across lags: {safe_fmt(mean_ic_i, '+.4f')}  ({n_pos_i}/{len(LAGS)} lags positive)")

    sharpe_n = normal.get("sharpe", np.nan)
    sharpe_i = inverted.get("sharpe", np.nan)
    print(f"\n  Normal Sharpe  : {safe_fmt(sharpe_n, '+.3f')}")
    print(f"  Inverted Sharpe: {safe_fmt(sharpe_i, '+.3f')}")

    print()
    inv_better_ic     = (not np.isnan(mean_ic_i)) and (not np.isnan(mean_ic_n)) and mean_ic_i > mean_ic_n
    inv_better_sharpe = (not np.isnan(sharpe_i)) and (not np.isnan(sharpe_n)) and sharpe_i > sharpe_n
    inv_positive_ic   = (not np.isnan(mean_ic_i)) and mean_ic_i > 0

    if inv_better_ic and inv_positive_ic and inv_better_sharpe:
        print("  CONFIRMED: Inversion improves both IC and Sharpe.")
        print("  The model is a CONTRARIAN indicator — positive FinBERT sentiment")
        print("  precedes price declines (sell the news); negative precedes recovery.")
        print()
        print("  Recommended action:")
        print("    1. Flip SIGNAL_MAP in config/stock_engine to use inverted signals.")
        print("    2. Re-run the trust scorer with inverted signals to verify IC > 0.")
        print("    3. Set BUY_THRESHOLD to the value that gives best hit rate above 50%.")
    elif inv_better_ic and not inv_positive_ic:
        print("  PARTIAL: Inversion improves IC but it remains negative.")
        print("  The signal has low predictive power in either direction.")
        print("  Consider: gathering more headline data, different news sources,")
        print("  or moving to a cross-sectional (ranking-based) strategy.")
    elif not inv_better_ic:
        print("  NO BENEFIT: Normal signal direction is at least as good as inverted.")
        print("  The issue is not label polarity — look elsewhere:")
        print("    - Data staleness: are headlines arriving after price moves?")
        print("    - Universe: too few tickers or too few actionable headlines.")
        print("    - Feature gap: try adding macro/sector context to the signal.")
    print()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_results(normal: dict, inverted: dict) -> None:
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    for variant in [normal, inverted]:
        tag = variant["label"].replace(" ", "_")
        ic_path = config.METRICS_DIR / f"ablation_inversion_{tag}_ic.csv"
        variant["ic_df"].to_csv(ic_path, index=False)

    # Summary row comparison
    summary = []
    for v in [normal, inverted]:
        ic_df   = v["ic_df"]
        mean_ic = float(ic_df["ic"].dropna().mean()) if ic_df["ic"].notna().any() else np.nan
        summary.append({
            "variant":    v["label"],
            "n_signals":  v["n_signals"],
            "n_active":   v["n_active"],
            "mean_ic":    round(mean_ic, 4) if not np.isnan(mean_ic) else np.nan,
            "ic_lag1":    ic_df.loc[ic_df["lag"] == 1, "ic"].values[0] if 1 in ic_df["lag"].values else np.nan,
            "hit_lag1":   ic_df.loc[ic_df["lag"] == 1, "hit_rate"].values[0] if 1 in ic_df["lag"].values else np.nan,
            "q_spread":   v["q_spread"],
            "sharpe":     v["sharpe"],
            "calmar":     v["calmar"],
            "win_rate":   v["win_rate"],
            "ann_ret":    v["ann_ret"],
            "max_dd":     v["max_dd"],
        })
    out = config.METRICS_DIR / "ablation_inversion_summary.csv"
    pd.DataFrame(summary).to_csv(out, index=False)
    logger.info("Results saved -> %s", config.METRICS_DIR)
    print(f"  Results saved -> {config.METRICS_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    force_vader = "--vader" in sys.argv
    model, model_name = load_model(force_vader)

    print(f"\n  AlphaLens — Inverted Signal Test  [{model_name.upper()}]")
    print(f"  Processed data dir: {config.PROCESSED_DATA_DIR}")

    pred_df = load_and_predict(model)

    start  = pred_df["date"].min().strftime("%Y-%m-%d")
    end    = (pred_df["date"].max() + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    prices = fetch_price_data(config.TICKERS, start=start, end=end)

    if prices.empty:
        print("\n  ERROR: Could not fetch price data. Aborting.")
        sys.exit(1)

    buy, sell, conf = config.BUY_THRESHOLD, config.SELL_THRESHOLD, config.CONFIDENCE_THRESHOLD

    print(f"\n  Using thresholds: buy={buy}, sell={sell}, conf={conf}")
    print("  Running normal signal ...")
    sig_normal   = aggregate(pred_df, "signal_normal", conf)
    result_normal = evaluate_variant(sig_normal, prices, buy, sell, label="normal")

    print("  Running inverted signal ...")
    sig_inverted   = aggregate(pred_df, "signal_inv", conf)
    result_inverted = evaluate_variant(sig_inverted, prices, buy, sell, label="inverted")

    print_ic_table(result_normal, result_inverted)
    print_backtest_table(result_normal, result_inverted)
    print_verdict(result_normal, result_inverted)
    save_results(result_normal, result_inverted)


if __name__ == "__main__":
    main()
