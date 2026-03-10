"""test_05_threshold_ablation.py
=================================
Ablation study for the AlphaLens signal pipeline.

Three independent experiments are run sequentially using a *single* model
inference pass (predictions are cached):

1. **Threshold ablation** — vary BUY/SELL threshold and confidence threshold.
2. **Horizon ablation**   — test IC and directional hit rate at lags 1–10 days.
3. **Keyword ablation**   — compare pure FinBERT signals vs keyword-CAR-adjusted
                            signals (using keyword_summary.csv).

Results are printed as formatted tables and saved to results/metrics/ablation_*.csv.

Usage
-----
    python test_05_threshold_ablation.py [--vader]

Flags
-----
    --vader   Force use of the VADER baseline (faster, no GPU needed).
              Default: loads the fine-tuned FinBERT checkpoint.
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
logger = logging.getLogger("ablation")
logger.setLevel(logging.INFO)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.backtest import fetch_price_data, calculate_portfolio_metrics
from src.model import SIGNAL_MAP


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def load_model(force_vader: bool = False):
    """Load fine-tuned FinBERT (default) or VADER baseline."""
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
# Step 1: run model inference once, cache results
# ---------------------------------------------------------------------------


def load_and_predict(model) -> pd.DataFrame:
    """Load combined_news.csv, run model, return prediction-enriched DataFrame."""
    news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
    if not news_path.exists():
        raise FileNotFoundError(
            f"combined_news.csv not found at {news_path}.\n"
            "Run src/data_sources.py (or main.py) first."
        )
    news_df = pd.read_csv(news_path, parse_dates=["date"])
    logger.info("Loaded %d headlines — running model inference …", len(news_df))

    texts = news_df["headline"].tolist()
    predictions: list[dict] = []
    for i in range(0, len(texts), config.BATCH_SIZE):
        predictions.extend(model.predict(texts[i : i + config.BATCH_SIZE]))

    news_df = news_df.copy().reset_index(drop=True)
    news_df["label_name"] = [p["label_name"] for p in predictions]
    news_df["confidence"]  = [p["confidence"]  for p in predictions]
    news_df["signal_raw"]  = news_df["label_name"].map(SIGNAL_MAP).fillna(0)
    news_df["date"]        = pd.to_datetime(news_df["date"]).dt.normalize()

    logger.info(
        "Inference complete. Label distribution: %s",
        news_df["label_name"].value_counts().to_dict(),
    )
    return news_df


# ---------------------------------------------------------------------------
# Helpers: signal aggregation and metrics
# ---------------------------------------------------------------------------


def aggregate_signals(
    pred_df: pd.DataFrame,
    confidence_threshold: float,
    signal_col: str = "signal_raw",
) -> pd.DataFrame:
    """Filter by confidence and aggregate to (date, ticker) level."""
    confident = pred_df[pred_df["confidence"] >= confidence_threshold].copy()
    if confident.empty:
        return pd.DataFrame(columns=["date", "ticker", "signal", "avg_confidence", "headline_count"])
    grouped = (
        confident.groupby(["date", "ticker"])
        .agg(
            signal=(signal_col, "mean"),
            avg_confidence=("confidence", "mean"),
            headline_count=(signal_col, "count"),
        )
        .reset_index()
    )
    return grouped


def signals_to_returns(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    tc_bps: float = 10.0,
    slip_bps: float = 5.0,
) -> pd.DataFrame:
    """Convert aggregated signals to daily strategy returns (next-day execution)."""
    if signals.empty or prices.empty:
        return pd.DataFrame()
    daily_returns = prices.pct_change().shift(-1)
    rows: list[dict] = []
    for _, row in signals.iterrows():
        date   = pd.Timestamp(row["date"])
        ticker = str(row["ticker"])
        signal = float(row["signal"])
        if ticker not in daily_returns.columns or date not in daily_returns.index:
            continue
        next_ret = daily_returns.loc[date, ticker]
        if pd.isna(next_ret):
            continue
        position = 1 if signal > buy_threshold else (-1 if signal < sell_threshold else 0)
        total_cost = (tc_bps + slip_bps) / 10_000.0
        cost_drag  = abs(position) * total_cost
        rows.append({
            "date":             date,
            "ticker":           ticker,
            "signal":           signal,
            "position":         position,
            "strategy_return":  position * float(next_ret) - cost_drag,
            "gross_return":     position * float(next_ret),
            "benchmark_return": float(next_ret),
            "cost_drag":        cost_drag,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def compute_ic(signals: pd.DataFrame, prices: pd.DataFrame, lag: int = 1) -> tuple[float, float, int]:
    """Spearman IC between aggregated signal and lag-day forward return.

    Returns (ic, p_value, n).
    """
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
        rows.append({"signal": float(row["signal"]), "fwd_ret": float((p_lag - p0) / p0)})
    if len(rows) < 10:
        return np.nan, np.nan, len(rows)
    df  = pd.DataFrame(rows)
    ic, p = stats.spearmanr(df["signal"], df["fwd_ret"])
    return float(ic), float(p), len(rows)


def compute_hit_rate(signals: pd.DataFrame, prices: pd.DataFrame, lag: int, buy_thresh: float, sell_thresh: float) -> float:
    """Fraction of active positions where direction matched forward return at *lag* days."""
    price_sorted = prices.sort_index()
    correct_total = total = 0
    for _, row in signals.iterrows():
        signal = float(row["signal"])
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
        fwd_ret = (p_lag - p0) / p0
        total  += 1
        if (position > 0 and fwd_ret > 0) or (position < 0 and fwd_ret < 0):
            correct_total += 1
    return float(correct_total / total) if total > 0 else np.nan


def safe_fmt(val, fmt=".4f", na="  n/a  "):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    return format(val, fmt)


# ---------------------------------------------------------------------------
# Ablation 1: Threshold grid
# ---------------------------------------------------------------------------

THRESHOLD_CONFIGS = [
    # label,                    conf,  buy,  sell
    ("base  (buy=0.45 / conf=0.70)",  0.70, 0.45, -0.45),
    ("loose (buy=0.30 / conf=0.70)",  0.70, 0.30, -0.30),
    ("tight (buy=0.60 / conf=0.70)",  0.70, 0.60, -0.60),
    ("xtight(buy=0.70 / conf=0.70)",  0.70, 0.70, -0.70),
    ("loconf(buy=0.45 / conf=0.55)",  0.55, 0.45, -0.45),
    ("hiconf(buy=0.45 / conf=0.80)",  0.80, 0.45, -0.45),
    ("hiconf(buy=0.60 / conf=0.80)",  0.80, 0.60, -0.60),
]


def run_threshold_ablation(pred_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 75)
    print("  ABLATION 1 — Threshold Grid")
    print("  (IC lag=1d, hit rate lag=1d, Sharpe with 10+5 bps cost)")
    print("=" * 75)
    header = f"  {'Config':<40} {'n_sig':>6} {'n_act':>6} {'IC':>8} {'hit':>7} {'Sharpe':>8}"
    print(header)
    print("  " + "-" * 73)

    rows: list[dict] = []
    for label, conf, buy, sell in THRESHOLD_CONFIGS:
        signals = aggregate_signals(pred_df, conf)
        if signals.empty:
            print(f"  {label:<40} {'0':>6} {'—':>6} {'—':>8} {'—':>7} {'—':>8}")
            continue

        # Apply buy/sell threshold for trades (IC uses raw continuous signal)
        ic, p, n_ic = compute_ic(signals, prices, lag=1)
        hr           = compute_hit_rate(signals, prices, lag=1, buy_thresh=buy, sell_thresh=sell)
        returns_df   = signals_to_returns(signals, prices, buy, sell)
        n_active     = int((returns_df["position"] != 0).sum()) if not returns_df.empty else 0
        metrics      = calculate_portfolio_metrics(returns_df, label) if not returns_df.empty and n_active > 0 else {}
        sharpe       = metrics.get("sharpe_ratio", np.nan)

        print(
            f"  {label:<40} {len(signals):>6} {n_active:>6} "
            f"{safe_fmt(ic, '+.4f'):>8} {safe_fmt(hr, '.3f'):>7} {safe_fmt(sharpe, '+.3f'):>8}"
        )
        rows.append({
            "config": label, "conf_thresh": conf, "buy_thresh": buy, "sell_thresh": sell,
            "n_signals": len(signals), "n_active": n_active,
            "ic_1d": ic, "ic_p": p, "hit_rate_1d": hr, "sharpe": sharpe,
        })

    df = pd.DataFrame(rows)
    out = config.METRICS_DIR / "ablation_threshold.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")
    return df


# ---------------------------------------------------------------------------
# Ablation 2: Horizon (IC + hit rate at multiple lags)
# ---------------------------------------------------------------------------


def run_horizon_ablation(pred_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    buy, sell, conf = 0.45, -0.45, 0.70
    signals = aggregate_signals(pred_df, conf)

    print("\n" + "=" * 65)
    print("  ABLATION 2 — Horizon (buy=0.45, conf=0.70)")
    print("=" * 65)
    header = f"  {'Lag':>6} {'n':>6} {'IC':>9} {'p-val':>8} {'sig':>5} {'hit_rate':>10}"
    print(header)
    print("  " + "-" * 63)

    rows: list[dict] = []
    for lag in [1, 2, 3, 5, 10]:
        ic, p, n = compute_ic(signals, prices, lag=lag)
        hr        = compute_hit_rate(signals, prices, lag=lag, buy_thresh=buy, sell_thresh=sell)
        sig       = "YES" if (not np.isnan(p) and p < 0.05) else "no"
        print(
            f"  {lag:>5}d {n:>6} {safe_fmt(ic, '+.4f'):>9} "
            f"{safe_fmt(p, '.4f'):>8} {sig:>5} {safe_fmt(hr, '.3f'):>10}"
        )
        rows.append({"lag_days": lag, "n": n, "ic": ic, "p_value": p, "significant": sig == "YES", "hit_rate": hr})

    df = pd.DataFrame(rows)
    out = config.METRICS_DIR / "ablation_horizon.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")

    # Interpretation
    best_lag = df.dropna(subset=["ic"]).loc[df["ic"].abs().idxmax(), "lag_days"] if not df.dropna(subset=["ic"]).empty else "?"
    print(f"\n  Best IC at lag = {best_lag} day(s)")
    any_positive = df["ic"].dropna()
    if any_positive.gt(0).any():
        print("  IC turns positive at some horizon — signal may suit a longer hold.")
    else:
        print("  IC negative at all horizons — signal direction is persistently reversed.")
    return df


# ---------------------------------------------------------------------------
# Ablation 3: Keyword CAR weighting
# ---------------------------------------------------------------------------

# Hard-filter patterns for SEC filing artefacts (same logic as stock_engine.py)
_FILING_ARTEFACTS = {
    "cik", "filing", "item", "form", "exhibit", "amendment",
    "sec", "edgar", "registrant", "pursuant",
}


def _is_filing_artefact(kw: str) -> bool:
    tokens = set(kw.lower().split())
    return bool(tokens & _FILING_ARTEFACTS)


def build_keyword_adjusted_signals(pred_df: pd.DataFrame, min_car: float = 0.003) -> pd.DataFrame:
    """Adjust signal_raw by keyword CAR lookup, return enriched pred_df."""
    kw_path = config.METRICS_DIR / "keyword_summary.csv"
    if not kw_path.exists():
        logger.warning("keyword_summary.csv not found — keyword ablation skipped")
        return pd.DataFrame()

    kw_df = pd.read_csv(kw_path)
    if "keyword" not in kw_df.columns or "avg_CAR" not in kw_df.columns:
        logger.warning("keyword_summary.csv missing required columns")
        return pd.DataFrame()

    # Filter: minimum |CAR| and exclude filing artefacts
    valid = kw_df[
        (kw_df["avg_CAR"].abs() >= min_car) &
        (~kw_df["keyword"].apply(_is_filing_artefact))
    ].copy()
    kw_map = dict(zip(valid["keyword"].str.lower(), valid["avg_CAR"]))
    logger.info("Keyword map: %d entries after artefact filter (min_car=%.3f)", len(kw_map), min_car)

    max_adj = getattr(config, "ENHANCED_SIGNAL_MAX_ADJUSTMENT", 0.25)
    pred_adj = pred_df.copy()

    def _adjust(row):
        headline = str(row["headline"]).lower()
        adj = sum(car for kw, car in kw_map.items() if kw in headline)
        adj = max(-max_adj, min(max_adj, adj))
        return float(np.clip(row["signal_raw"] + adj, -1.0, 1.0))

    pred_adj["signal_raw"] = pred_adj.apply(_adjust, axis=1)
    return pred_adj


def run_keyword_ablation(pred_df: pd.DataFrame, prices: pd.DataFrame) -> None:
    buy, sell, conf = 0.45, -0.45, 0.70

    print("\n" + "=" * 65)
    print("  ABLATION 3 — Keyword CAR Weighting")
    print("  (buy=0.45, conf=0.70, lag=1d)")
    print("=" * 65)
    header = f"  {'Variant':<28} {'n_sig':>6} {'n_act':>6} {'IC':>9} {'hit':>7} {'Sharpe':>8}"
    print(header)
    print("  " + "-" * 63)

    variants: list[tuple[str, pd.DataFrame]] = []

    # Baseline: no keyword adjustment
    signals_base = aggregate_signals(pred_df, conf)
    variants.append(("no keywords (base)", signals_base))

    # With keyword adjustment
    pred_kw = build_keyword_adjusted_signals(pred_df, min_car=0.003)
    if not pred_kw.empty:
        signals_kw = aggregate_signals(pred_kw, conf)
        variants.append(("with keywords (CAR adj)", signals_kw))

        # Stricter keyword filter
        pred_kw2 = build_keyword_adjusted_signals(pred_df, min_car=0.005)
        if not pred_kw2.empty:
            signals_kw2 = aggregate_signals(pred_kw2, conf)
            variants.append(("with keywords (CAR>0.5%)", signals_kw2))

    rows: list[dict] = []
    for label, signals in variants:
        if signals.empty:
            print(f"  {label:<28} {'0':>6} {'—':>6} {'—':>9} {'—':>7} {'—':>8}")
            continue
        ic, p, n     = compute_ic(signals, prices, lag=1)
        hr            = compute_hit_rate(signals, prices, lag=1, buy_thresh=buy, sell_thresh=sell)
        returns_df    = signals_to_returns(signals, prices, buy, sell)
        n_active      = int((returns_df["position"] != 0).sum()) if not returns_df.empty else 0
        metrics       = calculate_portfolio_metrics(returns_df, label) if not returns_df.empty and n_active > 0 else {}
        sharpe        = metrics.get("sharpe_ratio", np.nan)

        print(
            f"  {label:<28} {len(signals):>6} {n_active:>6} "
            f"{safe_fmt(ic, '+.4f'):>9} {safe_fmt(hr, '.3f'):>7} {safe_fmt(sharpe, '+.3f'):>8}"
        )
        rows.append({
            "variant": label, "n_signals": len(signals), "n_active": n_active,
            "ic_1d": ic, "p_value": p, "hit_rate_1d": hr, "sharpe": sharpe,
        })

    if rows:
        out = config.METRICS_DIR / "ablation_keywords.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n  Saved -> {out}")

    if len(rows) >= 2:
        ic_base = rows[0].get("ic_1d", np.nan)
        ic_kw   = rows[1].get("ic_1d", np.nan)
        delta   = (ic_kw - ic_base) if not (np.isnan(ic_base) or np.isnan(ic_kw)) else np.nan
        print(f"\n  IC delta (keywords vs base): {safe_fmt(delta, '+.4f')}")
        if not np.isnan(delta):
            if delta > 0.005:
                print("  Keywords IMPROVE IC — keep them in the pipeline.")
            elif delta < -0.005:
                print("  Keywords HURT IC — consider removing keyword weighting.")
            else:
                print("  Keywords have negligible IC impact (<0.005).")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_diagnosis(thresh_df: pd.DataFrame, horizon_df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  DIAGNOSIS SUMMARY")
    print("=" * 65)

    if not thresh_df.empty:
        best_ic_row  = thresh_df.loc[thresh_df["ic_1d"].abs().idxmax()] if thresh_df["ic_1d"].notna().any() else None
        best_sharpe_row = thresh_df.loc[thresh_df["sharpe"].idxmax()] if thresh_df["sharpe"].notna().any() else None
        if best_ic_row is not None:
            print(f"\n  Best IC config   : {best_ic_row['config'].strip()}")
            print(f"    IC = {safe_fmt(best_ic_row['ic_1d'], '+.4f')}  |  hit_rate = {safe_fmt(best_ic_row['hit_rate_1d'], '.3f')}  |  Sharpe = {safe_fmt(best_ic_row['sharpe'], '+.3f')}")
        if best_sharpe_row is not None:
            print(f"\n  Best Sharpe config: {best_sharpe_row['config'].strip()}")
            print(f"    IC = {safe_fmt(best_sharpe_row['ic_1d'], '+.4f')}  |  hit_rate = {safe_fmt(best_sharpe_row['hit_rate_1d'], '.3f')}  |  Sharpe = {safe_fmt(best_sharpe_row['sharpe'], '+.3f')}")

    if not horizon_df.empty:
        ic_vals = horizon_df.dropna(subset=["ic"])
        if not ic_vals.empty:
            best_hr  = horizon_df.loc[horizon_df["hit_rate"].idxmax()]
            print(f"\n  Best hit_rate lag: {int(best_hr['lag_days'])}d  -> hit_rate = {safe_fmt(best_hr['hit_rate'], '.3f')}")
            max_ic_row = ic_vals.loc[ic_vals["ic"].idxmax()]
            print(f"  Highest IC lag   : {int(max_ic_row['lag_days'])}d  -> IC = {safe_fmt(max_ic_row['ic'], '+.4f')}")
            positive_ic_lags = horizon_df[horizon_df["ic"] > 0]["lag_days"].tolist()
            if positive_ic_lags:
                print(f"  Positive IC at lags: {positive_ic_lags} — these are the usable horizons.")
            else:
                print("  No positive IC at any tested horizon.")
                print("  Diagnosis: signal direction is systematically wrong — check label ordering or data leakage.")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    force_vader = "--vader" in sys.argv
    model, model_name = load_model(force_vader)

    print(f"\n  AlphaLens — Threshold / Horizon / Keyword Ablation  [{model_name.upper()}]")
    print(f"  Processed data dir : {config.PROCESSED_DATA_DIR}")
    print(f"  Results dir        : {config.METRICS_DIR}")

    # ── single inference pass ──────────────────────────────────────────────
    pred_df = load_and_predict(model)

    # ── fetch prices once ─────────────────────────────────────────────────
    start = pred_df["date"].min().strftime("%Y-%m-%d")
    end   = (pred_df["date"].max() + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    prices = fetch_price_data(config.TICKERS, start=start, end=end)

    if prices.empty:
        print("\n  ERROR: Could not fetch price data. Aborting.")
        sys.exit(1)

    # ── ablations ─────────────────────────────────────────────────────────
    thresh_df  = run_threshold_ablation(pred_df, prices)
    horizon_df = run_horizon_ablation(pred_df, prices)
    run_keyword_ablation(pred_df, prices)

    # ── summary ───────────────────────────────────────────────────────────
    print_diagnosis(thresh_df, horizon_df)


if __name__ == "__main__":
    main()
