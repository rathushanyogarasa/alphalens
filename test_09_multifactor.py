"""test_09_multifactor.py
==========================
Multi-factor signal combination test.

Combines the FinBERT sentiment signal with three price-based factors:

  momentum    20-day price return (skip 2 days) — trend continuation
  volatility  Negative realised vol — low-vol anomaly
  liquidity   Log average dollar volume — execution quality filter

All factors are cross-sectionally z-scored before combination so no
single factor dominates by scale.

What this script tests
----------------------
  1. Individual factor IC — each factor's standalone predictive power
  2. Strategy comparison — sentiment-only vs each factor vs combined
  3. Weight grid search — systematic search over weight combinations
  4. Best config equity curve + full metrics

Usage
-----
    python test_09_multifactor.py [--vader]
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test09")
logger.setLevel(logging.INFO)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.backtest import fetch_price_data
from src.factor_engine import (
    fetch_volume_data,
    build_factor_signals,
    combine_factors,
    compute_factor_ics,
)
from src.longshort_engine import (
    _aggregate_signals,
    run_longshort_from_signals,
    plot_longshort_equity_curve,
)
from src.model import SIGNAL_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_model(force_vader: bool = False):
    if force_vader:
        from src.model import VADERBaseline
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
            logger.info("Loaded FinBERT from %s", checkpoint)
            return model, "finbert"
        except Exception as exc:
            logger.warning("FinBERT load failed (%s) — using VADER", exc)
    from src.model import VADERBaseline
    return VADERBaseline(), "vader"


def load_and_predict(model) -> pd.DataFrame:
    news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
    if not news_path.exists():
        raise FileNotFoundError(f"{news_path} not found")
    news_df = pd.read_csv(news_path, parse_dates=["date"])
    logger.info("Loaded %d headlines — running inference ...", len(news_df))
    texts = news_df["headline"].tolist()
    preds: list[dict] = []
    for i in range(0, len(texts), config.BATCH_SIZE):
        preds.extend(model.predict(texts[i : i + config.BATCH_SIZE]))
    news_df = news_df.copy().reset_index(drop=True)
    news_df["label_name"]    = [p["label_name"] for p in preds]
    news_df["confidence"]    = [p["confidence"]  for p in preds]
    news_df["date"]          = pd.to_datetime(news_df["date"]).dt.normalize()
    news_df["signal_normal"] = news_df["label_name"].map(SIGNAL_MAP).fillna(0)
    return news_df


def sf(val, fmt=".4f", na="  n/a "):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    return format(val, fmt)


BACKTEST_KWARGS = dict(
    hold_days=2,
    quantile_cutoff=0.20,
    weighting="equal",
    tc_bps=config.TRANSACTION_COST_BPS,
    slip_bps=config.SLIPPAGE_BPS,
)

RESULT_COLS = [
    ("Sharpe",   "sharpe_ratio",       "+.3f"),
    ("Sortino",  "sortino_ratio",       "+.3f"),
    ("Ann.Ret",  "annualised_return",   "+.1%"),
    ("MaxDD",    "max_drawdown",        ".1%"),
    ("WinRate",  "win_rate",            ".1%"),
    ("HitRate",  "hit_rate",            ".3f"),
    ("IC",       "ic",                  "+.4f"),
    ("Turnover", "avg_turnover",        ".3f"),
]


def print_table(title: str, results: list[dict]) -> None:
    w = 28 + 10 * len(RESULT_COLS)
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")
    hdr = f"  {'Variant':<26}" + "".join(f" {c[0]:>9}" for c in RESULT_COLS)
    print(hdr)
    print("  " + "-" * (w - 2))
    for m in results:
        vals = "".join(f" {sf(m.get(k, np.nan), fmt):>9}" for _, k, fmt in RESULT_COLS)
        print(f"  {str(m.get('label','')):<26}{vals}")


def backtest(signals_df, signal_col, label) -> dict:
    _, m = run_longshort_from_signals(
        signals_df, prices,
        signal_col=signal_col,
        **BACKTEST_KWARGS,
    )
    m["label"] = label
    return m


# ---------------------------------------------------------------------------
# Weight grid
# ---------------------------------------------------------------------------

# (sentiment, momentum, volatility, liquidity)
WEIGHT_GRID = [
    # Sentiment-only baselines
    (1.00, 0.00, 0.00, 0.00),
    # Two-factor combos
    (0.70, 0.30, 0.00, 0.00),
    (0.60, 0.00, 0.40, 0.00),
    (0.60, 0.00, 0.00, 0.40),
    # Three-factor combos
    (0.50, 0.30, 0.20, 0.00),
    (0.50, 0.20, 0.30, 0.00),
    (0.40, 0.30, 0.20, 0.10),
    (0.40, 0.20, 0.20, 0.20),
    # Higher momentum weight
    (0.40, 0.40, 0.20, 0.00),
    (0.30, 0.50, 0.20, 0.00),
    # Higher vol weight
    (0.40, 0.20, 0.40, 0.00),
    # Equal-weight all non-zero
    (0.33, 0.33, 0.34, 0.00),
]


def weights_label(ws: tuple) -> str:
    s, m, v, l = ws
    parts = []
    if s: parts.append(f"S{int(s*100)}")
    if m: parts.append(f"M{int(m*100)}")
    if v: parts.append(f"V{int(v*100)}")
    if l: parts.append(f"L{int(l*100)}")
    return "+".join(parts)


# ---------------------------------------------------------------------------
# Main (uses module-level `prices` for brevity in backtest())
# ---------------------------------------------------------------------------

prices: pd.DataFrame = pd.DataFrame()  # filled in main()


def main():
    global prices

    force_vader = "--vader" in sys.argv
    model, model_name = load_model(force_vader)

    print(f"\n  AlphaLens — Multi-Factor Signal Test  [{model_name.upper()}]")

    # ── data ──────────────────────────────────────────────────────────────
    pred_df = load_and_predict(model)
    start   = pred_df["date"].min().strftime("%Y-%m-%d")
    end     = (pred_df["date"].max() + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    prices  = fetch_price_data(config.TICKERS, start=start, end=end)
    if prices.empty:
        print("ERROR: no price data")
        sys.exit(1)

    print(f"  Fetching volume data for {len(config.TICKERS)} tickers ...")
    volumes = fetch_volume_data(config.TICKERS, start=start, end=end)
    vol_ok  = not volumes.empty
    print(f"  Volume data: {'OK (%d days)' % len(volumes) if vol_ok else 'unavailable — liquidity factor will be 0'}")

    # ── aggregate signals once ─────────────────────────────────────────────
    raw_signals = _aggregate_signals(pred_df, "signal_normal", config.CONFIDENCE_THRESHOLD)
    if raw_signals.empty:
        print("ERROR: no signals generated")
        sys.exit(1)
    print(f"\n  Signal dates: {raw_signals['date'].nunique()} | signal rows: {len(raw_signals)}")

    # ── build factor matrix ────────────────────────────────────────────────
    print("  Building factor matrix (momentum, vol, liquidity) ...")
    factor_df = build_factor_signals(
        raw_signals, prices,
        volumes=volumes if vol_ok else None,
    )

    # ── SECTION 1: Individual factor ICs ──────────────────────────────────
    print("\n" + "=" * 62)
    print("  SECTION 1 — Individual Factor IC  (lag=2d)")
    print("=" * 62)
    ic_df = compute_factor_ics(factor_df, prices, lag=2)
    print(f"  {'Factor':<20} {'IC':>9} {'p-val':>8} {'sig':>6} {'n':>6}")
    print("  " + "-" * 55)
    for _, row in ic_df.iterrows():
        sig = "YES *" if row["significant"] else "no"
        print(f"  {row['factor']:<20} {sf(row['ic'], '+.4f'):>9} "
              f"{sf(row['p_value'], '.4f'):>8} {sig:>6} {int(row['n']):>6}")

    best_standalone = ic_df.dropna(subset=["ic"]).loc[ic_df["ic"].abs().idxmax(), "factor"] \
        if not ic_df.dropna(subset=["ic"]).empty else "signal"
    print(f"\n  Strongest standalone factor: {best_standalone}")

    # ── SECTION 2: Strategy comparison ────────────────────────────────────
    print("\n  Building per-factor signal DataFrames ...")
    s2_results: list[dict] = []

    # Sentiment only
    sent_df = raw_signals.copy()
    sent_df = sent_df.rename(columns={"signal": "signal"})  # no-op, explicit
    s2_results.append(backtest(sent_df, "signal", "Sentiment only"))

    # Each price factor alone (use the z-scored column directly)
    for col, lbl in [("momentum", "Momentum only"),
                     ("vol_factor", "Vol-factor only"),
                     ("liquidity", "Liquidity only")]:
        if col in factor_df.columns and factor_df[col].notna().any():
            df = factor_df[["date", "ticker", col]].dropna(subset=[col])
            if not df.empty:
                s2_results.append(backtest(df, col, lbl))
        else:
            print(f"  Skipping {lbl} — no data")

    print_table("Strategy Comparison — Single Factor", s2_results)

    # ── SECTION 3: Weight grid search ─────────────────────────────────────
    print("\n  Running weight grid (%d combinations) ..." % len(WEIGHT_GRID))
    grid_results: list[dict] = []

    for ws in WEIGHT_GRID:
        w_dict = {"sentiment": ws[0], "momentum": ws[1],
                  "volatility": ws[2], "liquidity": ws[3]}
        combined = combine_factors(factor_df, weights=w_dict)
        lbl = weights_label(ws)
        m   = backtest(combined, "combined_score", lbl)
        grid_results.append(m)

    grid_results.sort(key=lambda m: m.get("sharpe_ratio", -999), reverse=True)
    print_table("Weight Grid Search (ranked by Sharpe)", grid_results)

    # ── SECTION 4: Best config equity curve ───────────────────────────────
    champion = grid_results[0]
    champ_ws_label = champion.get("label", "")
    # Find the matching weights tuple
    champ_ws = next(
        (ws for ws in WEIGHT_GRID if weights_label(ws) == champ_ws_label),
        (0.5, 0.3, 0.2, 0.0)
    )
    print(f"\n  Champion: {champ_ws_label}  Sharpe={sf(champion.get('sharpe_ratio'), '+.3f')}")

    w_dict   = {"sentiment": champ_ws[0], "momentum": champ_ws[1],
                "volatility": champ_ws[2], "liquidity": champ_ws[3]}
    best_df  = combine_factors(factor_df, weights=w_dict)
    best_ret, best_met = run_longshort_from_signals(
        best_df, prices, signal_col="combined_score", **BACKTEST_KWARGS
    )
    if not best_ret.empty:
        plot_longshort_equity_curve(best_ret, best_met,
                                    label=f"{model_name}_multifactor")
        print(f"  Equity curve saved -> {config.PLOTS_DIR}/longshort_equity_{model_name}_multifactor.png")

    # ── Save ──────────────────────────────────────────────────────────────
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ic_path = config.METRICS_DIR / f"multifactor_ic_{model_name}.csv"
    ic_df.to_csv(ic_path, index=False)

    all_results = s2_results + grid_results
    lb_path = config.METRICS_DIR / f"multifactor_leaderboard_{model_name}.csv"
    pd.DataFrame(all_results).drop(columns=["label"], errors="ignore") \
        .assign(label=[m.get("label", "") for m in all_results]) \
        .to_csv(lb_path, index=False)
    print(f"  Leaderboard saved -> {lb_path}")

    # ── Verdict ───────────────────────────────────────────────────────────
    sent_sharpe   = next((m.get("sharpe_ratio", np.nan) for m in s2_results
                          if m.get("label") == "Sentiment only"), np.nan)
    best_sharpe   = champion.get("sharpe_ratio", np.nan)
    improvement   = (best_sharpe - sent_sharpe) if not (
        np.isnan(best_sharpe) or np.isnan(sent_sharpe)) else np.nan

    print("\n" + "=" * 62)
    print("  VERDICT")
    print("=" * 62)
    print(f"\n  Sentiment-only Sharpe : {sf(sent_sharpe, '+.3f')}")
    print(f"  Best multi-factor     : {sf(best_sharpe, '+.3f')}  ({champ_ws_label})")
    print(f"  Improvement           : {sf(improvement, '+.3f')} Sharpe units")

    if not np.isnan(improvement):
        if improvement > 0.05:
            print("\n  Multi-factor combination IMPROVES the signal.")
            print("  Recommendation: use the champion weights in production.")
            print(f"  -> Set DEFAULT_WEIGHTS = {w_dict}")
        elif improvement > -0.05:
            print("\n  Multi-factor has negligible impact on this dataset.")
            print("  Likely cause: universe too small (10 tickers) for cross-sectional")
            print("  factor diversification.  Expand to 50+ tickers to see benefit.")
        else:
            print("\n  Adding price factors HURTS the signal on this dataset.")
            print("  Stick with sentiment-only until the universe is wider.")
    print()


if __name__ == "__main__":
    main()
