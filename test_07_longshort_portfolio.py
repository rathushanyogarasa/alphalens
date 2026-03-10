"""test_07_longshort_portfolio.py
==================================
Validates the cross-sectional long-short strategy improvements.

Compares three strategies side-by-side:
  A. Baseline (original approach): individual stock signals, next-day, 1-stock-at-a-time
  B. Long-short (hold=1d):  quintile ranking, 1-day hold
  C. Long-short (hold=2d):  quintile ranking, 2-day hold  <-- primary target

Also prints:
  - IC decay at lags 1-10d for the ranked signal
  - Quantile return spread bar (Q5 vs Q1)
  - Keyword filter check (shows which keywords were blocked)

Usage
-----
    python test_07_longshort_portfolio.py [--vader]
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
logger = logging.getLogger("test07")
logger.setLevel(logging.INFO)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.backtest import fetch_price_data, calculate_portfolio_metrics, generate_signals, calculate_strategy_returns
from src.longshort_engine import (
    run_longshort_backtest,
    compute_ls_ic_decay,
    plot_longshort_equity_curve,
    plot_quantile_spread,
)
from src.model import SIGNAL_MAP


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def load_model(force_vader: bool = False):
    if force_vader:
        from src.model import VADERBaseline
        logger.info("Using VADERBaseline")
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
    return VADERBaseline(), "vader"


# ---------------------------------------------------------------------------
# Inference (single pass, both signal columns)
# ---------------------------------------------------------------------------


def load_and_predict(model) -> pd.DataFrame:
    news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
    if not news_path.exists():
        raise FileNotFoundError(f"combined_news.csv not found at {news_path}")
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
    news_df["signal_normal"] = news_df["label_name"].map(SIGNAL_MAP).fillna(0)

    dist = news_df["label_name"].value_counts().to_dict()
    logger.info("Labels: %s", dist)
    return news_df


# ---------------------------------------------------------------------------
# Baseline strategy (individual stock, single-day hold)
# ---------------------------------------------------------------------------


def run_baseline(news_df: pd.DataFrame, prices: pd.DataFrame, model) -> dict:
    """Original backtest approach using generate_signals / calculate_strategy_returns."""
    signals  = generate_signals(news_df, model)
    returns  = calculate_strategy_returns(signals, prices)
    return calculate_portfolio_metrics(returns, model_name="baseline")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def sf(val, fmt=".4f", na="  n/a  "):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    return format(val, fmt)


def print_comparison_table(results: list[tuple[str, dict]]) -> None:
    rows = [
        ("Sharpe ratio",         "sharpe_ratio",          "+.3f"),
        ("Sortino ratio",        "sortino_ratio",          "+.3f"),
        ("Calmar ratio",         "calmar_ratio",           "+.3f"),
        ("Annualised return",    "annualised_return",      "+.1%"),
        ("Ann. volatility",      "annualised_volatility",  ".2%"),
        ("Max drawdown",         "max_drawdown",           ".2%"),
        ("Win rate",             "win_rate",               ".1%"),
        ("Hit rate",             "hit_rate",               ".3f"),
        ("IC (position-level)",  "ic",                     "+.4f"),
        ("Total return",         "total_return",           "+.2%"),
    ]

    labels = [r[0] for r in results]
    header = f"  {'Metric':<28}" + "".join(f" {l:>14}" for l in labels)
    print(header)
    print("  " + "-" * (28 + 15 * len(results)))
    for name, key, fmt in rows:
        vals = "".join(f" {sf(m.get(key, np.nan), fmt):>14}" for _, m in results)
        print(f"  {name:<28}{vals}")


# ---------------------------------------------------------------------------
# Keyword filter validation
# ---------------------------------------------------------------------------


def print_keyword_filter_check() -> None:
    """Show which keywords in keyword_summary.csv are now blocked."""
    kw_path = config.METRICS_DIR / "keyword_summary.csv"
    if not kw_path.exists():
        return

    import re, sys
    sys.path.insert(0, str(ROOT / "src"))
    from keyword_analysis import _is_meaningful_keyword

    kw_df     = pd.read_csv(kw_path)
    blocked   = kw_df[~kw_df["keyword"].apply(_is_meaningful_keyword)]["keyword"].tolist()
    surviving = kw_df[kw_df["keyword"].apply(_is_meaningful_keyword)]["keyword"].tolist()

    print("\n" + "=" * 60)
    print("  Keyword Filter Check")
    print("=" * 60)
    print(f"  Total keywords : {len(kw_df)}")
    print(f"  Blocked        : {len(blocked)}")
    print(f"  Surviving      : {len(surviving)}")
    if blocked:
        print(f"\n  Blocked keywords:")
        for kw in blocked[:15]:
            print(f"    - {kw}")
        if len(blocked) > 15:
            print(f"    ... and {len(blocked) - 15} more")
    if surviving:
        print(f"\n  Surviving keywords (up to 10):")
        for kw in surviving[:10]:
            print(f"    + {kw}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    force_vader = "--vader" in sys.argv
    model, model_name = load_model(force_vader)

    print(f"\n  AlphaLens — Long-Short Portfolio Test  [{model_name.upper()}]")
    print(f"  BUY_THRESHOLD={config.BUY_THRESHOLD}  SELL_THRESHOLD={config.SELL_THRESHOLD}")
    print(f"  HOLD_DAYS={config.LONGSHORT_HOLD_DAYS}  QUANTILE_CUTOFF={config.LONGSHORT_QUANTILE_CUTOFF}")

    # ── single inference pass ─────────────────────────────────────────────
    pred_df = load_and_predict(model)

    start   = pred_df["date"].min().strftime("%Y-%m-%d")
    end     = (pred_df["date"].max() + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    prices  = fetch_price_data(config.TICKERS, start=start, end=end)
    if prices.empty:
        print("ERROR: no price data")
        sys.exit(1)

    # ── keyword filter check ──────────────────────────────────────────────
    print_keyword_filter_check()

    # ── IC decay on the ranked signal ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  IC Decay (cross-sectional, aggregated signal)")
    print("=" * 60)
    ic_df = compute_ls_ic_decay(pred_df, prices, lags=[1, 2, 3, 5, 10], signal_col="signal_normal")
    print(f"  {'Lag':>5}  {'IC':>9}  {'p-val':>8}  {'sig':>5}  {'n':>6}")
    print("  " + "-" * 50)
    for _, row in ic_df.iterrows():
        sig = "YES *" if row["significant"] else "no"
        print(f"  {int(row['lag']):>4}d  {sf(row['ic'], '+.4f'):>9}  {sf(row['p_value'], '.4f'):>8}  {sig:>5}  {int(row['n']):>6}")

    # ── strategy A: baseline ─────────────────────────────────────────────
    print("\n  Running Strategy A: Baseline (individual stock, 1d hold) ...")
    base_metrics = run_baseline(pred_df, prices, model)

    # ── strategy B: long-short, 1-day hold ───────────────────────────────
    print("  Running Strategy B: Long-Short (1d hold) ...")
    _, ls1_metrics = run_longshort_backtest(
        pred_df, prices,
        hold_days=1,
        quantile_cutoff=config.LONGSHORT_QUANTILE_CUTOFF,
        signal_col="signal_normal",
    )

    # ── strategy C: long-short, 2-day hold ───────────────────────────────
    print("  Running Strategy C: Long-Short (2d hold) ...")
    ls2_returns, ls2_metrics = run_longshort_backtest(
        pred_df, prices,
        hold_days=2,
        quantile_cutoff=config.LONGSHORT_QUANTILE_CUTOFF,
        signal_col="signal_normal",
    )

    # ── strategy D: long-short, 3-day hold ───────────────────────────────
    print("  Running Strategy D: Long-Short (3d hold) ...")
    _, ls3_metrics = run_longshort_backtest(
        pred_df, prices,
        hold_days=3,
        quantile_cutoff=config.LONGSHORT_QUANTILE_CUTOFF,
        signal_col="signal_normal",
    )

    # ── comparison table ─────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  Strategy Comparison")
    print("=" * 75)
    all_results = [
        ("A: Baseline", base_metrics),
        ("B: L/S 1d",   ls1_metrics),
        ("C: L/S 2d",   ls2_metrics),
        ("D: L/S 3d",   ls3_metrics),
    ]
    print_comparison_table(all_results)

    # ── plots ─────────────────────────────────────────────────────────────
    if not ls2_returns.empty:
        plot_longshort_equity_curve(ls2_returns, ls2_metrics, label=f"{model_name}_ls2d")
        plot_quantile_spread(pred_df, prices, lag=2, label=f"{model_name}_ls2d", signal_col="signal_normal")
        print(f"\n  Plots saved -> {config.PLOTS_DIR}")

    # ── save metrics ──────────────────────────────────────────────────────
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for label, m in all_results:
        row = {"strategy": label}
        row.update(m)
        summary_rows.append(row)
    out = config.METRICS_DIR / f"longshort_comparison_{model_name}.csv"
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    print(f"  Metrics saved -> {out}")

    # ── verdict ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)
    best_sharpe = max(all_results, key=lambda x: x[1].get("sharpe_ratio", -999))
    print(f"\n  Best strategy : {best_sharpe[0]}")
    print(f"  Sharpe        : {sf(best_sharpe[1].get('sharpe_ratio'), '+.3f')}")
    print(f"  Ann. return   : {sf(best_sharpe[1].get('annualised_return'), '+.1%')}")
    print(f"  Hit rate      : {sf(best_sharpe[1].get('hit_rate'), '.3f')}")

    base_sharpe = base_metrics.get("sharpe_ratio", -999)
    best_ls_sharpe = max(ls1_metrics.get("sharpe_ratio", -999),
                         ls2_metrics.get("sharpe_ratio", -999),
                         ls3_metrics.get("sharpe_ratio", -999))
    if best_ls_sharpe > base_sharpe:
        improvement = best_ls_sharpe - base_sharpe
        print(f"\n  Cross-sectional ranking IMPROVES Sharpe by {improvement:+.3f}")
        print("  -> Recommendation: adopt long-short mode as the primary backtest.")
    else:
        print("\n  Baseline matches or beats long-short.")
        print("  -> More headline data or a wider ticker universe may help.")
    print()


if __name__ == "__main__":
    main()
