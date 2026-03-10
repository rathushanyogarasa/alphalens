"""test_08_improvements.py
===========================
Comprehensive improvement test covering all five priority upgrades:

  1. Quantile cutoff sweep   — 10 / 15 / 20% with 2d hold
  2. Volatility scaling      — equal-weight vs inverse-vol-weight
  3. Macro VIX filter        — on vs off (halve size when VIX > 28)
  4. Pre vs post cost        — Sharpe before and after 10+5 bps
  5. Turnover + gross exposure reporting

Each test runs on the same single model inference pass (predictions cached).
A final summary table ranks all variants by Sharpe.

Usage
-----
    python test_08_improvements.py [--vader]
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
logger = logging.getLogger("test08")
logger.setLevel(logging.INFO)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.backtest import fetch_price_data
from src.longshort_engine import run_longshort_backtest, plot_longshort_equity_curve
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
        raise FileNotFoundError(f"combined_news.csv not found at {news_path}")
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


def run(pred_df, prices, label, **kwargs) -> dict:
    """Single backtest run, returns metrics dict augmented with label."""
    logger.info("  Running: %s", label)
    _, m = run_longshort_backtest(pred_df, prices, signal_col="signal_normal", **kwargs)
    m["label"] = label
    return m


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------


COLS = [
    ("Sharpe",     "sharpe_ratio",          "+.3f"),
    ("Sortino",    "sortino_ratio",          "+.3f"),
    ("Ann.Ret",    "annualised_return",      "+.1%"),
    ("MaxDD",      "max_drawdown",           ".1%"),
    ("WinRate",    "win_rate",               ".1%"),
    ("HitRate",    "hit_rate",               ".3f"),
    ("IC",         "ic",                     "+.4f"),
    ("Turnover",   "avg_turnover",           ".3f"),
    ("GrossExp",   "avg_gross_exposure",     ".3f"),
    ("N/side",     "avg_positions_per_side", ".1f"),
]


def print_section(title: str, results: list[dict]) -> None:
    print("\n" + "=" * (32 + 10 * len(COLS)))
    print(f"  {title}")
    print("=" * (32 + 10 * len(COLS)))
    hdr = f"  {'Variant':<30}" + "".join(f" {c[0]:>9}" for c in COLS)
    print(hdr)
    print("  " + "-" * (30 + 10 * len(COLS)))
    for m in results:
        vals = "".join(f" {sf(m.get(k, np.nan), fmt):>9}" for _, k, fmt in COLS)
        print(f"  {m.get('label','?'):<30}{vals}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    force_vader = "--vader" in sys.argv
    model, model_name = load_model(force_vader)

    print(f"\n  AlphaLens — Improvement Suite  [{model_name.upper()}]")
    print(f"  Thresholds: BUY={config.BUY_THRESHOLD} | SELL={config.SELL_THRESHOLD} | CONF={config.CONFIDENCE_THRESHOLD}")

    pred_df = load_and_predict(model)
    start   = pred_df["date"].min().strftime("%Y-%m-%d")
    end     = (pred_df["date"].max() + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    prices  = fetch_price_data(config.TICKERS, start=start, end=end)
    if prices.empty:
        print("ERROR: no price data")
        sys.exit(1)

    all_results: list[dict] = []

    # ── SECTION 1: Quantile cutoff sweep ────────────────────────────────
    print("\n  [1] Quantile Cutoff Sweep  (equal-weight, hold=2d, with costs)")
    q_results = []
    for q in [0.20, 0.15, 0.10]:
        label = f"Q={int(q*100)}%  2d equal"
        m = run(pred_df, prices, label,
                hold_days=2, quantile_cutoff=q,
                weighting="equal",
                tc_bps=config.TRANSACTION_COST_BPS,
                slip_bps=config.SLIPPAGE_BPS)
        q_results.append(m)
        all_results.append(m)
    print_section("Quantile Cutoff Sweep", q_results)

    best_q = max(q_results, key=lambda m: m.get("sharpe_ratio", -999))
    best_q_cutoff = best_q.get("quantile_cutoff", 0.20)
    print(f"\n  Best cutoff: {int(best_q_cutoff*100)}%  (Sharpe={sf(best_q.get('sharpe_ratio'), '+.3f')})")

    # ── SECTION 2: Weighting comparison ─────────────────────────────────
    print("\n  [2] Weighting Comparison  (hold=2d, with costs)")
    w_results = []
    for weighting in ["equal", "vol_scaled"]:
        for q in [best_q_cutoff, 0.10]:
            label = f"Q={int(q*100)}% {weighting[:3]}  2d"
            m = run(pred_df, prices, label,
                    hold_days=2, quantile_cutoff=q,
                    weighting=weighting,
                    tc_bps=config.TRANSACTION_COST_BPS,
                    slip_bps=config.SLIPPAGE_BPS)
            w_results.append(m)
            all_results.append(m)
    print_section("Weighting: Equal vs Vol-Scaled", w_results)

    best_w = max(w_results, key=lambda m: m.get("sharpe_ratio", -999))
    best_weighting  = best_w.get("weighting", "equal")
    best_w_cutoff   = best_w.get("quantile_cutoff", best_q_cutoff)
    print(f"\n  Best: Q={int(best_w_cutoff*100)}% {best_weighting}  (Sharpe={sf(best_w.get('sharpe_ratio'), '+.3f')})")

    # ── SECTION 3: Pre vs post cost ──────────────────────────────────────
    print("\n  [3] Pre vs Post Transaction Cost  (best config from above)")
    c_results = []
    for tc, sl, label in [
        (0.0,  0.0,  f"Q={int(best_w_cutoff*100)}% {best_weighting[:3]}  NO cost"),
        (5.0,  2.5,  f"Q={int(best_w_cutoff*100)}% {best_weighting[:3]}  low cost (5+2.5bps)"),
        (10.0, 5.0,  f"Q={int(best_w_cutoff*100)}% {best_weighting[:3]}  full cost (10+5bps)"),
        (20.0, 10.0, f"Q={int(best_w_cutoff*100)}% {best_weighting[:3]}  high cost (20+10bps)"),
    ]:
        m = run(pred_df, prices, label,
                hold_days=2, quantile_cutoff=best_w_cutoff,
                weighting=best_weighting, tc_bps=tc, slip_bps=sl)
        c_results.append(m)
        all_results.append(m)
    print_section("Pre vs Post Transaction Cost", c_results)

    # ── SECTION 4: VIX macro filter ──────────────────────────────────────
    print("\n  [4] VIX Macro Filter  (best config, with full costs)")
    v_results = []
    for vix_on, label in [
        (False, f"Q={int(best_w_cutoff*100)}% {best_weighting[:3]}  no VIX filter"),
        (True,  f"Q={int(best_w_cutoff*100)}% {best_weighting[:3]}  +VIX filter"),
    ]:
        m = run(pred_df, prices, label,
                hold_days=2, quantile_cutoff=best_w_cutoff,
                weighting=best_weighting,
                macro_vix_filter=vix_on,
                tc_bps=config.TRANSACTION_COST_BPS,
                slip_bps=config.SLIPPAGE_BPS)
        v_results.append(m)
        all_results.append(m)
    print_section("VIX Macro Filter", v_results)

    vix_sharpe_delta = (v_results[1].get("sharpe_ratio", 0) or 0) - (v_results[0].get("sharpe_ratio", 0) or 0)
    if vix_sharpe_delta > 0.05:
        print(f"\n  VIX filter IMPROVES Sharpe by {vix_sharpe_delta:+.3f} — enable it.")
    elif vix_sharpe_delta < -0.05:
        print(f"\n  VIX filter HURTS Sharpe by {vix_sharpe_delta:+.3f} — leave it off.")
    else:
        print(f"\n  VIX filter has negligible impact ({vix_sharpe_delta:+.3f}) — optional.")

    # ── SECTION 5: Hold period comparison ────────────────────────────────
    print("\n  [5] Hold Period Comparison  (best Q+weighting, full costs)")
    h_results = []
    for hd in [1, 2, 3, 5]:
        label = f"Q={int(best_w_cutoff*100)}% {best_weighting[:3]}  hold={hd}d"
        m = run(pred_df, prices, label,
                hold_days=hd, quantile_cutoff=best_w_cutoff,
                weighting=best_weighting,
                tc_bps=config.TRANSACTION_COST_BPS,
                slip_bps=config.SLIPPAGE_BPS)
        h_results.append(m)
        all_results.append(m)
    print_section("Hold Period Comparison", h_results)

    best_h = max(h_results, key=lambda m: m.get("sharpe_ratio", -999))
    print(f"\n  Best hold period: {best_h.get('hold_days','?')}d  (Sharpe={sf(best_h.get('sharpe_ratio'), '+.3f')})")

    # ── OVERALL LEADERBOARD ──────────────────────────────────────────────
    # Deduplicate by label, keep unique configs
    seen: set[str] = set()
    unique = []
    for m in all_results:
        lbl = m.get("label", "")
        if lbl not in seen:
            seen.add(lbl)
            unique.append(m)

    ranked = sorted(unique, key=lambda m: m.get("sharpe_ratio", -999), reverse=True)
    print("\n" + "=" * (32 + 10 * len(COLS)))
    print("  OVERALL LEADERBOARD (all variants, ranked by Sharpe)")
    print("=" * (32 + 10 * len(COLS)))
    hdr = f"  {'Variant':<30}" + "".join(f" {c[0]:>9}" for c in COLS)
    print(hdr)
    print("  " + "-" * (30 + 10 * len(COLS)))
    for i, m in enumerate(ranked[:12]):
        marker = " <-- BEST" if i == 0 else ""
        vals   = "".join(f" {sf(m.get(k, np.nan), fmt):>9}" for _, k, fmt in COLS)
        print(f"  {m.get('label','?'):<30}{vals}{marker}")

    # ── Save & plot best variant ─────────────────────────────────────────
    champion = ranked[0]
    champ_label = champion.get("label", "best")
    champ_q     = champion.get("quantile_cutoff", best_w_cutoff)
    champ_w     = champion.get("weighting", best_weighting)
    champ_hd    = champion.get("hold_days", 2)
    champ_vix   = champion.get("macro_vix_filter", False)

    print(f"\n  Generating equity curve for champion: {champ_label}")
    champ_returns, champ_metrics = run_longshort_backtest(
        pred_df, prices, signal_col="signal_normal",
        hold_days=champ_hd, quantile_cutoff=champ_q,
        weighting=champ_w, macro_vix_filter=champ_vix,
        tc_bps=config.TRANSACTION_COST_BPS, slip_bps=config.SLIPPAGE_BPS,
    )
    if not champ_returns.empty:
        plot_longshort_equity_curve(champ_returns, champ_metrics, label=f"{model_name}_champion")
        print(f"  Plot saved -> {config.PLOTS_DIR / f'longshort_equity_{model_name}_champion.png'}")

    # Save full leaderboard to CSV
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard_path = config.METRICS_DIR / f"improvement_leaderboard_{model_name}.csv"
    pd.DataFrame(ranked).to_csv(leaderboard_path, index=False)
    print(f"  Leaderboard saved -> {leaderboard_path}")

    # ── Final verdict ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL VERDICT")
    print("=" * 60)
    best_sharpe = champion.get("sharpe_ratio", np.nan)
    best_ann    = champion.get("annualised_return", np.nan)
    best_dd     = champion.get("max_drawdown", np.nan)
    best_hit    = champion.get("hit_rate", np.nan)

    print(f"\n  Champion config  : {champ_label}")
    print(f"  Sharpe           : {sf(best_sharpe, '+.3f')}")
    print(f"  Annualised return: {sf(best_ann, '+.1%')}")
    print(f"  Max drawdown     : {sf(best_dd, '.1%')}")
    print(f"  Hit rate         : {sf(best_hit, '.3f')}")

    print()
    target_sharpe = 0.60
    if not np.isnan(best_sharpe):
        if best_sharpe >= target_sharpe:
            print(f"  TARGET MET: Sharpe >= {target_sharpe:.2f}")
            print("  -> This config is ready for paper-trading validation.")
        elif best_sharpe > 0:
            gap = target_sharpe - best_sharpe
            print(f"  Positive Sharpe but below target ({target_sharpe:.2f})")
            print(f"  Gap: {gap:.3f} Sharpe units")
            print("  -> Gather more headlines; wider ticker universe may help.")
        else:
            print("  Sharpe still negative after all improvements.")
            print("  -> Root cause likely insufficient headline coverage per ticker.")
            print("     Consider expanding news sources or the ticker universe.")
    print()


if __name__ == "__main__":
    main()
