"""test_12_walk_forward.py
==========================
Quarterly walk-forward out-of-sample validation for the champion strategy
(S30+M50+V20, baseline signals, 2d hold, 20% quantile cutoff).

Method
------
Rather than optimising model parameters (FinBERT is pre-trained), the
walk-forward here validates that the SIGNAL is consistent across time:

1. Split the signal history into non-overlapping calendar quarters.
2. For each quarter, compute the cross-sectional long-short portfolio
   performance independently.
3. Report IC, Sharpe, hit rate, and cumulative return per quarter.
4. Compute the overall out-of-sample IC t-statistic and ICIR.

This is functionally equivalent to a "hold-out" validation where each
quarter is a fresh out-of-sample period for the signal structure.

Champion strategy (frozen after test_09/11 analysis)
-----------------------------------------------------
  Weights   : sentiment=0.30, momentum=0.50, volatility=0.20, liquidity=0.00
  Hold days : 2
  Quantile  : 20% (top/bottom quintile)
  Signal    : baseline (binary, no shock filter, no sector demeaning)
  Costs     : 10 bps transaction + 5 bps slippage (15 bps total round-trip)

Usage
-----
    python test_12_walk_forward.py [--vader] [--freq Q]  # Q=quarterly, M=monthly

Outputs
-------
    metrics/walk_forward_<model>.csv
    plots/walk_forward_<model>.png
"""

from __future__ import annotations

import argparse
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
logger = logging.getLogger("test12")
logger.setLevel(logging.INFO)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.backtest import fetch_price_data
from src.factor_engine import fetch_volume_data, build_factor_signals, combine_factors
from src.longshort_engine import _aggregate_signals, run_longshort_from_signals
from src.model import SIGNAL_MAP
from src.sentiment_cache import CachedPredictor

# ---------------------------------------------------------------------------
# Champion strategy (frozen)
# ---------------------------------------------------------------------------

CHAMPION_WEIGHTS = {
    "sentiment":  0.30,
    "momentum":   0.50,
    "volatility": 0.20,
    "liquidity":  0.00,
}

BACKTEST_KWARGS = dict(
    hold_days=2,
    quantile_cutoff=0.20,
    weighting="equal",
    tc_bps=config.TRANSACTION_COST_BPS,
    slip_bps=config.SLIPPAGE_BPS,
)


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
            return model, "finbert"
        except Exception as exc:
            logger.warning("FinBERT load failed (%s) -- using VADER", exc)
    from src.model import VADERBaseline
    return VADERBaseline(), "vader"


def sf(val, fmt=".4f", na="  n/a "):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    return format(val, fmt)


def _spearman_ic(factor_df: pd.DataFrame, prices: pd.DataFrame, fwd_days: int = 2) -> pd.Series:
    """Compute daily Spearman IC between combined_score and fwd_days return."""
    if "combined_score" not in factor_df.columns or prices.empty:
        return pd.Series(dtype=float)

    daily_ics: dict = {}
    for date, grp in factor_df.groupby("date"):
        if len(grp) < 5:
            continue
        fwd_returns = {}
        for _, row in grp.iterrows():
            t = row["ticker"]
            if t not in prices.columns:
                continue
            try:
                idx = prices.index.searchsorted(date)
                if idx + fwd_days >= len(prices):
                    continue
                p0 = prices[t].iloc[idx]
                p1 = prices[t].iloc[idx + fwd_days]
                if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
                    continue
                fwd_returns[t] = (p1 - p0) / p0
            except Exception:
                continue

        if len(fwd_returns) < 5:
            continue
        common = grp[grp["ticker"].isin(fwd_returns)].copy()
        common["fwd_ret"] = common["ticker"].map(fwd_returns)
        ic = common["combined_score"].corr(common["fwd_ret"], method="spearman")
        if pd.notna(ic):
            daily_ics[date] = ic

    return pd.Series(daily_ics).sort_index()


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def run_walk_forward(
    pred_df: pd.DataFrame,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    freq: str = "Q",
) -> pd.DataFrame:
    """Run the champion strategy independently on each calendar period.

    Args:
        pred_df: FinBERT predictions with ``date``, ``ticker``, ``signal_normal``.
        prices:  Daily close prices (date × ticker).
        volumes: Daily dollar volumes (date × ticker).
        freq:    Pandas period frequency: ``"Q"`` (quarter) or ``"M"`` (month).

    Returns:
        DataFrame with one row per period and columns:
        period, start, end, n_signal_days, n_tickers, sharpe, annualised_return,
        max_drawdown, hit_rate, ic, ic_t_stat.
    """
    pred_df = pred_df.copy()
    pred_df["period"] = pred_df["date"].dt.to_period(freq)
    periods = sorted(pred_df["period"].unique())

    records: list[dict] = []
    vol_arg = volumes if not volumes.empty else None

    for period in periods:
        p_preds = pred_df[pred_df["period"] == period].drop(columns=["period"])
        if p_preds.empty:
            continue

        p_start = str(period.start_time.date())
        p_end   = str(period.end_time.date())

        raw = _aggregate_signals(
            p_preds, "signal_normal", config.CONFIDENCE_THRESHOLD,
            confidence_weighted=False, max_headlines_per_day=0,
        )
        if raw.empty or len(raw["ticker"].unique()) < 5:
            logger.debug("Period %s: too few tickers (%d) — skipping",
                         period, raw["ticker"].nunique() if not raw.empty else 0)
            continue

        factor_df = build_factor_signals(raw, prices, vol_arg)
        if factor_df.empty:
            continue

        combined = combine_factors(factor_df, weights=CHAMPION_WEIGHTS)

        try:
            ret_df, metrics = run_longshort_from_signals(
                combined, prices, signal_col="combined_score", **BACKTEST_KWARGS,
            )
        except Exception as exc:
            logger.warning("Period %s backtest failed: %s", period, exc)
            continue

        # Per-period IC
        ic_series = _spearman_ic(combined, prices, fwd_days=BACKTEST_KWARGS["hold_days"])
        ic_mean   = float(ic_series.mean()) if len(ic_series) else np.nan
        ic_std    = float(ic_series.std())  if len(ic_series) > 1 else np.nan
        ic_t      = (ic_mean / (ic_std / np.sqrt(len(ic_series)))) if (ic_std and ic_std > 0 and len(ic_series) > 1) else np.nan

        records.append({
            "period":            str(period),
            "start":             p_start,
            "end":               p_end,
            "n_signal_days":     metrics.get("n_periods", len(ic_series)),
            "n_tickers":         int(raw["ticker"].nunique()),
            "sharpe":            metrics.get("sharpe_ratio", np.nan),
            "annualised_return": metrics.get("annualised_return", np.nan),
            "max_drawdown":      metrics.get("max_drawdown", np.nan),
            "hit_rate":          metrics.get("hit_rate", np.nan),
            "ic_mean":           round(ic_mean, 5) if not np.isnan(ic_mean) else np.nan,
            "ic_t_stat":         round(ic_t,    3)  if not np.isnan(ic_t)    else np.nan,
        })
        logger.info("Period %-8s  Sharpe=%+.2f  IC=%+.4f  tickers=%d",
                    period, metrics.get("sharpe_ratio", np.nan), ic_mean,
                    int(raw["ticker"].nunique()))

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_walk_forward(wf_df: pd.DataFrame, model_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return

    if wf_df.empty:
        return

    valid = wf_df.dropna(subset=["sharpe"])
    if valid.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"Walk-Forward Validation — {model_name.upper()} | Champion: S30+M50+V20", fontsize=13)

    periods = valid["period"].tolist()
    x = range(len(periods))

    # Sharpe per period
    ax = axes[0]
    colors = ["#2196F3" if s >= 0 else "#F44336" for s in valid["sharpe"]]
    ax.bar(x, valid["sharpe"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(valid["sharpe"].mean(), color="navy", linestyle="--", linewidth=1.2,
               label=f"Mean={valid['sharpe'].mean():+.2f}")
    ax.set_title("Sharpe Ratio by Period")
    ax.set_xticks(list(x)); ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9); ax.set_ylabel("Sharpe")

    # IC per period
    ax = axes[1]
    ic_vals = valid["ic_mean"].fillna(0)
    colors_ic = ["#4CAF50" if v >= 0 else "#FF5722" for v in ic_vals]
    ax.bar(x, ic_vals, color=colors_ic)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(ic_vals.mean(), color="darkgreen", linestyle="--", linewidth=1.2,
               label=f"Mean IC={ic_vals.mean():+.4f}")
    ax.set_title("Information Coefficient (IC) by Period")
    ax.set_xticks(list(x)); ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9); ax.set_ylabel("IC (Spearman)")

    # Cumulative return
    ax = axes[2]
    cum_ret = (1 + valid["annualised_return"].fillna(0) / 252 * valid["n_signal_days"].fillna(10) * 2).cumprod() - 1
    ax.plot(x, cum_ret * 100, marker="o", color="#673AB7", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Cumulative Return (approximate)")
    ax.set_xticks(list(x)); ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative Return (%)")

    plt.tight_layout()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PLOTS_DIR / f"walk_forward_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Walk-forward plot saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--vader", action="store_true")
    parser.add_argument("--freq",  default="Q",
                        help="Period frequency: Q=quarterly, M=monthly (default: Q)")
    args = parser.parse_args()

    model, model_name = load_model(args.vader)

    print(f"\n  AlphaLens — Walk-Forward Validation  [{model_name.upper()}]")
    print(f"  Champion weights: S30+M50+V20 | 2d hold | 20% quantile")
    print(f"  Frequency: {args.freq} | Costs: {config.TRANSACTION_COST_BPS+config.SLIPPAGE_BPS} bps total")

    # Load and score headlines
    news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
    if not news_path.exists():
        print("ERROR: combined_news.csv not found. Run main.py or fetch_gdelt_news.py first.")
        sys.exit(1)

    news_df = pd.read_csv(news_path, parse_dates=["date"])
    print(f"\n  Headlines: {len(news_df):,} | Tickers: {news_df['ticker'].nunique()}")

    predictor = CachedPredictor(model, model_name)
    cache_hits = min(predictor.cache_size, len(news_df))
    print(f"  Sentiment cache: {predictor.cache_size:,} entries "
          f"({'full cache hit — skipping inference' if cache_hits >= len(news_df) else 'partial'})")

    news_df = predictor.predict_dataframe(news_df)
    news_df["date"]          = pd.to_datetime(news_df["date"]).dt.normalize()
    news_df["signal_normal"] = news_df["label_name"].map(SIGNAL_MAP).fillna(0)

    # Fetch market data
    start = news_df["date"].min().strftime("%Y-%m-%d")
    end   = (news_df["date"].max() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"\n  Fetching prices ({start} to {end}) ...")
    prices = fetch_price_data(config.TICKERS, start=start, end=end)
    if prices.empty:
        print("ERROR: no price data"); sys.exit(1)

    volumes = fetch_volume_data(config.TICKERS, start=start, end=end)

    # Run walk-forward
    print(f"\n  Running {args.freq}-frequency walk-forward ...")
    wf_df = run_walk_forward(news_df, prices, volumes, freq=args.freq)

    if wf_df.empty:
        print("\n  No valid periods found. Need more data coverage.")
        return

    # =========================================================================
    # Results
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"  Walk-Forward Results  ({len(wf_df)} {'quarters' if args.freq=='Q' else 'periods'})")
    print("=" * 80)
    print(f"\n  {'Period':<10} {'Tickers':>8} {'Sharpe':>8} {'AnnRet':>9} "
          f"{'MaxDD':>8} {'HitRate':>9} {'IC':>8} {'IC_t':>7}")
    print("  " + "-" * 72)

    for _, row in wf_df.iterrows():
        print(f"  {row['period']:<10} {int(row['n_tickers']):>8} "
              f"{sf(row['sharpe'], '+.3f'):>8} "
              f"{sf(row['annualised_return'], '+.1%'):>9} "
              f"{sf(row['max_drawdown'], '.1%'):>8} "
              f"{sf(row['hit_rate'], '.3f'):>9} "
              f"{sf(row['ic_mean'], '+.4f'):>8} "
              f"{sf(row['ic_t_stat'], '+.2f'):>7}")

    valid = wf_df.dropna(subset=["sharpe"])
    if len(valid):
        print("\n  " + "-" * 72)
        pct_positive_sharpe = (valid["sharpe"] > 0).mean()
        pct_positive_ic     = (valid["ic_mean"] > 0).mean()
        mean_sharpe         = valid["sharpe"].mean()
        mean_ic             = valid["ic_mean"].mean()
        icir                = (mean_ic / valid["ic_mean"].std() * np.sqrt(len(valid))
                               if valid["ic_mean"].std() > 1e-9 else np.nan)

        print(f"\n  Summary across {len(valid)} periods:")
        print(f"    Mean Sharpe          : {mean_sharpe:+.3f}")
        print(f"    % Periods +Sharpe    : {pct_positive_sharpe:.0%}")
        print(f"    Mean IC              : {mean_ic:+.5f}")
        print(f"    % Periods +IC        : {pct_positive_ic:.0%}")
        print(f"    IC Information Ratio : {sf(icir, '+.3f')}")

        if pct_positive_sharpe >= 0.60 and mean_sharpe > 0:
            verdict = "PASS — signal is consistent across periods (≥60% positive Sharpe)"
        elif pct_positive_sharpe >= 0.50:
            verdict = "BORDERLINE — marginal consistency (50-60% positive Sharpe)"
        else:
            verdict = "FAIL — signal is inconsistent across periods (<50% positive Sharpe)"
        print(f"\n  Walk-forward verdict: {verdict}")

    # Save
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.METRICS_DIR / f"walk_forward_{model_name}.csv"
    wf_df.to_csv(out_path, index=False)
    print(f"\n  Results saved -> {out_path}")

    plot_walk_forward(wf_df, model_name)
    print()


if __name__ == "__main__":
    main()
