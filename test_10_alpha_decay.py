"""test_10_alpha_decay.py
========================
Alpha Decay Test and Signal Stability Test.

Tests how quickly signal alpha decays over time, and whether the IC
is stable across calendar years/quarters (robustness check).

Sections
--------
  1. IC Decay curve   -- Spearman IC at lags [1,2,3,5,10,15,20] days
                         for sentiment-only vs combined signal (S40+M40+V20)
  2. Signal Stability -- IC by calendar year and by quarter
                         (detects in-sample overfitting / regime sensitivity)
  3. IC Information Ratio (mean_IC / std_IC per period)
  4. Cost-sensitivity on combined signal (Sharpe vs tc_bps sweep)

Usage
-----
    python test_10_alpha_decay.py [--vader]

Outputs
-------
    plots/alpha_decay_<model>.png
    metrics/alpha_decay_stability_<model>.csv
    metrics/alpha_decay_ic_<model>.csv
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test10")
logger.setLevel(logging.INFO)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.backtest import fetch_price_data
from src.factor_engine import (
    fetch_volume_data,
    build_factor_signals,
    combine_factors,
)
from src.longshort_engine import (
    _aggregate_signals,
    run_longshort_from_signals,
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
            logger.warning("FinBERT load failed (%s) -- using VADER", exc)
    from src.model import VADERBaseline
    return VADERBaseline(), "vader"


def load_and_predict(model) -> pd.DataFrame:
    news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
    if not news_path.exists():
        raise FileNotFoundError(f"{news_path} not found")
    news_df = pd.read_csv(news_path, parse_dates=["date"])
    logger.info("Loaded %d headlines -- running inference ...", len(news_df))
    texts = news_df["headline"].tolist()
    preds: list[dict] = []
    for i in range(0, len(texts), config.BATCH_SIZE):
        preds.extend(model.predict(texts[i: i + config.BATCH_SIZE]))
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


# ---------------------------------------------------------------------------
# Core: IC at a single lag
# ---------------------------------------------------------------------------


def compute_ic_at_lag(
    signals_df: pd.DataFrame,
    prices: pd.DataFrame,
    signal_col: str,
    lag: int,
) -> tuple[float, float, int]:
    """Compute Spearman IC between signal_col and lag-day forward return.

    Returns:
        (ic, p_value, n_pairs)
    """
    price_sorted = prices.sort_index()
    pairs: list[tuple[float, float]] = []

    for _, row in signals_df.iterrows():
        date   = pd.Timestamp(row["date"])
        ticker = str(row["ticker"])
        val    = row.get(signal_col, np.nan)
        if pd.isna(val) or ticker not in price_sorted.columns:
            continue
        pos = price_sorted.index.searchsorted(date)
        if pos + lag >= len(price_sorted):
            continue
        p0    = price_sorted[ticker].iloc[pos]
        p_lag = price_sorted[ticker].iloc[pos + lag]
        if pd.isna(p0) or pd.isna(p_lag) or p0 == 0:
            continue
        pairs.append((float(val), float((p_lag - p0) / p0)))

    if len(pairs) < 10:
        return np.nan, np.nan, len(pairs)

    arr = pd.DataFrame(pairs, columns=["factor_val", "fwd_ret"])
    ic, p = scipy_stats.spearmanr(arr["factor_val"], arr["fwd_ret"])
    return float(ic), float(p), len(pairs)


# ---------------------------------------------------------------------------
# Section 1: IC decay curve
# ---------------------------------------------------------------------------


def compute_ic_decay(
    signals_df: pd.DataFrame,
    prices: pd.DataFrame,
    signal_col: str,
    lags: list[int],
) -> pd.DataFrame:
    """IC at each lag. Returns DataFrame with columns lag, ic, p_value, n, significant."""
    rows = []
    for lag in lags:
        ic, p, n = compute_ic_at_lag(signals_df, prices, signal_col, lag)
        rows.append({
            "lag":         lag,
            "ic":          ic,
            "p_value":     p,
            "n":           n,
            "significant": bool(not np.isnan(p) and p < 0.05),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 2: IC by calendar period
# ---------------------------------------------------------------------------


def compute_ic_by_period(
    signals_df: pd.DataFrame,
    prices: pd.DataFrame,
    signal_col: str,
    lag: int = 2,
    period: str = "year",  # "year" or "quarter"
) -> pd.DataFrame:
    """IC computed separately for each calendar year or quarter.

    Returns DataFrame with columns: period, ic, p_value, n, significant.
    """
    df = signals_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if period == "year":
        df["period_label"] = df["date"].dt.year.astype(str)
    elif period == "quarter":
        df["period_label"] = (
            df["date"].dt.year.astype(str)
            + "-Q"
            + df["date"].dt.quarter.astype(str)
        )
    else:
        raise ValueError(f"period must be 'year' or 'quarter', got {period!r}")

    rows = []
    for lbl, grp in df.groupby("period_label"):
        ic, p, n = compute_ic_at_lag(grp, prices, signal_col, lag)
        rows.append({
            "period":      lbl,
            "ic":          ic,
            "p_value":     p,
            "n":           n,
            "significant": bool(not np.isnan(p) and p < 0.05),
        })

    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 3: IC Information Ratio helper
# ---------------------------------------------------------------------------


def ic_information_ratio(ic_series: pd.Series) -> float:
    """mean(IC) / std(IC) -- Sharpe of the IC itself.  NaN if insufficient data."""
    clean = ic_series.dropna()
    if len(clean) < 3:
        return np.nan
    mean = float(clean.mean())
    std  = float(clean.std())
    return mean / std if std > 1e-8 else np.nan


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_alpha_decay(
    decay_sent: pd.DataFrame,
    decay_comb: pd.DataFrame,
    model_name: str,
) -> None:
    """Plot IC decay curves for sentiment-only vs combined signal."""
    try:
        from src.plot_style import GOLD, NAVY, FIG_DPI, apply_plot_style
        apply_plot_style()
        colour_sent = NAVY
        colour_comb = GOLD
    except ImportError:
        colour_sent = "#003366"
        colour_comb = "#FFB800"
        FIG_DPI = 150

    fig, ax = plt.subplots(figsize=(9, 5))

    # Shaded region: "healthy" single-factor decay zone (IC 0.01-0.06)
    ax.axhspan(0.01, 0.06, alpha=0.07, color="green", label="_healthy zone")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    lags_s = decay_sent["lag"].tolist()
    ics_s  = decay_sent["ic"].tolist()
    lags_c = decay_comb["lag"].tolist()
    ics_c  = decay_comb["ic"].tolist()

    ax.plot(lags_s, ics_s, "o-", color=colour_sent, linewidth=2,
            markersize=6, label="Sentiment only")
    ax.plot(lags_c, ics_c, "s-", color=colour_comb, linewidth=2,
            markersize=6, label="Combined (S40+M40+V20)")

    # Annotate significant points
    for _, row in decay_comb.iterrows():
        if row["significant"] and not np.isnan(row["ic"]):
            ax.annotate(
                f"*",
                xy=(row["lag"], row["ic"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color=colour_comb,
            )

    ax.set_xlabel("Forward lag (trading days)")
    ax.set_ylabel("Spearman IC")
    ax.set_title(f"Alpha Decay Curve  [{model_name.upper()}]\n"
                 f"* = p < 0.05  |  shaded: healthy IC zone (0.01-0.06)")
    ax.legend()
    ax.set_xticks(decay_comb["lag"].tolist())

    plt.tight_layout()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / f"alpha_decay_{model_name}.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Decay plot saved -> {out_path}")


def plot_ic_stability(
    by_year: pd.DataFrame,
    by_quarter: pd.DataFrame,
    signal_label: str,
    model_name: str,
) -> None:
    """Bar charts of IC by year and by quarter."""
    try:
        from src.plot_style import GOLD, NAVY, FIG_DPI, apply_plot_style
        apply_plot_style()
        colour = NAVY
    except ImportError:
        colour = "#003366"
        FIG_DPI = 150

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df, title in [
        (axes[0], by_year,    "IC by Calendar Year"),
        (axes[1], by_quarter, "IC by Quarter"),
    ]:
        if df.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(title)
            continue

        colours = [
            ("#28a745" if v > 0 else "#dc3545")
            for v in df["ic"].fillna(0)
        ]
        ax.bar(df["period"], df["ic"].fillna(0), color=colours, alpha=0.8)
        ax.axhline(0, color="gray", linewidth=0.8)

        # Error bars only where we have enough data (n >= 10)
        ax.set_xlabel("Period")
        ax.set_ylabel("Spearman IC")
        ax.set_title(f"{title}\n{signal_label}")
        ax.tick_params(axis="x", rotation=45)

        # Annotate n
        for _, row in df.iterrows():
            if not np.isnan(row["ic"]):
                ax.annotate(
                    f"n={int(row['n'])}",
                    xy=(row["period"], row["ic"]),
                    xytext=(0, 5 if row["ic"] >= 0 else -14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )

    fig.suptitle(f"Signal Stability  [{model_name.upper()}]", fontsize=13)
    plt.tight_layout()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / f"alpha_stability_{model_name}.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Stability plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Cost sensitivity sweep
# ---------------------------------------------------------------------------


BACKTEST_KWARGS_BASE = dict(
    hold_days=2,
    quantile_cutoff=0.20,
    weighting="equal",
)

CHAMPION_WEIGHTS = {"sentiment": 0.40, "momentum": 0.40, "volatility": 0.20, "liquidity": 0.00}


def cost_sensitivity(factor_df, prices) -> list[dict]:
    """Sweep tc_bps [0, 5, 10, 15, 20] and report Sharpe for combined signal."""
    results = []
    combined = combine_factors(factor_df, weights=CHAMPION_WEIGHTS)
    for tc in [0, 5, 10, 15, 20]:
        sl = tc / 2
        _, m = run_longshort_from_signals(
            combined, prices,
            signal_col="combined_score",
            **BACKTEST_KWARGS_BASE,
            tc_bps=float(tc),
            slip_bps=float(sl),
        )
        m["tc_bps"] = tc
        m["slip_bps"] = sl
        results.append(m)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DECAY_LAGS = [1, 2, 3, 5, 10, 15, 20]


def main():
    force_vader = "--vader" in sys.argv
    model, model_name = load_model(force_vader)

    print(f"\n  AlphaLens -- Alpha Decay & Signal Stability Test  [{model_name.upper()}]")

    # -- data -----------------------------------------------------------------
    pred_df = load_and_predict(model)
    start   = pred_df["date"].min().strftime("%Y-%m-%d")
    end     = (pred_df["date"].max() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    prices  = fetch_price_data(config.TICKERS, start=start, end=end)
    if prices.empty:
        print("ERROR: no price data")
        sys.exit(1)

    print(f"  Fetching volume data ...")
    volumes = fetch_volume_data(config.TICKERS, start=start, end=end)
    vol_ok  = not volumes.empty

    # -- aggregate signals once -----------------------------------------------
    raw_signals = _aggregate_signals(pred_df, "signal_normal", config.CONFIDENCE_THRESHOLD)
    if raw_signals.empty:
        print("ERROR: no signals generated")
        sys.exit(1)
    print(f"  Signal dates: {raw_signals['date'].nunique()} | rows: {len(raw_signals)}")

    # -- build factor matrix --------------------------------------------------
    print("  Building factor matrix ...")
    factor_df = build_factor_signals(
        raw_signals, prices,
        volumes=volumes if vol_ok else None,
    )
    combined_df = combine_factors(factor_df, weights=CHAMPION_WEIGHTS)

    # =========================================================================
    # SECTION 1: IC Decay curves
    # =========================================================================
    print("\n" + "=" * 62)
    print("  SECTION 1 -- Alpha Decay (IC vs forward lag)")
    print("=" * 62)

    decay_sent = compute_ic_decay(raw_signals,   prices, "signal",         DECAY_LAGS)
    decay_comb = compute_ic_decay(combined_df,   prices, "combined_score", DECAY_LAGS)

    print(f"\n  {'Lag':>5} | {'Sent IC':>9} {'p':>7} | {'Comb IC':>9} {'p':>7} {'n':>6}")
    print("  " + "-" * 52)
    for (_, rs), (_, rc) in zip(decay_sent.iterrows(), decay_comb.iterrows()):
        sig_s = "*" if rs["significant"] else " "
        sig_c = "*" if rc["significant"] else " "
        print(
            f"  {int(rs['lag']):>5} | "
            f"{sf(rs['ic'], '+.4f'):>9}{sig_s} {sf(rs['p_value'], '.4f'):>7} | "
            f"{sf(rc['ic'], '+.4f'):>9}{sig_c} {sf(rc['p_value'], '.4f'):>7} "
            f"{int(rc['n']) if not np.isnan(rc.get('n', np.nan)) else 0:>6}"
        )
    print("  (* = p < 0.05)")

    # Decay health assessment
    peak_ic_comb = decay_comb["ic"].dropna().abs().max() if not decay_comb["ic"].dropna().empty else 0
    lag2_ic      = decay_comb.loc[decay_comb["lag"] == 2, "ic"].values
    lag10_ic     = decay_comb.loc[decay_comb["lag"] == 10, "ic"].values
    lag2_v  = float(lag2_ic[0])  if len(lag2_ic)  and not np.isnan(lag2_ic[0])  else 0.0
    lag10_v = float(lag10_ic[0]) if len(lag10_ic) and not np.isnan(lag10_ic[0]) else 0.0

    print(f"\n  Peak IC (combined): {sf(peak_ic_comb, '.4f')}")
    if peak_ic_comb > 0.03:
        print("  -> IC reaches statistically meaningful level (>0.03). Alpha present.")
    elif peak_ic_comb > 0.01:
        print("  -> IC marginal (0.01-0.03). Signal exists but universe too small.")
    else:
        print("  -> Peak IC < 0.01. Very weak signal on this universe/sample.")

    if lag10_v != 0 and lag2_v != 0:
        decay_ratio = lag10_v / lag2_v
        print(f"  Decay ratio IC[lag=10] / IC[lag=2]: {decay_ratio:+.2f}")
        if decay_ratio > 0.5:
            print("  -> Slow decay: alpha persists >10 days. Longer hold periods viable.")
        elif decay_ratio > 0:
            print("  -> Moderate decay: alpha half-life ~5 days. Current 2d hold is appropriate.")
        else:
            print("  -> IC reverses sign at lag 10: short-term momentum reversing, "
                  "typical for news-driven signals.")

    plot_alpha_decay(decay_sent, decay_comb, model_name)

    # =========================================================================
    # SECTION 2: Signal Stability
    # =========================================================================
    print("\n" + "=" * 62)
    print("  SECTION 2 -- Signal Stability (IC by time period, lag=2d)")
    print("=" * 62)

    # Use combined signal for stability
    by_year    = compute_ic_by_period(combined_df, prices, "combined_score", lag=2, period="year")
    by_quarter = compute_ic_by_period(combined_df, prices, "combined_score", lag=2, period="quarter")

    # Also compute for sentiment-only
    by_year_sent    = compute_ic_by_period(raw_signals, prices, "signal", lag=2, period="year")
    by_quarter_sent = compute_ic_by_period(raw_signals, prices, "signal", lag=2, period="quarter")

    print(f"\n  {'Period':<12} {'Sent IC':>9} {'Comb IC':>9} {'n(comb)':>9}")
    print("  " + "-" * 44)
    for _, ry in by_year_sent.iterrows():
        rc_row = by_year[by_year["period"] == ry["period"]]
        ic_c = rc_row["ic"].values[0] if len(rc_row) else np.nan
        n_c  = int(rc_row["n"].values[0]) if len(rc_row) else 0
        print(f"  {str(ry['period']):<12} {sf(ry['ic'], '+.4f'):>9} {sf(ic_c, '+.4f'):>9} {n_c:>9}")

    # IC Information Ratio per period
    ir_sent = ic_information_ratio(by_year_sent["ic"])
    ir_comb = ic_information_ratio(by_year["ic"])
    print(f"\n  IC Information Ratio (by year):")
    print(f"    Sentiment-only : {sf(ir_sent, '+.3f')}")
    print(f"    Combined       : {sf(ir_comb, '+.3f')}")
    if not np.isnan(ir_comb):
        if ir_comb > 0.5:
            print("    -> ICIR > 0.5: IC is consistent across years. Strong.")
        elif ir_comb > 0.0:
            print("    -> ICIR in (0, 0.5): IC positive but variable year-to-year.")
        else:
            print("    -> ICIR <= 0: IC is unstable or negative in some years.")

    print(f"\n  By Quarter (combined signal):")
    print(f"  {'Quarter':<12} {'IC':>9} {'p':>8} {'n':>6} {'sig':>5}")
    print("  " + "-" * 44)
    for _, row in by_quarter.iterrows():
        sig = "YES" if row["significant"] else "no"
        print(f"  {str(row['period']):<12} {sf(row['ic'], '+.4f'):>9} "
              f"{sf(row['p_value'], '.4f'):>8} {int(row['n']):>6} {sig:>5}")

    # Stability summary
    n_pos_years = int((by_year["ic"].dropna() > 0).sum())
    n_tot_years = int(by_year["ic"].notna().sum())
    if n_tot_years > 0:
        pct_positive = n_pos_years / n_tot_years
        print(f"\n  Years with positive IC: {n_pos_years}/{n_tot_years} ({pct_positive:.0%})")
        if pct_positive >= 0.75:
            print("  -> Signal is stable: positive IC in >=75% of years.")
        elif pct_positive >= 0.50:
            print("  -> Signal is moderately stable (50-75% of years positive).")
        else:
            print("  -> Signal unstable: <50% of years positive. Likely regime-sensitive.")

    plot_ic_stability(by_year, by_quarter, "Combined S40+M40+V20", model_name)

    # =========================================================================
    # SECTION 3: Cost Sensitivity (combined signal)
    # =========================================================================
    print("\n" + "=" * 62)
    print("  SECTION 3 -- Cost Sensitivity (combined S40+M40+V20, lag=2d)")
    print("=" * 62)
    print(f"\n  {'tc_bps':>7} {'slip_bps':>9} {'Sharpe':>9} {'Ann.Ret':>9} {'MaxDD':>8}")
    print("  " + "-" * 48)

    cost_results = cost_sensitivity(factor_df, prices)
    breakeven_tc: float | None = None
    for m in cost_results:
        sharpe = m.get("sharpe_ratio", np.nan)
        ann    = m.get("annualised_return", np.nan)
        dd     = m.get("max_drawdown", np.nan)
        tc     = int(m["tc_bps"])
        sl     = int(m["slip_bps"])
        print(f"  {tc:>7} {sl:>9} {sf(sharpe, '+.3f'):>9} {sf(ann, '+.1%'):>9} {sf(dd, '.1%'):>8}")
        if breakeven_tc is None and not np.isnan(sharpe) and sharpe <= 0:
            breakeven_tc = float(tc)

    if breakeven_tc is not None:
        print(f"\n  Break-even tc_bps: ~{breakeven_tc:.0f} bps (Sharpe turns negative here)")
    else:
        print("\n  Sharpe remains positive across all tested cost levels.")

    # =========================================================================
    # Save outputs
    # =========================================================================
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # IC decay table
    decay_comb["signal"] = "combined_S40M40V20"
    decay_sent["signal"] = "sentiment_only"
    ic_table = pd.concat([decay_sent, decay_comb], ignore_index=True)
    ic_path  = config.METRICS_DIR / f"alpha_decay_ic_{model_name}.csv"
    ic_table.to_csv(ic_path, index=False)
    print(f"\n  IC decay table saved -> {ic_path}")

    # Stability table
    by_year["period_type"]    = "year"
    by_quarter["period_type"] = "quarter"
    stability = pd.concat([by_year, by_quarter], ignore_index=True)
    stab_path = config.METRICS_DIR / f"alpha_decay_stability_{model_name}.csv"
    stability.to_csv(stab_path, index=False)
    print(f"  Stability table saved -> {stab_path}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print("\n" + "=" * 62)
    print("  VERDICT")
    print("=" * 62)

    lag2_ic_comb = decay_comb.loc[decay_comb["lag"] == 2, "ic"].values
    lag2_val     = float(lag2_ic_comb[0]) if len(lag2_ic_comb) else np.nan
    lag2_sig     = bool(decay_comb.loc[decay_comb["lag"] == 2, "significant"].values[0]) \
        if len(decay_comb.loc[decay_comb["lag"] == 2]) else False

    print(f"\n  Combined signal IC at lag=2d : {sf(lag2_val, '+.4f')}  "
          f"({'significant' if lag2_sig else 'not significant'})")
    print(f"  IC Information Ratio (years) : {sf(ir_comb, '+.3f')}")
    print(f"  Pct years positive IC        : {n_pos_years}/{n_tot_years}")

    if not np.isnan(lag2_val):
        if lag2_val > 0.03 and lag2_sig:
            print("\n  PASS: IC > 0.03 and significant at lag=2d.")
            print("  Signal is ready for live paper-trading at moderate costs.")
        elif lag2_val > 0.01:
            print("\n  MARGINAL: IC positive but weak (0.01-0.03) or not significant.")
            print("  Expand to 50+ tickers to improve statistical power.")
        else:
            print("\n  FAIL: IC at lag=2 is below 0.01.")
            print("  Signal too weak for live trading. Investigate data quality.")
    print()


if __name__ == "__main__":
    main()
