"""test_11_whites_reality_check.py
===================================
White's (1999) Reality Check + full improvement suite.

Addresses the key statistical concern: when you test 12+ weight combinations
and pick the best Sharpe, some of that outperformance is selection bias.
White's Reality Check bootstraps the null distribution of the *best* Sharpe
across all strategies and reports the p-value.

If p < 0.05 — the champion strategy survives data-mining adjustment.
If p > 0.10 — the Sharpe is likely cherry-picked from noise.

Improvements tested in this script
------------------------------------
  A. Baseline                     (binary signal, no extras)
  B. Confidence-weighted signals  (signal * confidence vs binary ±1)
  C. News shock filter            (cap 3 headlines / ticker / day)
  D. Sector-neutral signals       (demean within GICS sector each day)
  E. Combined: B + C + D
  F. Volatility targeting         (position weight ∝ signal / realized_vol)
  G. VIX regime filter            (halve long exposure when VIX > 30)

Each improvement is backtested over the full weight grid.
White's Reality Check is run on the FULL strategy pool (all weight combos
× all signal variants).

Usage
-----
    python test_11_whites_reality_check.py [--vader] [--boot 5000]

Outputs
-------
    metrics/whites_rc_results_<model>.csv
    metrics/whites_rc_returns_<model>.csv
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
logger = logging.getLogger("test11")
logger.setLevel(logging.INFO)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.sentiment_cache import CachedPredictor
from src.backtest import fetch_price_data
from src.factor_engine import (
    fetch_volume_data,
    build_factor_signals,
    combine_factors,
    sector_neutral_signals,
)
from src.longshort_engine import (
    _aggregate_signals,
    run_longshort_from_signals,
)
from src.model import SIGNAL_MAP

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


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


def load_and_predict(model, model_name: str) -> pd.DataFrame:
    news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
    if not news_path.exists():
        raise FileNotFoundError(f"{news_path} not found")
    news_df = pd.read_csv(news_path, parse_dates=["date"])
    logger.info("Loaded %d headlines", len(news_df))

    predictor = CachedPredictor(model, model_name)
    if predictor.cache_size > 0:
        logger.info("Sentiment cache: %d entries pre-loaded", predictor.cache_size)

    news_df = predictor.predict_dataframe(news_df)
    news_df["date"]          = pd.to_datetime(news_df["date"]).dt.normalize()
    news_df["signal_normal"] = news_df["label_name"].map(SIGNAL_MAP).fillna(0)
    return news_df


def fetch_vix(start: str, end: str) -> pd.Series:
    """Return daily VIX closing values as a Series indexed by date."""
    if not _YF_AVAILABLE:
        return pd.Series(dtype=float)
    try:
        vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
        if vix.empty:
            return pd.Series(dtype=float)
        close = vix["Close"].squeeze()
        close.index = pd.to_datetime(close.index).normalize()
        close.name = "vix"
        return close
    except Exception as exc:
        logger.warning("VIX fetch failed: %s", exc)
        return pd.Series(dtype=float)


def apply_vol_targeting(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    lookback: int = 20,
    target_vol: float = 0.15,
) -> pd.DataFrame:
    """Rescale combined_score by inverse realized volatility.

    Position weight ∝ signal / realized_vol.  This reduces exposure in
    high-volatility names and increases it in stable names, producing a
    more stable portfolio-level volatility.

    Args:
        factor_df: Output of ``combine_factors`` with ``combined_score``.
        prices: Daily close prices (date × ticker).
        lookback: Rolling window for realized vol (default 20 trading days).
        target_vol: Annualised target volatility for scaling.

    Returns:
        factor_df with ``combined_score`` rescaled by vol target.
    """
    out = factor_df.copy()
    if prices.empty or "combined_score" not in out.columns:
        return out

    # Compute per-ticker realized vol: annualised std of daily log returns
    log_ret = np.log(prices / prices.shift(1))
    ann_vol = log_ret.rolling(lookback).std() * np.sqrt(252)

    # Align to factor_df rows
    def _get_vol(row):
        t = row["ticker"]
        d = row["date"]
        if t not in ann_vol.columns:
            return 1.0
        try:
            idx = ann_vol.index.searchsorted(d)
            if idx == 0:
                return 1.0
            v = ann_vol[t].iloc[idx - 1]
            return float(v) if pd.notna(v) and v > 0.01 else 1.0
        except Exception:
            return 1.0

    vols = out.apply(_get_vol, axis=1)
    out["combined_score"] = (out["combined_score"] * (target_vol / vols)).clip(-3, 3)
    return out


def apply_vix_filter(
    factor_df: pd.DataFrame,
    vix: pd.Series,
    vix_threshold: float = 30.0,
    reduction: float = 0.5,
) -> pd.DataFrame:
    """Halve long exposure when VIX > threshold (fear regime).

    When VIX spikes above 30, many quant strategies suffer from crowding
    and correlation breakdown.  This filter reduces net long exposure
    during high-fear periods while leaving short positions unchanged.

    Args:
        factor_df: Output of ``combine_factors``.
        vix: Daily VIX closing values (Series indexed by date).
        vix_threshold: VIX level triggering the filter (default 30).
        reduction: Fraction to keep on positive signals in high-VIX regime.

    Returns:
        factor_df with ``combined_score`` dampened on high-VIX days.
    """
    if vix.empty:
        return factor_df
    out = factor_df.copy()
    high_vix_dates = set(vix.index[vix > vix_threshold].normalize())

    def _scale(row):
        if row["date"] in high_vix_dates and row["combined_score"] > 0:
            return row["combined_score"] * reduction
        return row["combined_score"]

    out["combined_score"] = out.apply(_scale, axis=1)
    return out


def sf(val, fmt=".4f", na="  n/a "):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    return format(val, fmt)


# ---------------------------------------------------------------------------
# Weight grid (same as test_09)
# ---------------------------------------------------------------------------

WEIGHT_GRID = [
    (1.00, 0.00, 0.00, 0.00),
    (0.70, 0.30, 0.00, 0.00),
    (0.60, 0.00, 0.40, 0.00),
    (0.60, 0.00, 0.00, 0.40),
    (0.50, 0.30, 0.20, 0.00),
    (0.50, 0.20, 0.30, 0.00),
    (0.40, 0.30, 0.20, 0.10),
    (0.40, 0.20, 0.20, 0.20),
    (0.40, 0.40, 0.20, 0.00),
    (0.30, 0.50, 0.20, 0.00),
    (0.40, 0.20, 0.40, 0.00),
    (0.33, 0.33, 0.34, 0.00),
]

BACKTEST_KWARGS = dict(
    hold_days=2,
    quantile_cutoff=0.20,
    weighting="equal",
    tc_bps=config.TRANSACTION_COST_BPS,
    slip_bps=config.SLIPPAGE_BPS,
)


def weights_label(ws: tuple) -> str:
    s, m, v, l = ws
    parts = []
    if s: parts.append(f"S{int(s * 100)}")
    if m: parts.append(f"M{int(m * 100)}")
    if v: parts.append(f"V{int(v * 100)}")
    if l: parts.append(f"L{int(l * 100)}")
    return "+".join(parts)


# ---------------------------------------------------------------------------
# White's Reality Check
# ---------------------------------------------------------------------------


def _sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe of a per-period return series."""
    if len(returns) < 3 or returns.std() < 1e-10:
        return np.nan
    periods_per_year = 252 / BACKTEST_KWARGS["hold_days"]
    excess = returns - config.RISK_FREE_RATE / periods_per_year
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def whites_reality_check(
    returns_df: pd.DataFrame,
    n_boot: int = 2000,
    rng_seed: int = 42,
) -> dict:
    """White's (1999) Reality Check for data-mining bias.

    Tests whether the best Sharpe across all strategies is statistically
    better than what would arise by chance from testing many strategies.

    The bootstrap samples *returns_df* rows (rebalance periods) with
    replacement, recomputes each strategy's Sharpe, records the max,
    and builds a null distribution of the best attainable Sharpe.

    Args:
        returns_df: DataFrame where each column is a strategy's per-period
            return series (index = rebalance date).
        n_boot: Number of bootstrap resamples.
        rng_seed: Random seed for reproducibility.

    Returns:
        Dict with keys: best_strategy, best_sharpe, p_value, boot_mean,
        boot_std, boot_95pct.
    """
    rng = np.random.default_rng(rng_seed)
    T   = len(returns_df)

    # Observed Sharpes
    observed_sharpes = returns_df.apply(_sharpe)
    best_strategy    = observed_sharpes.idxmax()
    best_sharpe      = float(observed_sharpes.max())

    # Bootstrap null distribution of max Sharpe
    boot_max: list[float] = []
    for _ in range(n_boot):
        idx    = rng.integers(0, T, size=T)
        sample = returns_df.iloc[idx]
        boot_s = sample.apply(_sharpe)
        valid  = boot_s.dropna()
        if len(valid):
            boot_max.append(float(valid.max()))

    boot_arr   = np.array(boot_max)
    p_value    = float(np.mean(boot_arr >= best_sharpe))
    boot_mean  = float(boot_arr.mean())
    boot_std   = float(boot_arr.std())
    boot_95    = float(np.percentile(boot_arr, 95))

    return {
        "best_strategy": best_strategy,
        "best_sharpe":   round(best_sharpe, 4),
        "p_value":       round(p_value, 4),
        "boot_mean":     round(boot_mean, 4),
        "boot_std":      round(boot_std, 4),
        "boot_95pct":    round(boot_95, 4),
        "n_strategies":  len(returns_df.columns),
        "n_periods":     T,
        "n_boot":        n_boot,
    }


# ---------------------------------------------------------------------------
# Build signal variants
# ---------------------------------------------------------------------------


def build_signal_variants(
    pred_df: pd.DataFrame,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    vix: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """Return a dict of {variant_name -> factor_df} for each signal variant."""
    variants: dict[str, pd.DataFrame] = {}
    vol_arg = volumes if not volumes.empty else None

    # ── A: baseline (binary signal, no extras) ──────────────────────────────
    raw_base = _aggregate_signals(
        pred_df, "signal_normal", config.CONFIDENCE_THRESHOLD,
        confidence_weighted=False, max_headlines_per_day=0,
    )
    if not raw_base.empty:
        variants["A_baseline"] = build_factor_signals(raw_base, prices, vol_arg)

    # ── B: confidence-weighted signal ───────────────────────────────────────
    raw_conf = _aggregate_signals(
        pred_df, "signal_normal", config.CONFIDENCE_THRESHOLD,
        confidence_weighted=True, max_headlines_per_day=0,
    )
    if not raw_conf.empty:
        variants["B_conf_weighted"] = build_factor_signals(raw_conf, prices, vol_arg)

    # ── C: shock filter (cap 3 headlines/day/ticker) ─────────────────────────
    raw_shock = _aggregate_signals(
        pred_df, "signal_normal", config.CONFIDENCE_THRESHOLD,
        confidence_weighted=False, max_headlines_per_day=3,
    )
    if not raw_shock.empty:
        variants["C_shock_filter"] = build_factor_signals(raw_shock, prices, vol_arg)

    # ── D: sector-neutral (confidence-weighted + sector demean) ─────────────
    if not raw_conf.empty:
        raw_sector = sector_neutral_signals(raw_conf, signal_col="signal")
        variants["D_sector_neutral"] = build_factor_signals(raw_sector, prices, vol_arg)

    # ── E: combined — B + C + D ─────────────────────────────────────────────
    raw_all = _aggregate_signals(
        pred_df, "signal_normal", config.CONFIDENCE_THRESHOLD,
        confidence_weighted=True, max_headlines_per_day=3,
    )
    if not raw_all.empty:
        raw_all_sn = sector_neutral_signals(raw_all, signal_col="signal")
        variants["E_all_improvements"] = build_factor_signals(raw_all_sn, prices, vol_arg)

    # ── F: volatility targeting ──────────────────────────────────────────────
    # Uses baseline signals but rescales combined_score by 1/realized_vol
    # (applied after combine_factors in the strategy loop via a flag)
    if not raw_base.empty and not prices.empty:
        variants["F_vol_target"] = build_factor_signals(raw_base, prices, vol_arg)
        # Mark with metadata so strategy loop knows to apply vol-targeting
        variants["F_vol_target"].attrs["apply_vol_target"] = True

    # ── G: VIX regime filter ─────────────────────────────────────────────────
    if not raw_base.empty and vix is not None and not vix.empty:
        variants["G_vix_filter"] = build_factor_signals(raw_base, prices, vol_arg)
        variants["G_vix_filter"].attrs["vix"] = vix
    elif not raw_base.empty:
        logger.warning("VIX data unavailable — skipping G_vix_filter variant")

    logger.info("Built %d signal variants: %s", len(variants), list(variants.keys()))
    return variants


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vader",   action="store_true")
    parser.add_argument("--boot",    type=int, default=2000)
    args = parser.parse_args()

    model, model_name = load_model(args.vader)

    print(f"\n  AlphaLens -- White's Reality Check + Improvement Suite  [{model_name.upper()}]")

    # -- data -----------------------------------------------------------------
    pred_df = load_and_predict(model, model_name)
    start   = pred_df["date"].min().strftime("%Y-%m-%d")
    end     = (pred_df["date"].max() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    prices  = fetch_price_data(config.TICKERS, start=start, end=end)
    if prices.empty:
        print("ERROR: no price data")
        sys.exit(1)

    print(f"  Fetching volume data ...")
    volumes = fetch_volume_data(config.TICKERS, start=start, end=end)
    vol_ok  = not volumes.empty

    print(f"  Fetching VIX data ...")
    vix = fetch_vix(start, end)
    vix_ok = not vix.empty
    print(f"  VIX data: {'OK (%d days)' % len(vix) if vix_ok else 'unavailable (skipping G_vix_filter)'}")

    # =========================================================================
    # SECTION 1: Build all signal variants
    # =========================================================================
    print("\n  Building signal variants (A–G) ...")
    variants = build_signal_variants(
        pred_df, prices,
        volumes if vol_ok else pd.DataFrame(),
        vix=vix if vix_ok else None,
    )

    # =========================================================================
    # SECTION 2: Run full weight grid × all variants, collect period returns
    # =========================================================================
    print(f"\n  Running {len(WEIGHT_GRID)} weight combos × {len(variants)} signal variants "
          f"= {len(WEIGHT_GRID) * len(variants)} strategies (A–G × 12 weight grids) ...")

    all_returns:  dict[str, pd.Series] = {}  # strategy_label -> period returns
    all_metrics:  list[dict]           = []

    for var_name, factor_df in variants.items():
        if factor_df.empty:
            logger.warning("Variant %s produced empty factor_df, skipping", var_name)
            continue

        # Extract variant-level flags stored in attrs
        _apply_vol  = factor_df.attrs.get("apply_vol_target", False)
        _vix_series = factor_df.attrs.get("vix", None)

        for ws in WEIGHT_GRID:
            w_dict = {"sentiment": ws[0], "momentum": ws[1],
                      "volatility": ws[2], "liquidity": ws[3]}
            combined = combine_factors(factor_df, weights=w_dict)

            # Apply F: volatility targeting (rescale by 1/realized_vol)
            if _apply_vol and not prices.empty:
                combined = apply_vol_targeting(combined, prices)

            # Apply G: VIX regime filter (halve longs when VIX > 30)
            if _vix_series is not None:
                combined = apply_vix_filter(combined, _vix_series)

            label = f"{var_name}|{weights_label(ws)}"

            try:
                ret_df, metrics = run_longshort_from_signals(
                    combined, prices, signal_col="combined_score",
                    **BACKTEST_KWARGS,
                )
            except Exception as exc:
                logger.warning("Strategy %s failed: %s", label, exc)
                continue

            metrics["label"]   = label
            metrics["variant"] = var_name
            metrics["weights"] = weights_label(ws)
            all_metrics.append(metrics)

            # Collect period-level net returns for White's RC
            if not ret_df.empty:
                period_pnl = (
                    ret_df.groupby("rebalance_date")["ls_return"]
                    .sum()
                    .rename(label)
                )
                all_returns[label] = period_pnl

    print(f"  Collected {len(all_returns)} strategy return series")

    # =========================================================================
    # SECTION 3: Summary table per variant (best weight per variant)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  SECTION 3 -- Best config per signal variant")
    print("=" * 80)

    COLS = [("Sharpe", "sharpe_ratio", "+.3f"), ("Ann.Ret", "annualised_return", "+.1%"),
            ("MaxDD",  "max_drawdown", ".1%"),   ("HitRate", "hit_rate", ".3f"),
            ("IC",     "ic", "+.4f")]

    print(f"  {'Variant':<25} {'Weights':<15}" +
          "".join(f" {c[0]:>9}" for c in COLS))
    print("  " + "-" * 75)

    best_by_variant: dict[str, dict] = {}
    for var_name in variants:
        var_metrics = [m for m in all_metrics if m.get("variant") == var_name]
        if not var_metrics:
            continue
        best = max(var_metrics, key=lambda m: m.get("sharpe_ratio", -999))
        best_by_variant[var_name] = best
        vals = "".join(f" {sf(best.get(k, np.nan), fmt):>9}" for _, k, fmt in COLS)
        print(f"  {var_name:<25} {best.get('weights', '?'):<15}{vals}")

    # =========================================================================
    # SECTION 4: White's Reality Check
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"  SECTION 4 -- White's Reality Check  ({args.boot} bootstrap samples)")
    print("=" * 80)

    if len(all_returns) < 2:
        print("  ERROR: need at least 2 strategies for White's RC")
    else:
        # Align all return series to common index (fill missing periods with 0)
        returns_df = pd.DataFrame(all_returns).fillna(0.0)
        returns_df = returns_df.sort_index()

        print(f"\n  Strategies: {len(returns_df.columns)}")
        print(f"  Periods   : {len(returns_df)}")
        print(f"  Bootstrapping ... ", end="", flush=True)

        rc = whites_reality_check(returns_df, n_boot=args.boot)
        print("done.")

        print(f"\n  WHITE'S REALITY CHECK RESULTS")
        print(f"  {'Best strategy':<30}: {rc['best_strategy']}")
        print(f"  {'Observed Sharpe':<30}: {rc['best_sharpe']:+.4f}")
        print(f"  {'Bootstrap mean Sharpe':<30}: {rc['boot_mean']:+.4f}")
        print(f"  {'Bootstrap 95th pct':<30}: {rc['boot_95pct']:+.4f}")
        print(f"  {'Bootstrap p-value':<30}: {rc['p_value']:.4f}")

        if rc["p_value"] < 0.05:
            verdict = "PASS -- Strategy survives data-mining bias (p < 0.05)."
        elif rc["p_value"] < 0.10:
            verdict = "BORDERLINE -- Weak evidence against data-mining (0.05 < p < 0.10)."
        else:
            verdict = "FAIL -- Sharpe may be selection bias (p > 0.10)."
        print(f"\n  Verdict: {verdict}")

        # Save returns
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        ret_path = config.METRICS_DIR / f"whites_rc_returns_{model_name}.csv"
        returns_df.to_csv(ret_path)
        print(f"\n  Strategy returns saved -> {ret_path}")

        # Save summary
        rc_summary = pd.DataFrame([{**rc, "model": model_name}])
        rc_summary.to_csv(config.METRICS_DIR / f"whites_rc_results_{model_name}.csv", index=False)

    # =========================================================================
    # SECTION 5: Improvement delta table (vs baseline)
    # =========================================================================
    if "A_baseline" in best_by_variant:
        base_sharpe = best_by_variant["A_baseline"].get("sharpe_ratio", np.nan)
        print("\n" + "=" * 80)
        print("  SECTION 5 -- Improvement Delta vs Baseline")
        print("=" * 80)
        print(f"  {'Variant':<25} {'Best Weights':<15} {'Sharpe':>9} {'Delta':>9} {'Verdict':>20}")
        print("  " + "-" * 80)
        for var_name, m in best_by_variant.items():
            s = m.get("sharpe_ratio", np.nan)
            delta = s - base_sharpe if not (np.isnan(s) or np.isnan(base_sharpe)) else np.nan
            if np.isnan(delta):
                verdict = "n/a"
            elif delta > 0.10:
                verdict = "IMPROVEMENT"
            elif delta > -0.05:
                verdict = "neutral"
            else:
                verdict = "HURTS"
            print(f"  {var_name:<25} {m.get('weights', '?'):<15} "
                  f"{sf(s, '+.3f'):>9} {sf(delta, '+.3f'):>9} {verdict:>20}")

    # =========================================================================
    # Save all metrics
    # =========================================================================
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = config.METRICS_DIR / f"whites_rc_all_strategies_{model_name}.csv"
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    print(f"\n  All strategy metrics saved -> {metrics_path}")

    # =========================================================================
    # Verdict
    # =========================================================================
    print("\n" + "=" * 80)
    print("  FINAL VERDICT")
    print("=" * 80)

    all_sharpes = [(m["label"], m.get("sharpe_ratio", np.nan)) for m in all_metrics]
    all_sharpes_clean = [(l, s) for l, s in all_sharpes if not np.isnan(s)]
    if all_sharpes_clean:
        overall_best_label, overall_best_sharpe = max(all_sharpes_clean, key=lambda x: x[1])
        print(f"\n  Overall best strategy : {overall_best_label}")
        print(f"  Overall best Sharpe   : {overall_best_sharpe:+.3f}")
        if "rc" in dir() and isinstance(rc, dict):
            print(f"  White's RC p-value    : {rc['p_value']:.4f}")
            conclusion = (
                "The signal is statistically validated — it is NOT an artefact of testing many strategies."
                if rc["p_value"] < 0.05
                else "The signal requires more data or wider universe to be statistically confirmed."
            )
            print(f"\n  {conclusion}")
    print()


if __name__ == "__main__":
    main()
