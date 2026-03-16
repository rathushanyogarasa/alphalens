"""Cross-sectional long-short backtesting engine for AlphaLens.

Each rebalance day the universe is scored, ranked by aggregated sentiment
signal, and divided into quantile buckets.  The top bucket is held long,
the bottom bucket held short, for a configurable number of trading days.
Positions are equal-weighted within each book.

This converts a weak IC (~0.02) into a meaningful Sharpe by exploiting the
cross-sectional spread rather than trying to time individual stocks.

Key design
----------
- **No look-ahead**: signals are generated from headlines on day T; positions
  enter at the next-available price (open of day T+1 in practice, proxied
  here as close of day T; returns measured from close T to close T+hold_days).
- **Rebalance frequency**: every `hold_days` trading days (non-overlapping).
- **Cost**: round-trip (tc_bps + slip_bps) charged once per position on entry.
- **Equal weight**: 1/N_long per long, 1/N_short per short.
- **Net return**: (long book return - short book return) / 2, so the strategy
  is self-financing and dollar-neutral.
"""

import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.model import SIGNAL_MAP
from src.plot_style import GOLD, NAVY, FIG_DPI, apply_plot_style

logger = logging.getLogger(__name__)
_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Signal aggregation (reused from backtest, kept local to avoid circular import)
# ---------------------------------------------------------------------------


def _aggregate_signals(
    pred_df: pd.DataFrame,
    signal_col: str,
    confidence_threshold: float,
    confidence_weighted: bool = True,
    max_headlines_per_day: int = 3,
) -> pd.DataFrame:
    """Filter by confidence and aggregate headlines to (date, ticker) level.

    Args:
        pred_df: Raw predictions DataFrame with at least ``date``,
            ``ticker``, ``confidence``, and *signal_col* columns.
        signal_col: Column holding the per-headline signal value (±1/0).
        confidence_threshold: Minimum confidence to include a headline.
        confidence_weighted: If True (default), scale each headline's
            signal by its confidence before averaging.  A headline with
            confidence 0.95 contributes 0.95× as much as one at 1.0,
            vs a borderline 0.71 headline contributing 0.71×.  This
            extracts more information from FinBERT's probability output.
        max_headlines_per_day: Cap on headlines per (ticker, date) kept
            before aggregation — sorted by confidence descending.
            Set to 0 to disable.  Prevents headline-spam days from
            dominating the signal (news shock filter).

    Returns:
        DataFrame with columns ``date``, ``ticker``, ``signal``,
        ``headline_count``.
    """
    confident = pred_df[pred_df["confidence"] >= confidence_threshold].copy()
    if confident.empty:
        return pd.DataFrame(columns=["date", "ticker", "signal"])

    # News shock filter: keep top-N highest-confidence headlines per (date, ticker)
    if max_headlines_per_day > 0 and "confidence" in confident.columns:
        confident = (
            confident
            .sort_values("confidence", ascending=False)
            .groupby(["date", "ticker"], group_keys=False)
            .head(max_headlines_per_day)
        )

    # Confidence weighting: scale signal by confidence so high-certainty
    # headlines dominate and borderline ones contribute proportionally less
    if confidence_weighted and "confidence" in confident.columns:
        confident = confident.copy()
        confident["_weighted_signal"] = confident[signal_col] * confident["confidence"]
        agg_col = "_weighted_signal"
    else:
        agg_col = signal_col

    return (
        confident.groupby(["date", "ticker"])
        .agg(
            signal=(agg_col, "mean"),
            headline_count=(signal_col, "count"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Volatility helpers
# ---------------------------------------------------------------------------


def _realized_vol(price_sorted: pd.DataFrame, ticker: str, entry_pos: int, window: int = 20) -> float:
    """20-day realized volatility for one ticker, annualised."""
    start = max(0, entry_pos - window)
    slice_ = price_sorted[ticker].iloc[start:entry_pos]
    if len(slice_) < 3:
        return 1.0
    rets = slice_.pct_change().dropna()
    std  = float(rets.std())
    return std * math.sqrt(_TRADING_DAYS) if std > 1e-8 else 1.0


def _inverse_vol_weights(active: pd.DataFrame, price_sorted: pd.DataFrame, entry_pos: int) -> dict[str, float]:
    """Return a dict {ticker: weight} using inverse-volatility sizing.

    Weights are computed per book (longs separately from shorts) so that each
    book sums to 1.0 before the long/short netting step.

    Returns raw inverse-vol weights (not yet split by book direction).
    """
    weights: dict[str, float] = {}
    for _, r in active.iterrows():
        ticker = str(r["ticker"])
        vol    = _realized_vol(price_sorted, ticker, entry_pos)
        weights[ticker] = 1.0 / max(vol, 1e-6)

    # Normalise within each direction separately
    longs  = {t: w for t, w in weights.items()
               if int(active.loc[active["ticker"] == t, "position"].values[0]) > 0}
    shorts = {t: w for t, w in weights.items()
               if int(active.loc[active["ticker"] == t, "position"].values[0]) < 0}

    def _norm(d: dict) -> dict:
        total = sum(d.values()) or 1.0
        return {k: v / total for k, v in d.items()}

    return {**_norm(longs), **_norm(shorts)}


# ---------------------------------------------------------------------------
# VIX helper
# ---------------------------------------------------------------------------


def _fetch_vix_series(start: str, end: str) -> pd.Series:
    """Fetch VIX daily close prices from yfinance.  Returns empty series on failure."""
    try:
        import yfinance as yf
        raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            return pd.Series(dtype=float)
        if isinstance(raw.columns, pd.MultiIndex):
            col = raw.columns.get_level_values(0)[0]
            vix = raw[col].squeeze()
        else:
            vix = raw.iloc[:, 0]
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        return vix
    except Exception as exc:
        logger.warning("VIX fetch failed: %s", exc)
        return pd.Series(dtype=float)


def _get_vix_at(vix_series: pd.Series, date: pd.Timestamp) -> float:
    """Return the most recent VIX reading on or before *date*."""
    if vix_series.empty:
        return 20.0
    candidates = vix_series[vix_series.index <= date]
    return float(candidates.iloc[-1]) if len(candidates) > 0 else 20.0


# ---------------------------------------------------------------------------
# Cross-sectional ranking
# ---------------------------------------------------------------------------


def rank_cross_section(
    signals_day: pd.DataFrame,
    quantile_cutoff: float,
) -> pd.DataFrame:
    """Assign long/short/neutral positions for a single day's signal cross-section.

    Args:
        signals_day: Rows for one date with ``ticker`` and ``signal`` columns.
        quantile_cutoff: Fraction of universe to trade on each side.
            0.20 = top 20% long, bottom 20% short.

    Returns:
        DataFrame with ``ticker``, ``signal``, ``position`` (+1/-1/0) columns.
    """
    df = signals_day.copy().reset_index(drop=True)
    n = len(df)
    if n < 4:
        df["position"] = 0
        return df

    n_each = max(1, int(math.floor(n * quantile_cutoff)))
    df = df.sort_values("signal", ascending=False).reset_index(drop=True)

    df["position"] = 0
    df.loc[:n_each - 1, "position"] = 1        # top quintile: long
    df.loc[n - n_each:, "position"] = -1       # bottom quintile: short

    return df


# ---------------------------------------------------------------------------
# Long-short backtest
# ---------------------------------------------------------------------------


def run_longshort_backtest(
    pred_df: pd.DataFrame,
    prices: pd.DataFrame,
    hold_days: int | None = None,
    quantile_cutoff: float | None = None,
    confidence_threshold: float | None = None,
    tc_bps: float | None = None,
    slip_bps: float | None = None,
    signal_col: str = "signal_normal",
    weighting: str = "equal",
    macro_vix_filter: bool = False,
    vix_threshold: float | None = None,
    vix_exposure_scale: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Run a cross-sectional long-short backtest.

    Args:
        pred_df: DataFrame with ``date``, ``ticker``, ``confidence``, and
            a signal column (``signal_normal`` by default).
        prices: Wide adjusted-close price DataFrame indexed by date.
        hold_days: Holding period in trading days.  Defaults to
            ``config.LONGSHORT_HOLD_DAYS``.
        quantile_cutoff: Top/bottom fraction to trade.  Defaults to
            ``config.LONGSHORT_QUANTILE_CUTOFF``.
        confidence_threshold: Confidence filter.  Defaults to
            ``config.CONFIDENCE_THRESHOLD``.
        tc_bps: Transaction cost basis points.  Defaults to
            ``config.TRANSACTION_COST_BPS``.
        slip_bps: Slippage basis points.  Defaults to
            ``config.SLIPPAGE_BPS``.
        signal_col: Column in ``pred_df`` holding the numeric signal.
        weighting: ``"equal"`` (default) or ``"vol_scaled"`` — use
            inverse-volatility weights instead of equal-weighting within each book.
        macro_vix_filter: If ``True``, fetch the VIX time series and halve
            gross exposure on days when VIX exceeds ``vix_threshold``.
        vix_threshold: VIX level above which exposure is reduced.  Defaults to
            ``config.MACRO_REGIME_VIX_RISK_OFF`` (28.0).
        vix_exposure_scale: Fraction of normal position size taken when VIX
            is elevated (default 0.5 = half size).

    Returns:
        tuple:
            - ``returns_df``: One row per (rebalance_date, ticker) with
              ``position``, ``weight``, ``period_return``, ``ls_return``,
              ``cost_drag``, ``vix_level``.
            - ``metrics``: Dict of portfolio-level performance statistics,
              including ``avg_gross_exposure``, ``avg_turnover``, and
              ``avg_positions_per_side``.
    """
    hold_days       = hold_days         or getattr(config, "LONGSHORT_HOLD_DAYS", 2)
    quantile_cutoff = quantile_cutoff   or getattr(config, "LONGSHORT_QUANTILE_CUTOFF", 0.20)
    conf_thresh     = confidence_threshold or config.CONFIDENCE_THRESHOLD
    tc              = tc_bps   if tc_bps   is not None else config.TRANSACTION_COST_BPS
    slip            = slip_bps if slip_bps is not None else config.SLIPPAGE_BPS
    total_cost      = (tc + slip) / 10_000.0
    vix_thresh      = vix_threshold or getattr(config, "MACRO_REGIME_VIX_RISK_OFF", 28.0)

    signals = _aggregate_signals(pred_df, signal_col, conf_thresh)
    if signals.empty:
        logger.warning("No signals after confidence filter — returning empty results")
        return pd.DataFrame(), {}

    # Need at least 2 tickers per rebalance date to form a long AND short leg.
    # Calling rank_cross_section on 1-3 rows produces degenerate quantiles and
    # fires spurious "No completed long-short positions" warnings.
    min_tickers_required = max(
        4, math.ceil(2 / max(getattr(config, "LONGSHORT_QUANTILE_CUTOFF", 0.20), 0.01))
    )
    ticker_counts = signals.groupby("date")["ticker"].nunique()
    usable_dates = ticker_counts[ticker_counts >= min_tickers_required].index
    if usable_dates.empty:
        logger.debug(
            "run_longshort_backtest: no dates have >= %d tickers after confidence "
            "filter (max was %d). Dataset too small for long-short — skipping.",
            min_tickers_required,
            int(ticker_counts.max()) if not ticker_counts.empty else 0,
        )
        return pd.DataFrame(), {}
    dropped = len(ticker_counts) - len(usable_dates)
    if dropped > 0:
        logger.info(
            "Long-short: dropped %d/%d dates with fewer than %d tickers",
            dropped, len(ticker_counts), min_tickers_required,
        )
    signals = signals[signals["date"].isin(usable_dates)]

    signals["date"] = pd.to_datetime(signals["date"])
    price_sorted    = prices.sort_index()
    all_dates       = sorted(signals["date"].unique())

    # Optionally fetch VIX series for macro filter
    vix_series: pd.Series = pd.Series(dtype=float)
    if macro_vix_filter:
        start_str = all_dates[0].strftime("%Y-%m-%d")
        end_str   = (all_dates[-1] + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        vix_series = _fetch_vix_series(start_str, end_str)
        logger.info("VIX filter enabled (threshold=%.1f) | %d VIX observations",
                    vix_thresh, len(vix_series))

    # Non-overlapping rebalance dates spaced `hold_days` apart
    rebalance_dates: list[pd.Timestamp] = []
    last = None
    for d in all_dates:
        if last is None:
            rebalance_dates.append(d)
            last = d
        else:
            trading_days_gap = len(pd.bdate_range(last, d)) - 1
            if trading_days_gap >= hold_days:
                rebalance_dates.append(d)
                last = d

    rows: list[dict] = []
    prev_active_tickers: set[str] = set()
    turnover_list: list[float] = []
    gross_exp_list: list[float] = []

    for rdate in rebalance_dates:
        day_signals = signals[signals["date"] == rdate][["ticker", "signal"]].copy()
        day_signals = day_signals[day_signals["ticker"].isin(price_sorted.columns)]
        if len(day_signals) < 4:
            continue

        ranked = rank_cross_section(day_signals, quantile_cutoff)
        active = ranked[ranked["position"] != 0].copy()
        if active.empty:
            continue

        entry_pos = price_sorted.index.searchsorted(rdate)
        exit_pos  = entry_pos + hold_days
        if entry_pos >= len(price_sorted) or exit_pos >= len(price_sorted):
            continue

        entry_date = price_sorted.index[entry_pos]
        exit_date  = price_sorted.index[exit_pos]

        # VIX exposure scaling
        vix_level   = _get_vix_at(vix_series, rdate)
        exposure_scale = vix_exposure_scale if (macro_vix_filter and vix_level > vix_thresh) else 1.0

        # Position weights
        if weighting == "vol_scaled":
            inv_vol_w = _inverse_vol_weights(active, price_sorted, entry_pos)
        else:
            # Equal weight within each book direction
            n_long  = (active["position"] > 0).sum()
            n_short = (active["position"] < 0).sum()
            inv_vol_w = {}
            for _, r in active.iterrows():
                ticker = str(r["ticker"])
                pos    = int(r["position"])
                inv_vol_w[ticker] = (1.0 / n_long if pos > 0 else 1.0 / n_short) if (n_long > 0 and n_short > 0) else 0.5

        # Turnover vs previous rebalance
        cur_tickers = set(active["ticker"])
        if prev_active_tickers:
            changed = len(cur_tickers.symmetric_difference(prev_active_tickers))
            turnover = changed / max(len(cur_tickers), len(prev_active_tickers))
            turnover_list.append(turnover)
        prev_active_tickers = cur_tickers

        # Gross exposure this period (sum of abs weights × exposure_scale)
        gross_exp = exposure_scale * sum(abs(w) for w in inv_vol_w.values())
        gross_exp_list.append(gross_exp)

        for _, r in active.iterrows():
            ticker   = str(r["ticker"])
            position = int(r["position"])
            weight   = inv_vol_w.get(ticker, 0.0) * exposure_scale
            if weight == 0.0 or ticker not in price_sorted.columns:
                continue
            p_entry = price_sorted[ticker].iloc[entry_pos]
            p_exit  = price_sorted[ticker].iloc[exit_pos]
            if pd.isna(p_entry) or pd.isna(p_exit) or p_entry == 0:
                continue

            period_ret = float((p_exit - p_entry) / p_entry)
            cost_drag  = total_cost * weight
            ls_ret     = position * weight * period_ret - cost_drag

            rows.append({
                "rebalance_date": rdate,
                "entry_date":     entry_date,
                "exit_date":      exit_date,
                "ticker":         ticker,
                "signal":         float(r["signal"]),
                "position":       position,
                "weight":         round(weight, 6),
                "period_return":  round(period_ret, 6),
                "ls_return":      round(ls_ret, 6),
                "cost_drag":      round(cost_drag, 6),
                "vix_level":      round(vix_level, 2),
            })

    if not rows:
        logger.warning("No completed long-short positions — returning empty results")
        return pd.DataFrame(), {}

    returns_df = pd.DataFrame(rows).sort_values("rebalance_date").reset_index(drop=True)

    # Net L/S return per rebalance period (weighted sum, already dollar-neutral)
    period_pnl = (
        returns_df.groupby("rebalance_date")["ls_return"]
        .sum()
        .rename("net_return")
        .reset_index()
    )

    metrics = _compute_metrics(period_pnl["net_return"], returns_df, hold_days)
    metrics["rebalance_periods"]     = len(rebalance_dates)
    metrics["hold_days"]             = hold_days
    metrics["quantile_cutoff"]       = quantile_cutoff
    metrics["weighting"]             = weighting
    metrics["macro_vix_filter"]      = macro_vix_filter
    metrics["avg_gross_exposure"]    = round(float(np.mean(gross_exp_list)), 4) if gross_exp_list else np.nan
    metrics["avg_turnover"]          = round(float(np.mean(turnover_list)), 4) if turnover_list else np.nan
    metrics["avg_positions_per_side"] = round(
        float(returns_df.groupby("rebalance_date").apply(lambda g: (g["position"] == 1).sum()).mean()), 1
    )

    logger.info(
        "L/S backtest [%s%s]: %d rebalances | hold=%dd | q=%.0f%% | "
        "Sharpe=%.3f | ann_ret=%.2f%% | turnover=%.2f",
        weighting,
        "+VIX" if macro_vix_filter else "",
        len(rebalance_dates), hold_days, quantile_cutoff * 100,
        metrics.get("sharpe_ratio", 0),
        metrics.get("annualised_return", 0) * 100,
        metrics.get("avg_turnover", 0) or 0,
    )
    return returns_df, metrics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(net_returns: pd.Series, returns_df: pd.DataFrame, hold_days: int) -> dict:
    """Compute annualised performance metrics from the per-period net L/S returns."""
    if net_returns.empty:
        return {}

    # Annualisation factor: periods per year
    periods_per_year = _TRADING_DAYS / hold_days
    n = len(net_returns)

    total_ret  = float((1 + net_returns).prod() - 1)
    ann_ret    = float((1 + total_ret) ** (periods_per_year / max(n, 1)) - 1)
    ann_vol    = float(net_returns.std() * math.sqrt(periods_per_year))

    daily_rf   = config.RISK_FREE_RATE / _TRADING_DAYS
    period_rf  = config.RISK_FREE_RATE / periods_per_year
    excess     = net_returns - period_rf
    sharpe     = float(excess.mean() / excess.std() * math.sqrt(periods_per_year)) if excess.std() > 0 else 0.0

    downside   = net_returns[net_returns < 0]
    down_dev   = float(downside.std() * math.sqrt(periods_per_year)) if len(downside) > 1 else 1e-9
    sortino    = float((ann_ret - config.RISK_FREE_RATE) / down_dev) if down_dev > 0 else 0.0

    cum        = (1 + net_returns).cumprod()
    rolling_mx = cum.cummax()
    drawdown   = (cum - rolling_mx) / rolling_mx
    max_dd     = float(drawdown.min())
    calmar     = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

    win_rate   = float((net_returns > 0).mean())

    # IC: Spearman between signal and realised period return per position
    sub = returns_df.copy()
    ic, p = (np.nan, np.nan)
    if len(sub) >= 10:
        try:
            ic, p = stats.spearmanr(sub["signal"], sub["period_return"])
            ic, p = float(ic), float(p)
        except Exception:
            pass

    # Hit rate of active positions
    active  = returns_df[returns_df["position"] != 0]
    hit_rate = float((
        ((active["position"] > 0) & (active["period_return"] > 0)) |
        ((active["position"] < 0) & (active["period_return"] < 0))
    ).mean()) if len(active) > 0 else np.nan

    return {
        "total_return":         round(total_ret, 4),
        "annualised_return":    round(ann_ret, 4),
        "annualised_volatility": round(ann_vol, 4),
        "sharpe_ratio":         round(sharpe, 4),
        "sortino_ratio":        round(sortino, 4),
        "calmar_ratio":         round(calmar, 4),
        "max_drawdown":         round(max_dd, 4),
        "win_rate":             round(win_rate, 4),
        "ic":                   round(ic, 4) if not np.isnan(ic) else np.nan,
        "ic_p_value":           round(p, 4)  if not np.isnan(p)  else np.nan,
        "hit_rate":             round(hit_rate, 4) if not np.isnan(hit_rate) else np.nan,
        "n_periods":            n,
    }


# ---------------------------------------------------------------------------
# IC by lag (used for horizon comparison)
# ---------------------------------------------------------------------------


def run_longshort_from_signals(
    signals_df: pd.DataFrame,
    prices: pd.DataFrame,
    signal_col: str = "signal",
    hold_days: int | None = None,
    quantile_cutoff: float | None = None,
    tc_bps: float | None = None,
    slip_bps: float | None = None,
    weighting: str = "equal",
    macro_vix_filter: bool = False,
    vix_threshold: float | None = None,
    vix_exposure_scale: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Run the long-short backtest from pre-aggregated (date, ticker) signals.

    Identical to :func:`run_longshort_backtest` but accepts a signals DataFrame
    directly (e.g. from :mod:`src.factor_engine`) instead of raw predictions.
    The ``signal_col`` column is used for cross-sectional ranking.

    Args:
        signals_df: DataFrame with ``date``, ``ticker``, and at least one
            numeric signal column (named by ``signal_col``).
        prices: Wide adjusted-close price DataFrame indexed by date.
        signal_col: Column to rank on.  Must exist in ``signals_df``.
        hold_days, quantile_cutoff, tc_bps, slip_bps, weighting,
        macro_vix_filter, vix_threshold, vix_exposure_scale:
            See :func:`run_longshort_backtest`.

    Returns:
        Same tuple as :func:`run_longshort_backtest`.
    """
    hold_days       = hold_days       or getattr(config, "LONGSHORT_HOLD_DAYS", 2)
    quantile_cutoff = quantile_cutoff or getattr(config, "LONGSHORT_QUANTILE_CUTOFF", 0.20)
    tc              = tc_bps   if tc_bps   is not None else config.TRANSACTION_COST_BPS
    slip            = slip_bps if slip_bps is not None else config.SLIPPAGE_BPS
    total_cost      = (tc + slip) / 10_000.0
    vix_thresh      = vix_threshold or getattr(config, "MACRO_REGIME_VIX_RISK_OFF", 28.0)

    if signals_df.empty or signal_col not in signals_df.columns:
        logger.warning("run_longshort_from_signals: empty or missing column '%s'", signal_col)
        return pd.DataFrame(), {}

    # Overwrite "signal" with the chosen column so rank_cross_section works uniformly
    signals = signals_df.copy()
    signals["date"]   = pd.to_datetime(signals["date"])
    signals["signal"] = signals[signal_col]  # safe even when signal_col == "signal"

    price_sorted = prices.sort_index()
    all_dates    = sorted(signals["date"].unique())

    vix_series: pd.Series = pd.Series(dtype=float)
    if macro_vix_filter:
        start_str = all_dates[0].strftime("%Y-%m-%d")
        end_str   = (all_dates[-1] + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        vix_series = _fetch_vix_series(start_str, end_str)

    # Non-overlapping rebalance schedule
    rebalance_dates: list[pd.Timestamp] = []
    last = None
    for d in all_dates:
        if last is None:
            rebalance_dates.append(d)
            last = d
        else:
            gap = len(pd.bdate_range(last, d)) - 1
            if gap >= hold_days:
                rebalance_dates.append(d)
                last = d

    rows: list[dict] = []
    prev_tickers: set[str] = set()
    turnover_list: list[float] = []
    gross_exp_list: list[float] = []

    for rdate in rebalance_dates:
        day_signals = signals[signals["date"] == rdate][["ticker", "signal"]].copy()
        day_signals = day_signals[day_signals["ticker"].isin(price_sorted.columns)]
        if len(day_signals) < 4:
            continue

        ranked = rank_cross_section(day_signals, quantile_cutoff)
        active = ranked[ranked["position"] != 0].copy()
        if active.empty:
            continue

        entry_pos = price_sorted.index.searchsorted(rdate)
        exit_pos  = entry_pos + hold_days
        if entry_pos >= len(price_sorted) or exit_pos >= len(price_sorted):
            continue

        entry_date = price_sorted.index[entry_pos]
        exit_date  = price_sorted.index[exit_pos]

        vix_level      = _get_vix_at(vix_series, rdate)
        exposure_scale = vix_exposure_scale if (macro_vix_filter and vix_level > vix_thresh) else 1.0

        if weighting == "vol_scaled":
            inv_vol_w = _inverse_vol_weights(active, price_sorted, entry_pos)
        else:
            n_long  = (active["position"] > 0).sum()
            n_short = (active["position"] < 0).sum()
            inv_vol_w = {}
            for _, r in active.iterrows():
                pos = int(r["position"])
                inv_vol_w[str(r["ticker"])] = (
                    (1.0 / n_long if pos > 0 else 1.0 / n_short)
                    if (n_long > 0 and n_short > 0) else 0.5
                )

        cur_tickers = set(active["ticker"])
        if prev_tickers:
            changed  = len(cur_tickers.symmetric_difference(prev_tickers))
            turnover_list.append(changed / max(len(cur_tickers), len(prev_tickers)))
        prev_tickers = cur_tickers

        gross_exp_list.append(exposure_scale * sum(abs(w) for w in inv_vol_w.values()))

        for _, r in active.iterrows():
            ticker   = str(r["ticker"])
            position = int(r["position"])
            weight   = inv_vol_w.get(ticker, 0.0) * exposure_scale
            if weight == 0.0 or ticker not in price_sorted.columns:
                continue
            p_entry = price_sorted[ticker].iloc[entry_pos]
            p_exit  = price_sorted[ticker].iloc[exit_pos]
            if pd.isna(p_entry) or pd.isna(p_exit) or p_entry == 0:
                continue
            period_ret = float((p_exit - p_entry) / p_entry)
            cost_drag  = total_cost * weight
            rows.append({
                "rebalance_date": rdate,
                "entry_date":     entry_date,
                "exit_date":      exit_date,
                "ticker":         ticker,
                "signal":         float(r["signal"]),
                "position":       position,
                "weight":         round(weight, 6),
                "period_return":  round(period_ret, 6),
                "ls_return":      round(position * weight * period_ret - cost_drag, 6),
                "cost_drag":      round(cost_drag, 6),
            })

    if not rows:
        return pd.DataFrame(), {}

    returns_df = pd.DataFrame(rows).sort_values("rebalance_date").reset_index(drop=True)
    period_pnl = (returns_df.groupby("rebalance_date")["ls_return"]
                  .sum().rename("net_return").reset_index())

    metrics = _compute_metrics(period_pnl["net_return"], returns_df, hold_days)
    metrics.update({
        "rebalance_periods":      len(rebalance_dates),
        "hold_days":              hold_days,
        "quantile_cutoff":        quantile_cutoff,
        "weighting":              weighting,
        "signal_col":             signal_col,
        "avg_gross_exposure":     round(float(np.mean(gross_exp_list)), 4) if gross_exp_list else np.nan,
        "avg_turnover":           round(float(np.mean(turnover_list)), 4) if turnover_list else np.nan,
        "avg_positions_per_side": round(
            float(returns_df.groupby("rebalance_date")
                  .apply(lambda g: (g["position"] == 1).sum()).mean()), 1),
    })
    logger.info(
        "L/S [%s]: %d rebalances | hold=%dd | Sharpe=%.3f | ann=%.1f%%",
        signal_col, len(rebalance_dates), hold_days,
        metrics.get("sharpe_ratio", 0),
        metrics.get("annualised_return", 0) * 100,
    )
    return returns_df, metrics


def compute_ls_ic_decay(
    pred_df: pd.DataFrame,
    prices: pd.DataFrame,
    lags: list[int] | None = None,
    confidence_threshold: float | None = None,
    signal_col: str = "signal_normal",
) -> pd.DataFrame:
    """Compute cross-sectional IC at multiple forward lags.

    Uses the full (overlapping) signal set — each day's signals are paired
    with forward returns at each lag.  This gives more data points than the
    non-overlapping backtest and is the standard way to measure IC decay.

    Returns:
        DataFrame with columns ``lag``, ``ic``, ``p_value``, ``significant``, ``n``.
    """
    conf_thresh = confidence_threshold or config.CONFIDENCE_THRESHOLD
    lags        = lags or [1, 2, 3, 5, 10]
    signals     = _aggregate_signals(pred_df, signal_col, conf_thresh)
    price_sorted = prices.sort_index()

    rows: list[dict] = []
    for lag in lags:
        pairs: list[dict] = []
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
            pairs.append({"signal": float(row["signal"]), "fwd": float((p_lag - p0) / p0)})

        if len(pairs) < 10:
            rows.append({"lag": lag, "ic": np.nan, "p_value": np.nan, "significant": False, "n": len(pairs)})
            continue

        df      = pd.DataFrame(pairs)
        ic, p   = stats.spearmanr(df["signal"], df["fwd"])
        rows.append({
            "lag":         lag,
            "ic":          round(float(ic), 4),
            "p_value":     round(float(p), 4),
            "significant": bool(p < 0.05),
            "n":           len(pairs),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_longshort_equity_curve(
    returns_df: pd.DataFrame,
    metrics: dict,
    label: str = "longshort",
) -> None:
    """Plot cumulative net L/S return over time."""
    if returns_df.empty:
        return

    apply_plot_style()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    period_pnl = (
        returns_df.groupby("rebalance_date")
        .apply(lambda g: (g.loc[g["position"] == 1, "ls_return"].mean() +
                          g.loc[g["position"] == -1, "ls_return"].mean()) / 2
               if (g["position"] == 1).any() and (g["position"] == -1).any()
               else g["ls_return"].mean())
        .rename("net_return")
        .sort_index()
    )

    cum = (1 + period_pnl).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cum.index, cum.values, color=NAVY, linewidth=2.0, label="L/S strategy")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")

    rolling_max = cum.cummax()
    in_dd       = (cum < rolling_max).values
    dates       = cum.index
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

    sharpe  = metrics.get("sharpe_ratio", 0)
    ann_ret = metrics.get("annualised_return", 0)
    ax.set_title(
        f"Long-Short Equity Curve — {label.upper()}\n"
        f"Sharpe={sharpe:.3f}  Ann.Ret={ann_ret:.1%}  Hold={metrics.get('hold_days','?')}d"
    )
    ax.set_ylabel("Cumulative return (x)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = config.PLOTS_DIR / f"longshort_equity_{label}.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("L/S equity curve saved -> %s", out)


def plot_quantile_spread(
    pred_df: pd.DataFrame,
    prices: pd.DataFrame,
    lag: int = 2,
    n_quantiles: int = 5,
    label: str = "longshort",
    signal_col: str = "signal_normal",
    confidence_threshold: float | None = None,
) -> None:
    """Bar chart of mean forward return per signal quintile at `lag` days."""
    conf_thresh = confidence_threshold or config.CONFIDENCE_THRESHOLD
    signals     = _aggregate_signals(pred_df, signal_col, conf_thresh)
    price_sorted = prices.sort_index()

    pairs: list[dict] = []
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
        pairs.append({"signal": float(row["signal"]), "fwd": float((p_lag - p0) / p0)})

    if len(pairs) < n_quantiles * 4:
        logger.warning("Too few signal-return pairs for quantile plot (%d)", len(pairs))
        return

    df  = pd.DataFrame(pairs)
    df["q"] = pd.qcut(df["signal"], q=n_quantiles, labels=False, duplicates="drop") + 1
    qret = df.groupby("q")["fwd"].mean()

    apply_plot_style()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    colours = [NAVY if r >= 0 else GOLD for r in qret.values]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(qret.index.astype(str), qret.values, color=colours, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Signal quintile (1=most negative, {n_quantiles}=most positive)")
    ax.set_ylabel(f"Mean {lag}-day forward return")
    ax.set_title(f"Quintile Return Spread — {label.upper()}  (lag={lag}d)")
    ax.grid(axis="y", alpha=0.3)
    spread = float(qret.iloc[-1] - qret.iloc[0])
    ax.text(0.98, 0.02, f"Q5-Q1 spread = {spread:+.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    fig.tight_layout()
    out = config.PLOTS_DIR / f"longshort_quantile_{label}_lag{lag}.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Quantile spread plot saved -> %s", out)
