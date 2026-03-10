"""Multi-factor signal construction for AlphaLens.

Computes four cross-sectional alpha factors from price and volume data,
z-scores them within each rebalance day's cross-section, and combines
them with the FinBERT sentiment signal into a single composite score.

Factors
-------
sentiment   Already computed by the FinBERT pipeline (passed in).
momentum    20-day price return, skipping the last 2 days to avoid
            short-term reversal.  Captures price trend continuation.
volatility  Negative of 20-day realised volatility.  Exploits the
            low-volatility anomaly: lower-vol stocks tend to outperform
            on a risk-adjusted basis.
liquidity   Log of average daily dollar volume over 20 days.  Liquid
            stocks carry tighter spreads and are easier to execute —
            this factor prefers names the strategy can actually trade.

Combination
-----------
combined = w_s*sentiment + w_m*momentum + w_v*volatility + w_l*liquidity

All four factors are z-scored cross-sectionally (per rebalance date)
before combination so that no single factor dominates by scale.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252

# Default factor weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "sentiment":  0.50,
    "momentum":   0.30,
    "volatility": 0.20,
    "liquidity":  0.00,   # off by default; activate with weight > 0
}


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------


def fetch_volume_data(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch daily trading volume for *tickers* from yfinance.

    Returns:
        Wide DataFrame indexed by date with one column per ticker.
        Columns with all-NaN are dropped.  Returns empty DataFrame on failure.
    """
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
    except Exception as exc:
        logger.warning("Volume download failed: %s", exc)
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Volume" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        volumes = raw["Volume"].copy()
        if isinstance(volumes, pd.Series):
            volumes = volumes.to_frame(name=tickers[0])
    else:
        col = "Volume" if "Volume" in raw.columns else None
        if col is None:
            return pd.DataFrame()
        volumes = raw[[col]].rename(columns={col: tickers[0]})

    missing = [t for t in volumes.columns if volumes[t].isna().all()]
    volumes = volumes.drop(columns=missing, errors="ignore")
    volumes.index = pd.to_datetime(volumes.index).tz_localize(None)
    logger.info("Volume data: %d tickers x %d days", len(volumes.columns), len(volumes))
    return volumes


# ---------------------------------------------------------------------------
# Cross-sectional z-score
# ---------------------------------------------------------------------------


def _zscore(series: pd.Series, clip: float = 3.0) -> pd.Series:
    """Z-score a Series cross-sectionally, clip outliers to [-clip, +clip].

    Returns a zero Series if std is near zero (all tickers have same value).
    """
    series = series.dropna()
    std = series.std()
    if std < 1e-8 or len(series) < 2:
        return pd.Series(0.0, index=series.index)
    return ((series - series.mean()) / std).clip(-clip, clip)


# ---------------------------------------------------------------------------
# Individual factor computation (one rebalance date at a time)
# ---------------------------------------------------------------------------


def _price_pos(price_sorted: pd.DataFrame, date: pd.Timestamp) -> int:
    """Return the integer index in price_sorted for *date* (closest at or before)."""
    pos = price_sorted.index.searchsorted(date)
    # searchsorted returns insertion point; use it directly (entry = close of signal day)
    return int(pos)


def compute_momentum_cross_section(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    window: int = 21,
    skip: int = 2,
) -> pd.Series:
    """20-day momentum (skip last 2 days) cross-sectional z-score.

    Skipping the last 2 days avoids the well-documented 1-week reversal
    that contaminates shorter momentum signals.

    Returns:
        Series indexed by ticker (NaN for tickers without sufficient history).
    """
    price_sorted = prices.sort_index()
    pos          = _price_pos(price_sorted, date)
    end_pos      = max(0, pos - skip)
    start_pos    = max(0, pos - window - skip)

    if end_pos <= start_pos:
        return pd.Series(dtype=float)

    raw: dict[str, float] = {}
    for ticker in prices.columns:
        if ticker not in price_sorted.columns:
            continue
        p_end   = price_sorted[ticker].iloc[end_pos]
        p_start = price_sorted[ticker].iloc[start_pos]
        if pd.isna(p_start) or pd.isna(p_end) or p_start == 0:
            continue
        raw[ticker] = float((p_end - p_start) / p_start)

    return _zscore(pd.Series(raw))


def compute_vol_factor_cross_section(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    window: int = 20,
) -> pd.Series:
    """Low-volatility anomaly factor: NEGATIVE of realised vol, z-scored.

    Lower actual volatility → higher factor score.

    Returns:
        Series indexed by ticker.
    """
    price_sorted = prices.sort_index()
    pos          = _price_pos(price_sorted, date)
    start_pos    = max(0, pos - window)

    raw: dict[str, float] = {}
    for ticker in prices.columns:
        if ticker not in price_sorted.columns:
            continue
        slice_ = price_sorted[ticker].iloc[start_pos:pos]
        if len(slice_) < 4:
            continue
        rets = slice_.pct_change().dropna()
        if len(rets) < 3:
            continue
        vol = float(rets.std()) * np.sqrt(_TRADING_DAYS)
        if vol > 1e-8:
            raw[ticker] = -vol  # negative: lower vol = higher score

    return _zscore(pd.Series(raw))


def compute_liquidity_factor_cross_section(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    date: pd.Timestamp,
    window: int = 20,
) -> pd.Series:
    """Liquidity factor: log average daily dollar volume, z-scored.

    Higher dollar volume → tighter spreads → more tradeable → higher score.
    Returns empty Series if volume data unavailable.

    Returns:
        Series indexed by ticker.
    """
    if volumes.empty:
        return pd.Series(dtype=float)

    price_sorted = prices.sort_index()
    vol_sorted   = volumes.sort_index()
    pos          = _price_pos(price_sorted, date)
    start_pos    = max(0, pos - window)

    raw: dict[str, float] = {}
    for ticker in prices.columns:
        if ticker not in price_sorted.columns or ticker not in vol_sorted.columns:
            continue
        p_slice = price_sorted[ticker].iloc[start_pos:pos]
        v_slice = vol_sorted[ticker].iloc[start_pos:pos]
        if len(p_slice) < 4 or len(v_slice) < 4:
            continue
        # Align by index
        common = p_slice.index.intersection(v_slice.index)
        if len(common) < 4:
            continue
        dollar_vol = p_slice.loc[common] * v_slice.loc[common]
        avg_dv = float(dollar_vol.mean())
        if avg_dv > 0:
            raw[ticker] = float(np.log(avg_dv))

    return _zscore(pd.Series(raw))


# ---------------------------------------------------------------------------
# Batch factor matrix builder
# ---------------------------------------------------------------------------


def build_factor_signals(
    signals_df: pd.DataFrame,
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    mom_window: int = 21,
    vol_window: int = 20,
    liq_window: int = 20,
) -> pd.DataFrame:
    """Compute and attach price-based factors to an aggregated signals DataFrame.

    Args:
        signals_df: DataFrame with ``date``, ``ticker``, ``signal`` columns,
            as produced by ``_aggregate_signals`` in longshort_engine.
        prices: Wide adjusted-close price DataFrame indexed by date.
        volumes: Wide volume DataFrame indexed by date.  May be ``None``
            or empty; liquidity factor will be zero if so.
        mom_window: Lookback window (trading days) for momentum.
        vol_window: Lookback window (trading days) for vol factor.
        liq_window: Lookback window (trading days) for liquidity factor.

    Returns:
        DataFrame identical to *signals_df* but with three additional
        columns: ``momentum``, ``vol_factor``, ``liquidity``
        (all cross-sectionally z-scored per date).
    """
    if signals_df.empty:
        return signals_df.assign(momentum=np.nan, vol_factor=np.nan, liquidity=np.nan)

    volumes_df = volumes if (volumes is not None and not volumes.empty) else pd.DataFrame()
    unique_dates = sorted(pd.to_datetime(signals_df["date"]).unique())

    mom_map: dict[tuple, float] = {}
    vol_map: dict[tuple, float] = {}
    liq_map: dict[tuple, float] = {}

    for date in unique_dates:
        date_ts = pd.Timestamp(date)
        tickers_today = signals_df.loc[
            pd.to_datetime(signals_df["date"]) == date_ts, "ticker"
        ].tolist()

        # Subset prices/volumes to tickers present today
        avail_p = [t for t in tickers_today if t in prices.columns]
        prices_sub  = prices[avail_p] if avail_p else pd.DataFrame()
        volumes_sub = volumes_df[
            [t for t in avail_p if t in volumes_df.columns]
        ] if not volumes_df.empty and avail_p else pd.DataFrame()

        if prices_sub.empty:
            continue

        mom  = compute_momentum_cross_section(prices_sub, date_ts, window=mom_window)
        vol  = compute_vol_factor_cross_section(prices_sub, date_ts, window=vol_window)
        liq  = compute_liquidity_factor_cross_section(prices_sub, volumes_sub, date_ts, window=liq_window)

        for ticker in tickers_today:
            key = (date_ts, ticker)
            mom_map[key] = float(mom.get(ticker, np.nan))
            vol_map[key] = float(vol.get(ticker, np.nan))
            liq_map[key] = float(liq.get(ticker, np.nan))

    out = signals_df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["momentum"]   = out.apply(lambda r: mom_map.get((r["date"], r["ticker"]), np.nan), axis=1)
    out["vol_factor"] = out.apply(lambda r: vol_map.get((r["date"], r["ticker"]), np.nan), axis=1)
    out["liquidity"]  = out.apply(lambda r: liq_map.get((r["date"], r["ticker"]), np.nan), axis=1)

    n_with_mom = out["momentum"].notna().sum()
    n_with_vol = out["vol_factor"].notna().sum()
    n_with_liq = out["liquidity"].notna().sum()
    logger.info(
        "Factor matrix built: %d rows | momentum=%d | vol=%d | liquidity=%d",
        len(out), n_with_mom, n_with_vol, n_with_liq,
    )
    return out


# ---------------------------------------------------------------------------
# Sector neutralisation
# ---------------------------------------------------------------------------


def sector_neutral_signals(
    signals_df: pd.DataFrame,
    signal_col: str = "signal",
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Cross-sectionally demean signals within each GICS sector per date.

    News signals often cluster in certain sectors (e.g. tech earnings season),
    creating implicit sector bets rather than pure alpha.  Sector neutralisation
    removes the sector-mean each day so the strategy bets on *within-sector*
    relative strength only.

    Algorithm (per date):
        z = (signal - sector_mean) / sector_std

    When a sector has only one ticker on a given date, the signal is set to
    0.0 (no within-sector information) rather than distorting the cross-section.

    Args:
        signals_df: DataFrame with ``date``, ``ticker``, and *signal_col*.
        signal_col: Column holding the signal to neutralise.
        sector_map: Dict mapping ticker -> GICS sector string.
            Defaults to ``config.SECTOR_MAP``.

    Returns:
        Copy of *signals_df* with the *signal_col* values replaced by
        sector-neutral z-scores.  Tickers not found in *sector_map* are
        assigned to an "Other" sector and normalised within that group.
    """
    if sector_map is None:
        sector_map = config.SECTOR_MAP

    out = signals_df.copy()
    out["_sector"] = out["ticker"].map(sector_map).fillna("Other")

    def _sector_zscore(grp: pd.Series) -> pd.Series:
        std = grp.std()
        if len(grp) < 2 or std < 1e-9:
            return pd.Series(0.0, index=grp.index)
        return (grp - grp.mean()) / std

    out[signal_col] = (
        out.groupby(["date", "_sector"])[signal_col]
        .transform(_sector_zscore)
    )
    out = out.drop(columns=["_sector"])
    logger.info(
        "Sector neutralisation applied to %d rows across %d dates",
        len(out), out["date"].nunique(),
    )
    return out


# ---------------------------------------------------------------------------
# Factor combination
# ---------------------------------------------------------------------------


def combine_factors(
    factor_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute a ``combined_score`` column from weighted factor columns.

    Missing factor values (NaN) are replaced with 0.0 (neutral contribution).
    The weights are normalised to sum to 1.0 before application.

    Args:
        factor_df: DataFrame from :func:`build_factor_signals` containing
            ``signal`` (sentiment), ``momentum``, ``vol_factor``, ``liquidity``.
        weights: Dict mapping factor name to weight.  Defaults to
            ``DEFAULT_WEIGHTS``.  Factor names: ``sentiment``, ``momentum``,
            ``volatility``, ``liquidity``.

    Returns:
        DataFrame with an additional ``combined_score`` column.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # Normalise weights to sum 1
    total = sum(abs(v) for v in w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}

    out = factor_df.copy()

    col_map = {
        "sentiment":  "signal",
        "momentum":   "momentum",
        "volatility": "vol_factor",
        "liquidity":  "liquidity",
    }

    score = pd.Series(0.0, index=out.index)
    for factor_name, col in col_map.items():
        wt = w.get(factor_name, 0.0)
        if abs(wt) < 1e-9 or col not in out.columns:
            continue
        vals = out[col].fillna(0.0)
        score = score + wt * vals

    out["combined_score"] = score.round(6)
    return out


# ---------------------------------------------------------------------------
# News-derived features: volume shock + temporal sentiment
# ---------------------------------------------------------------------------


def compute_news_features(
    signals_df: pd.DataFrame,
    raw_preds: pd.DataFrame,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """Compute news volume shock and temporal sentiment features.

    These features capture *how* sentiment is changing, not just what it is:

    news_volume_shock
        Today's headline count divided by the trailing 30-day average for that
        ticker.  Values > 1 = above-average news day (potential event).
        Large spikes often precede volatility and short-term price moves.

    sentiment_velocity
        sentiment_today - sentiment_3day_avg.  Captures how quickly sentiment
        is shifting.  Fast positive swings predict short-term momentum;
        fast negative swings predict mean-reversion.

    sentiment_trend
        sentiment_today - sentiment_7day_avg.  Longer-horizon change, useful
        for distinguishing sustained regime shifts from noise.

    Args:
        signals_df: DataFrame from ``_aggregate_signals`` with columns
            ``date``, ``ticker``, ``signal``, ``headline_count``.
        raw_preds: Raw per-headline predictions with ``date``, ``ticker``,
            ``signal_normal`` (one row per headline).
        lookback_days: Window for rolling volume baseline (default 30).

    Returns:
        ``signals_df`` with three additional columns: ``news_volume_shock``,
        ``sentiment_velocity``, ``sentiment_trend``.
    """
    out = signals_df.copy()

    # --- news volume shock ---------------------------------------------------
    if "headline_count" in out.columns:
        out = out.sort_values(["ticker", "date"])
        rolling_vol = (
            out.groupby("ticker")["headline_count"]
            .transform(lambda s: s.shift(1).rolling(lookback_days, min_periods=5).mean())
        )
        out["news_volume_shock"] = (
            out["headline_count"] / rolling_vol.clip(lower=1)
        ).fillna(1.0).clip(upper=10.0)
    else:
        out["news_volume_shock"] = 1.0

    # --- temporal sentiment features -----------------------------------------
    out = out.sort_values(["ticker", "date"])
    roll3 = (
        out.groupby("ticker")["signal"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    roll7 = (
        out.groupby("ticker")["signal"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    out["sentiment_velocity"] = (out["signal"] - roll3).fillna(0.0)
    out["sentiment_trend"]    = (out["signal"] - roll7).fillna(0.0)

    return out


def compute_market_sentiment(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a market-wide sentiment index as a macro indicator.

    Aggregates sentiment across all tickers per date into a single value
    that can be used as a market regime filter or cross-sectional demeaning
    baseline.

    Args:
        signals_df: Per-ticker daily signals with ``date``, ``ticker``, ``signal``.

    Returns:
        DataFrame with ``date`` and ``market_sentiment`` (mean signal across
        all tickers on that date).
    """
    mkt = (
        signals_df.groupby("date")["signal"]
        .agg(market_sentiment="mean", market_sentiment_std="std", ticker_count="count")
        .reset_index()
    )
    return mkt


# ---------------------------------------------------------------------------
# Individual factor IC (for diagnostic reporting)
# ---------------------------------------------------------------------------


def compute_factor_ics(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    lag: int = 2,
) -> pd.DataFrame:
    """Compute Spearman IC between each factor and the lag-day forward return.

    Args:
        factor_df: Output of :func:`build_factor_signals` (date, ticker level).
        prices: Price DataFrame.
        lag: Forward-return horizon in trading days.

    Returns:
        DataFrame with columns ``factor``, ``ic``, ``p_value``,
        ``significant``, ``n``.
    """
    from scipy import stats as _stats

    price_sorted = prices.sort_index()
    factors = ["signal", "momentum", "vol_factor", "liquidity", "combined_score"]
    rows: list[dict] = []

    for col in factors:
        if col not in factor_df.columns:
            continue

        pairs: list[tuple[float, float]] = []
        for _, row in factor_df.iterrows():
            date   = pd.Timestamp(row["date"])
            ticker = str(row["ticker"])
            val    = row.get(col, np.nan)
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
            rows.append({"factor": col, "ic": np.nan, "p_value": np.nan,
                         "significant": False, "n": len(pairs)})
            continue

        arr = pd.DataFrame(pairs, columns=["factor_val", "fwd_ret"])
        ic, p = _stats.spearmanr(arr["factor_val"], arr["fwd_ret"])
        rows.append({
            "factor":      col,
            "ic":          round(float(ic), 4),
            "p_value":     round(float(p), 4),
            "significant": bool(p < 0.05),
            "n":           len(pairs),
        })
        logger.info("Factor IC [%s, lag=%dd]: IC=%+.4f  p=%.4f  n=%d",
                    col, lag, ic, p, len(pairs))

    return pd.DataFrame(rows)
