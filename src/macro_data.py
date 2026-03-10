"""Macroeconomic data layer for AlphaLens.

Collects and derives macroeconomic features used to contextualise
sentiment-driven recommendations.  Supports two data backends:

* **FRED API** — CPI, unemployment, payrolls, GDP, Fed funds rate
  (requires ``FRED_API_KEY`` in ``.env``; graceful fallback if absent).
* **yfinance** — VIX (``^VIX``), 10-yr yield (``^TNX``), 2-yr yield
  (``^UST2Y``), 3-month T-bill (``^IRX``), S&P 500 (``^GSPC``).

Derived features
~~~~~~~~~~~~~~~~
* ``macro_surprise`` — deviation of latest reading vs 12-month mean.
* ``macro_trend``    — sign of 3-month change.
* ``macro_regime``   — one of :class:`MacroRegime` labels.

Usage::

    from src.macro_data import MacroDataCollector
    collector = MacroDataCollector()
    snapshot = collector.get_macro_snapshot()
    print(snapshot.regime)          # e.g. MacroRegime.TIGHTENING
    print(snapshot.to_feature_dict())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FRED series identifiers
# ---------------------------------------------------------------------------

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

_FRED_SERIES: dict[str, str] = {
    "cpi":          "CPIAUCSL",     # Consumer Price Index, All Urban Consumers
    "unemployment": "UNRATE",       # Unemployment Rate
    "payrolls":     "PAYEMS",       # Nonfarm Payroll Employment (thousands)
    "gdp":          "GDP",          # Gross Domestic Product (quarterly)
    "fed_funds":    "FEDFUNDS",     # Effective Federal Funds Rate
    "yield_10yr":   "DGS10",        # 10-Year Treasury Constant Maturity Rate
    "yield_2yr":    "DGS2",         # 2-Year Treasury Constant Maturity Rate
}

# yfinance proxies (used when FRED key absent or as supplements)
_YF_SYMBOLS: dict[str, str] = {
    "vix":        "^VIX",
    "spy":        "^GSPC",
    "yield_10yr_yf": "^TNX",
    "yield_3m_yf":   "^IRX",
}


# ---------------------------------------------------------------------------
# MacroRegime enum
# ---------------------------------------------------------------------------


class MacroRegime(str, Enum):
    """Categorical macroeconomic environment label."""

    TIGHTENING      = "tightening"       # Fed hiking, yields rising
    EASING          = "easing"           # Fed cutting, yields falling
    RISK_ON         = "risk_on"          # Low VIX, positive growth signals
    RISK_OFF        = "risk_off"         # High VIX, flight to safety
    INFLATION_SHOCK = "inflation_shock"  # CPI surprise materially above trend
    GROWTH_SLOWDOWN = "growth_slowdown"  # GDP miss, rising unemployment
    NEUTRAL         = "neutral"          # No dominant regime signal


# ---------------------------------------------------------------------------
# MacroSnapshot dataclass
# ---------------------------------------------------------------------------


@dataclass
class MacroSnapshot:
    """Point-in-time macroeconomic state.

    Attributes:
        timestamp:     UTC time when the snapshot was captured.
        cpi_yoy:       CPI year-over-year change (fraction, e.g. 0.035).
        unemployment:  Unemployment rate (e.g. 0.038).
        payrolls_chg:  Monthly nonfarm payroll change (thousands).
        gdp_growth:    Annualised GDP growth rate (fraction).
        fed_funds:     Effective Federal Funds Rate (fraction).
        yield_10yr:    10-year Treasury yield (fraction).
        yield_2yr:     2-year Treasury yield (fraction).
        yield_spread:  10yr − 2yr spread (positive = normal curve).
        vix:           CBOE VIX level.
        macro_surprise: Standardised surprise score across all indicators.
        macro_trend:    Direction of recent macro change (1, 0, -1).
        regime:         Inferred :class:`MacroRegime`.
        raw:            Dict of raw latest values for each indicator.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cpi_yoy: float = 0.0
    unemployment: float = 0.0
    payrolls_chg: float = 0.0
    gdp_growth: float = 0.0
    fed_funds: float = 0.0
    yield_10yr: float = 0.0
    yield_2yr: float = 0.0
    yield_spread: float = 0.0
    vix: float = 20.0
    macro_surprise: float = 0.0
    macro_trend: int = 0           # +1 improving, -1 deteriorating, 0 flat
    regime: MacroRegime = MacroRegime.NEUTRAL
    raw: dict = field(default_factory=dict)

    def to_feature_dict(self) -> dict[str, float]:
        """Return a flat dict of numeric features for model input."""
        return {
            "macro_cpi_yoy":       self.cpi_yoy,
            "macro_unemployment":  self.unemployment,
            "macro_payrolls_chg":  self.payrolls_chg,
            "macro_gdp_growth":    self.gdp_growth,
            "macro_fed_funds":     self.fed_funds,
            "macro_yield_10yr":    self.yield_10yr,
            "macro_yield_2yr":     self.yield_2yr,
            "macro_yield_spread":  self.yield_spread,
            "macro_vix":           self.vix,
            "macro_surprise":      self.macro_surprise,
            "macro_trend":         float(self.macro_trend),
            # One-hot encode regime (7 classes)
            **{f"macro_regime_{r.value}": float(self.regime == r) for r in MacroRegime},
        }


# ---------------------------------------------------------------------------
# MacroDataCollector
# ---------------------------------------------------------------------------


class MacroDataCollector:
    """Fetches, caches, and derives macroeconomic features.

    Args:
        fred_api_key: FRED API key.  Defaults to ``config.FRED_API_KEY``.
        cache_dir:    Directory for parquet caches.  Defaults to
                      ``config.PROCESSED_DATA_DIR``.
        lookback_days: Number of days of history to retrieve (default 730).
    """

    def __init__(
        self,
        fred_api_key: str | None = None,
        cache_dir: Path | None = None,
        lookback_days: int | None = None,
    ) -> None:
        self._api_key: str = fred_api_key or getattr(config, "FRED_API_KEY", "")
        self._cache_dir: Path = cache_dir or config.PROCESSED_DATA_DIR
        self._lookback: int = lookback_days or getattr(config, "MACRO_LOOKBACK_DAYS", 730)
        logger.info(
            "MacroDataCollector init | FRED key=%s | lookback=%d days",
            "set" if self._api_key else "absent",
            self._lookback,
        )

    # ------------------------------------------------------------------
    # FRED fetch
    # ------------------------------------------------------------------

    def _fetch_fred_series(self, series_id: str) -> pd.Series:
        """Download a FRED time series via the REST API.

        Args:
            series_id: FRED series identifier (e.g. ``"CPIAUCSL"``).

        Returns:
            pd.Series: Date-indexed float series, or empty Series on failure.
        """
        if not self._api_key:
            return pd.Series(dtype=float, name=series_id)

        start_date = (pd.Timestamp.now() - pd.Timedelta(days=self._lookback)).strftime("%Y-%m-%d")
        params = {
            "series_id":         series_id,
            "api_key":           self._api_key,
            "file_type":         "json",
            "observation_start": start_date,
            "sort_order":        "asc",
        }
        try:
            resp = requests.get(_FRED_BASE, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("observations", [])
            if not data:
                return pd.Series(dtype=float, name=series_id)
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            series = df.set_index("date")["value"].dropna()
            series.name = series_id
            logger.info("FRED [%s]: %d observations", series_id, len(series))
            return series
        except Exception as exc:
            logger.warning("FRED fetch failed [%s]: %s", series_id, exc)
            return pd.Series(dtype=float, name=series_id)

    # ------------------------------------------------------------------
    # yfinance fetch
    # ------------------------------------------------------------------

    def _fetch_yfinance_series(self, symbol: str, field: str = "Close") -> pd.Series:
        """Download a yfinance time series.

        Args:
            symbol: yfinance ticker symbol.
            field:  OHLCV column to extract (default ``"Close"``).

        Returns:
            pd.Series: Date-indexed float series, or empty Series on failure.
        """
        try:
            import yfinance as yf

            period = f"{max(1, self._lookback // 365 + 1)}y"
            df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
            if df.empty:
                return pd.Series(dtype=float, name=symbol)
            col = df[field]
            # Newer yfinance returns MultiIndex columns; squeeze to 1-D Series
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            series = col.squeeze().dropna()
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = series.astype(float)
            series.index = pd.to_datetime(series.index).tz_localize(None)
            series.name = symbol
            logger.info("yfinance [%s]: %d observations", symbol, len(series))
            return series
        except Exception as exc:
            logger.warning("yfinance fetch failed [%s]: %s", symbol, exc)
            return pd.Series(dtype=float, name=symbol)

    # ------------------------------------------------------------------
    # Derived features
    # ------------------------------------------------------------------

    def _compute_yoy_change(self, series: pd.Series) -> float:
        """Compute year-over-year percentage change for the latest reading."""
        if len(series) < 13:
            return 0.0
        latest = series.iloc[-1]
        year_ago = series.iloc[-13] if len(series) >= 13 else series.iloc[0]
        if year_ago == 0:
            return 0.0
        return float((latest - year_ago) / abs(year_ago))

    def _compute_mom_change(self, series: pd.Series) -> float:
        """Compute month-over-month change for the latest reading."""
        if len(series) < 2:
            return 0.0
        return float(series.iloc[-1] - series.iloc[-2])

    def _compute_surprise(self, series: pd.Series, window: int = 12) -> float:
        """Standardised surprise: (latest − rolling mean) / rolling std."""
        if len(series) < window:
            return 0.0
        recent = series.iloc[-window:]
        mu, sigma = float(recent.mean()), float(recent.std())
        if sigma == 0:
            return 0.0
        return float((series.iloc[-1] - mu) / sigma)

    def _compute_trend(self, series: pd.Series, window: int = 3) -> int:
        """Trend direction over last *window* periods: +1, 0, or -1."""
        if len(series) < window + 1:
            return 0
        change = float(series.iloc[-1] - series.iloc[-(window + 1)])
        threshold = 0.05 * abs(float(series.mean())) if series.mean() != 0 else 1e-6
        if change > threshold:
            return 1
        if change < -threshold:
            return -1
        return 0

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def _classify_regime(
        self,
        cpi_yoy: float,
        cpi_surprise: float,
        fed_funds: float,
        fed_trend: int,
        yield_spread: float,
        vix: float,
        gdp_trend: int,
        unemployment_trend: int,
    ) -> MacroRegime:
        """Classify the macro environment into a :class:`MacroRegime`.

        Uses a priority-ordered rule set so that extreme conditions take
        precedence over mild ones.

        Args:
            cpi_yoy:          CPI year-over-year change.
            cpi_surprise:     Standardised CPI surprise.
            fed_funds:        Current Fed Funds Rate.
            fed_trend:        Direction of recent Fed Funds change.
            yield_spread:     10yr − 2yr Treasury spread.
            vix:              Current VIX level.
            gdp_trend:        Direction of recent GDP change.
            unemployment_trend: Direction of recent unemployment change.

        Returns:
            MacroRegime: The dominant regime label.
        """
        # Inflation shock: CPI materially above 3% YoY and rising fast
        if cpi_yoy > 0.04 and cpi_surprise > 1.5:
            return MacroRegime.INFLATION_SHOCK

        # Growth slowdown: GDP falling and unemployment rising
        if gdp_trend < 0 and unemployment_trend > 0:
            return MacroRegime.GROWTH_SLOWDOWN

        # Risk-off: elevated VIX or inverted yield curve
        if vix > 28 or yield_spread < -0.003:
            return MacroRegime.RISK_OFF

        # Tightening: Fed funds rising or above 4%, above-trend CPI
        if fed_trend > 0 or (fed_funds > 0.04 and cpi_yoy > 0.025):
            return MacroRegime.TIGHTENING

        # Easing: Fed funds falling or near zero
        if fed_trend < 0 or fed_funds < 0.01:
            return MacroRegime.EASING

        # Risk-on: low VIX, positive growth, normal yield curve
        if vix < 16 and gdp_trend >= 0 and yield_spread > 0.005:
            return MacroRegime.RISK_ON

        return MacroRegime.NEUTRAL

    # ------------------------------------------------------------------
    # Synthetic fallback
    # ------------------------------------------------------------------

    def _synthetic_snapshot(self) -> MacroSnapshot:
        """Return a plausible neutral snapshot when all data sources fail."""
        logger.warning("All macro data sources failed — returning synthetic snapshot")
        return MacroSnapshot(
            cpi_yoy=0.031,
            unemployment=0.039,
            payrolls_chg=200.0,
            gdp_growth=0.025,
            fed_funds=0.053,
            yield_10yr=0.044,
            yield_2yr=0.049,
            yield_spread=-0.005,
            vix=18.5,
            macro_surprise=0.0,
            macro_trend=0,
            regime=MacroRegime.NEUTRAL,
            raw={"source": "synthetic"},
        )

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def get_macro_snapshot(self) -> MacroSnapshot:
        """Collect all macro indicators and return a populated snapshot.

        Data is fetched from FRED (if key available) and yfinance.
        Falls back gracefully per indicator when fetches fail.

        Returns:
            MacroSnapshot: The latest macroeconomic state.
        """
        cache_path = self._cache_dir / "macro_snapshot.parquet"

        # --- Fetch CPI ---
        cpi_series = self._fetch_fred_series(_FRED_SERIES["cpi"])
        cpi_yoy = self._compute_yoy_change(cpi_series)
        cpi_surprise = self._compute_surprise(cpi_series)

        # --- Fetch unemployment ---
        unemp_series = self._fetch_fred_series(_FRED_SERIES["unemployment"])
        unemployment = float(unemp_series.iloc[-1]) / 100 if len(unemp_series) > 0 else 0.04
        unemployment_trend = self._compute_trend(unemp_series)

        # --- Fetch payrolls ---
        payrolls_series = self._fetch_fred_series(_FRED_SERIES["payrolls"])
        payrolls_chg = self._compute_mom_change(payrolls_series)

        # --- Fetch GDP ---
        gdp_series = self._fetch_fred_series(_FRED_SERIES["gdp"])
        gdp_growth = self._compute_yoy_change(gdp_series)
        gdp_trend = self._compute_trend(gdp_series, window=2)

        # --- Fed Funds ---
        fed_series = self._fetch_fred_series(_FRED_SERIES["fed_funds"])
        fed_funds = float(fed_series.iloc[-1]) / 100 if len(fed_series) > 0 else 0.05
        fed_trend = self._compute_trend(fed_series, window=3)

        # --- Yield curve (FRED preferred, yfinance fallback) ---
        y10_series = self._fetch_fred_series(_FRED_SERIES["yield_10yr"])
        if y10_series.empty:
            y10_series = self._fetch_yfinance_series(_YF_SYMBOLS["yield_10yr_yf"])
            if not y10_series.empty:
                y10_series = y10_series / 100  # TNX is in percent

        y2_series = self._fetch_fred_series(_FRED_SERIES["yield_2yr"])

        yield_10yr = float(y10_series.iloc[-1]) if len(y10_series) > 0 else 0.044
        yield_2yr = float(y2_series.iloc[-1]) / 100 if len(y2_series) > 0 else 0.049
        if len(y10_series) > 0 and y10_series.iloc[-1] > 1:
            yield_10yr /= 100  # FRED gives percent; normalise to fraction
        yield_spread = yield_10yr - yield_2yr

        # --- VIX ---
        vix_series = self._fetch_yfinance_series(_YF_SYMBOLS["vix"])
        vix = float(vix_series.iloc[-1]) if len(vix_series) > 0 else 20.0

        # --- Aggregate surprise & trend ---
        surprises = []
        for s in [cpi_series, unemp_series, payrolls_series, gdp_series, fed_series]:
            if len(s) > 3:
                surprises.append(self._compute_surprise(s))
        macro_surprise = float(np.mean(surprises)) if surprises else 0.0

        trends = [
            self._compute_trend(s)
            for s in [gdp_series, payrolls_series]
            if len(s) > 3
        ]
        macro_trend = int(np.sign(sum(trends))) if trends else 0

        # --- Regime classification ---
        regime = self._classify_regime(
            cpi_yoy=cpi_yoy,
            cpi_surprise=cpi_surprise,
            fed_funds=fed_funds,
            fed_trend=fed_trend,
            yield_spread=yield_spread,
            vix=vix,
            gdp_trend=gdp_trend,
            unemployment_trend=unemployment_trend,
        )

        snapshot = MacroSnapshot(
            cpi_yoy=round(cpi_yoy, 5),
            unemployment=round(unemployment, 5),
            payrolls_chg=round(payrolls_chg, 1),
            gdp_growth=round(gdp_growth, 5),
            fed_funds=round(fed_funds, 5),
            yield_10yr=round(yield_10yr, 5),
            yield_2yr=round(yield_2yr, 5),
            yield_spread=round(yield_spread, 5),
            vix=round(vix, 2),
            macro_surprise=round(macro_surprise, 4),
            macro_trend=macro_trend,
            regime=regime,
            raw={
                "cpi_series_len":     len(cpi_series),
                "unemp_series_len":   len(unemp_series),
                "payrolls_series_len": len(payrolls_series),
                "gdp_series_len":     len(gdp_series),
                "fed_series_len":     len(fed_series),
                "y10_series_len":     len(y10_series),
                "vix_series_len":     len(vix_series),
            },
        )

        logger.info(
            "MacroSnapshot | regime=%s | CPI YoY=%.1f%% | VIX=%.1f | spread=%.0fbps",
            regime.value,
            cpi_yoy * 100,
            vix,
            yield_spread * 10_000,
        )

        # Cache to parquet for downstream use
        try:
            pd.DataFrame([snapshot.to_feature_dict()]).to_parquet(cache_path, index=False)
        except Exception as exc:
            logger.debug("Macro cache write failed: %s", exc)

        return snapshot


def run_macro_data() -> MacroSnapshot:
    """Pipeline-stage entry point: fetch and return a :class:`MacroSnapshot`.

    Returns:
        MacroSnapshot: Latest macroeconomic state.
    """
    collector = MacroDataCollector()
    return collector.get_macro_snapshot()
