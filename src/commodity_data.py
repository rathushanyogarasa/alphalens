"""Commodity shock data layer for AlphaLens.

Fetches daily prices for key commodity markets via yfinance and derives
features for downstream model enrichment:

* **Brent oil** (``BZ=F``)
* **Natural gas** (``NG=F``)
* **Copper**     (``HG=F``)
* **Gold**       (``GC=F``)

Derived features
~~~~~~~~~~~~~~~~
* ``price_change_*``     — rolling 5-day percentage change.
* ``volatility_*``       — 20-day rolling annualised volatility.
* ``shock_*``            — :class:`ShockType` classification per commodity.

Usage::

    from src.commodity_data import CommodityDataCollector
    collector = CommodityDataCollector()
    snapshot = collector.get_commodity_snapshot()
    print(snapshot.oil_shock)             # e.g. ShockType.SUPPLY_SHOCK
    print(snapshot.to_feature_dict())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Commodity yfinance symbols
# ---------------------------------------------------------------------------

_COMMODITY_SYMBOLS: dict[str, str] = {
    "oil":      "BZ=F",   # Brent Crude Oil Futures
    "gas":      "NG=F",   # Henry Hub Natural Gas Futures
    "copper":   "HG=F",   # Copper Futures (USD/lb)
    "gold":     "GC=F",   # Gold Futures (USD/troy oz)
}

# Lookback for history download
_LOOKBACK_DAYS: int = 365


# ---------------------------------------------------------------------------
# ShockType enum
# ---------------------------------------------------------------------------


class ShockType(str, Enum):
    """Classification of a commodity price move as supply or demand driven."""

    SUPPLY_SHOCK  = "supply_shock"   # Price spike driven by supply disruption
    DEMAND_SHOCK  = "demand_shock"   # Price move driven by demand change
    NEUTRAL       = "neutral"        # No material shock
    PRICE_CRASH   = "price_crash"    # Sharp price decline


# ---------------------------------------------------------------------------
# CommoditySnapshot dataclass
# ---------------------------------------------------------------------------


@dataclass
class CommoditySnapshot:
    """Point-in-time commodity market state.

    Attributes:
        timestamp:         UTC time of snapshot.
        oil_price:         Latest Brent crude price (USD/bbl).
        oil_change_5d:     5-day percentage change.
        oil_volatility:    20-day annualised return volatility.
        oil_shock:         Classified shock type.
        gas_price:         Latest natural gas price (USD/MMBtu).
        gas_change_5d:     5-day percentage change.
        gas_volatility:    20-day annualised return volatility.
        gas_shock:         Classified shock type.
        copper_price:      Latest copper price (USD/lb).
        copper_change_5d:  5-day percentage change.
        copper_volatility: 20-day annualised return volatility.
        copper_shock:      Classified shock type.
        gold_price:        Latest gold price (USD/troy oz).
        gold_change_5d:    5-day percentage change.
        gold_volatility:   20-day annualised return volatility.
        gold_shock:        Classified shock type.
        commodity_stress:  Aggregate stress indicator across all commodities.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    oil_price: float = 0.0
    oil_change_5d: float = 0.0
    oil_volatility: float = 0.0
    oil_shock: ShockType = ShockType.NEUTRAL

    gas_price: float = 0.0
    gas_change_5d: float = 0.0
    gas_volatility: float = 0.0
    gas_shock: ShockType = ShockType.NEUTRAL

    copper_price: float = 0.0
    copper_change_5d: float = 0.0
    copper_volatility: float = 0.0
    copper_shock: ShockType = ShockType.NEUTRAL

    gold_price: float = 0.0
    gold_change_5d: float = 0.0
    gold_volatility: float = 0.0
    gold_shock: ShockType = ShockType.NEUTRAL

    commodity_stress: float = 0.0    # 0 = calm, 1 = extreme stress

    def to_feature_dict(self) -> dict[str, float]:
        """Return a flat dict of numeric features for model input."""
        features: dict[str, float] = {}
        for name in ("oil", "gas", "copper", "gold"):
            features[f"commodity_{name}_price"]     = getattr(self, f"{name}_price")
            features[f"commodity_{name}_change_5d"] = getattr(self, f"{name}_change_5d")
            features[f"commodity_{name}_vol"]       = getattr(self, f"{name}_volatility")
            shock: ShockType = getattr(self, f"{name}_shock")
            for st in ShockType:
                features[f"commodity_{name}_shock_{st.value}"] = float(shock == st)
        features["commodity_stress"] = self.commodity_stress
        return features


# ---------------------------------------------------------------------------
# CommodityDataCollector
# ---------------------------------------------------------------------------


class CommodityDataCollector:
    """Fetches commodity prices and classifies shock regimes.

    Args:
        lookback_days: Days of price history to download (default 365).
        cache_dir:     Directory for parquet caches.
    """

    def __init__(
        self,
        lookback_days: int | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._lookback = lookback_days or getattr(config, "COMMODITY_LOOKBACK_DAYS", _LOOKBACK_DAYS)
        self._cache_dir = cache_dir or config.PROCESSED_DATA_DIR

    # ------------------------------------------------------------------
    # Price fetch
    # ------------------------------------------------------------------

    def _fetch_price_series(self, symbol: str) -> pd.Series:
        """Download daily closing prices for *symbol* from yfinance.

        Args:
            symbol: yfinance commodity futures ticker.

        Returns:
            pd.Series: Date-indexed closing price series.
        """
        try:
            import yfinance as yf

            period = f"{max(1, self._lookback // 365 + 1)}y"
            df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
            if df.empty:
                logger.warning("Empty price data for %s", symbol)
                return pd.Series(dtype=float, name=symbol)
            col = df["Close"]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            series = col.squeeze().dropna()
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = series.astype(float)
            series.index = pd.to_datetime(series.index).tz_localize(None)
            series.name = symbol
            logger.info("Commodity [%s]: %d days downloaded", symbol, len(series))
            return series
        except Exception as exc:
            logger.warning("yfinance fetch failed [%s]: %s", symbol, exc)
            return pd.Series(dtype=float, name=symbol)

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_change(self, series: pd.Series, window: int = 5) -> float:
        """Percentage change over last *window* trading days."""
        if len(series) < window + 1:
            return 0.0
        base = float(series.iloc[-(window + 1)])
        if base == 0:
            return 0.0
        return float((series.iloc[-1] - base) / abs(base))

    def _compute_volatility(self, series: pd.Series, window: int = 20) -> float:
        """Annualised return volatility over last *window* trading days."""
        if len(series) < window + 1:
            return 0.0
        rets = series.pct_change().dropna().iloc[-window:]
        if len(rets) < 5:
            return 0.0
        return float(rets.std() * np.sqrt(252))

    def _classify_shock(
        self,
        change_5d: float,
        volatility: float,
        name: str,
    ) -> ShockType:
        """Classify the commodity price move as a shock type.

        Classification rules:

        * **SUPPLY_SHOCK** — sharp price spike (>5%) in oil or gas with
          elevated volatility (>40% annualised).
        * **DEMAND_SHOCK** — sharp price spike (>5%) in copper or gold
          (demand-driven proxy) with elevated volatility.
        * **PRICE_CRASH** — sharp decline (< -5%).
        * **NEUTRAL** — otherwise.

        Args:
            change_5d:  5-day price percentage change.
            volatility: 20-day annualised volatility.
            name:       Commodity name for rule routing.

        Returns:
            ShockType: Classified shock category.
        """
        high_vol_threshold = 0.40

        if change_5d < -0.05:
            return ShockType.PRICE_CRASH

        if change_5d > 0.05 and volatility > high_vol_threshold:
            if name in ("oil", "gas"):
                return ShockType.SUPPLY_SHOCK
            return ShockType.DEMAND_SHOCK

        if abs(change_5d) > 0.03 and volatility > 0.30:
            return ShockType.DEMAND_SHOCK

        return ShockType.NEUTRAL

    def _compute_stress(
        self,
        oil_shock: ShockType,
        gas_shock: ShockType,
        copper_shock: ShockType,
        gold_shock: ShockType,
        oil_vol: float,
        gas_vol: float,
    ) -> float:
        """Aggregate commodity stress score in [0, 1].

        Higher values indicate greater cross-commodity stress.
        """
        shock_score = sum(
            1 for s in [oil_shock, gas_shock, copper_shock, gold_shock]
            if s != ShockType.NEUTRAL
        ) / 4.0

        vol_score = min(1.0, (oil_vol + gas_vol) / 2.0)
        return round(0.6 * shock_score + 0.4 * vol_score, 4)

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def get_commodity_snapshot(self) -> CommoditySnapshot:
        """Fetch latest commodity prices and derive a snapshot.

        Returns:
            CommoditySnapshot: Latest commodity state with shock labels.
        """
        cache_path = self._cache_dir / "commodity_snapshot.parquet"

        results: dict[str, dict] = {}
        for name, symbol in _COMMODITY_SYMBOLS.items():
            series = self._fetch_price_series(symbol)
            if series.empty:
                results[name] = {
                    "price": 0.0, "change_5d": 0.0,
                    "volatility": 0.0, "shock": ShockType.NEUTRAL,
                }
                continue
            change = self._compute_change(series, window=5)
            vol = self._compute_volatility(series, window=20)
            shock = self._classify_shock(change, vol, name)
            results[name] = {
                "price":     round(float(series.iloc[-1]), 4),
                "change_5d": round(change, 5),
                "volatility": round(vol, 5),
                "shock":     shock,
            }

        stress = self._compute_stress(
            oil_shock=results["oil"]["shock"],
            gas_shock=results["gas"]["shock"],
            copper_shock=results["copper"]["shock"],
            gold_shock=results["gold"]["shock"],
            oil_vol=results["oil"]["volatility"],
            gas_vol=results["gas"]["volatility"],
        )

        snapshot = CommoditySnapshot(
            oil_price=results["oil"]["price"],
            oil_change_5d=results["oil"]["change_5d"],
            oil_volatility=results["oil"]["volatility"],
            oil_shock=results["oil"]["shock"],
            gas_price=results["gas"]["price"],
            gas_change_5d=results["gas"]["change_5d"],
            gas_volatility=results["gas"]["volatility"],
            gas_shock=results["gas"]["shock"],
            copper_price=results["copper"]["price"],
            copper_change_5d=results["copper"]["change_5d"],
            copper_volatility=results["copper"]["volatility"],
            copper_shock=results["copper"]["shock"],
            gold_price=results["gold"]["price"],
            gold_change_5d=results["gold"]["change_5d"],
            gold_volatility=results["gold"]["volatility"],
            gold_shock=results["gold"]["shock"],
            commodity_stress=stress,
        )

        logger.info(
            "CommoditySnapshot | oil=%.1f(%s) gas=%.2f(%s) cu=%.2f(%s) au=%.0f(%s) stress=%.2f",
            snapshot.oil_price, snapshot.oil_shock.value,
            snapshot.gas_price, snapshot.gas_shock.value,
            snapshot.copper_price, snapshot.copper_shock.value,
            snapshot.gold_price, snapshot.gold_shock.value,
            snapshot.commodity_stress,
        )

        try:
            pd.DataFrame([snapshot.to_feature_dict()]).to_parquet(cache_path, index=False)
        except Exception as exc:
            logger.debug("Commodity cache write failed: %s", exc)

        return snapshot


def run_commodity_data() -> CommoditySnapshot:
    """Pipeline-stage entry point.

    Returns:
        CommoditySnapshot: Latest commodity state.
    """
    collector = CommodityDataCollector()
    return collector.get_commodity_snapshot()
