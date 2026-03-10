"""Enhanced model architecture for AlphaLens.

Extends FinBERT sentiment signals with time-series market features,
macroeconomic features, and commodity features using two architectures:

**Simple version** (default)
    sentiment + market + macro + commodity → XGBoost / MLP classifier.

**Advanced version** (optional, requires additional dependencies)
    Multi-encoder fusion model combining FinBERT embeddings, market
    time-series, macro, and commodity encoders → linear fusion → sigmoid.

Usage::

    from src.enhanced_model import FeatureBuilder, SimpleAlphaModel

    builder = FeatureBuilder()
    X, y = builder.build_training_data(news_df, price_df, macro, commodities)

    model = SimpleAlphaModel()
    model.fit(X, y)
    prediction = model.predict(X[-1:])

    # Or use adjust_signal() to post-process an existing RecommendationResult:
    adjusted = adjust_signal_with_context(result, macro, commodities)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from src.macro_data import MacroSnapshot, MacroRegime
from src.commodity_data import CommoditySnapshot, ShockType
from src.transmission_chain import TransmissionChainAnalyser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature names (determines column ordering)
# ---------------------------------------------------------------------------

_MARKET_FEATURES = [
    "market_spy_5d_ret",
    "market_spy_20d_ret",
    "market_spy_vol_20d",
    "market_ticker_5d_ret",
    "market_ticker_20d_ret",
    "market_ticker_vol_20d",
    "market_ticker_beta",
    "market_rsi_14",
]

_SENTIMENT_FEATURES = [
    "sent_signal_score",
    "sent_confidence",
    "sent_lexicon_score",
    "sent_bullish_count",
    "sent_bearish_count",
    "sent_headline_count",
    "sent_confident_count",
]


# ---------------------------------------------------------------------------
# FeatureBuilder
# ---------------------------------------------------------------------------


class FeatureBuilder:
    """Assembles the combined feature matrix from all data sources.

    Args:
        analyser: :class:`~src.transmission_chain.TransmissionChainAnalyser`.
            If None a new instance is created.
    """

    def __init__(self, analyser: TransmissionChainAnalyser | None = None) -> None:
        self._analyser = analyser or TransmissionChainAnalyser()

    # ------------------------------------------------------------------
    # Market feature extraction
    # ------------------------------------------------------------------

    def _fetch_market_features(
        self, ticker: str, spy_prices: pd.Series | None = None
    ) -> dict[str, float]:
        """Fetch rolling market features for *ticker* and SPY.

        Args:
            ticker:     Equity ticker symbol.
            spy_prices: Pre-fetched SPY close prices (optional).

        Returns:
            dict: Market feature values.  Falls back to zeros on failure.
        """
        features: dict[str, float] = {k: 0.0 for k in _MARKET_FEATURES}
        try:
            import yfinance as yf

            tk_df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
            if spy_prices is None:
                spy_df = yf.download("SPY", period="6mo", progress=False, auto_adjust=True)
                spy_prices = spy_df["Close"].dropna() if not spy_df.empty else None

            if tk_df.empty:
                return features

            tk_close = tk_df["Close"].dropna()
            tk_rets = tk_close.pct_change().dropna()

            def _roll_ret(s: pd.Series, w: int) -> float:
                if len(s) < w + 1:
                    return 0.0
                base = float(s.iloc[-(w + 1)])
                return float((s.iloc[-1] - base) / abs(base)) if base != 0 else 0.0

            def _roll_vol(s: pd.Series, w: int) -> float:
                if len(s) < w:
                    return 0.0
                return float(s.iloc[-w:].std() * np.sqrt(252))

            features["market_ticker_5d_ret"]   = _roll_ret(tk_close, 5)
            features["market_ticker_20d_ret"]  = _roll_ret(tk_close, 20)
            features["market_ticker_vol_20d"]  = _roll_vol(tk_rets, 20)

            if spy_prices is not None and len(spy_prices) > 20:
                spy_rets = spy_prices.pct_change().dropna()
                features["market_spy_5d_ret"]  = _roll_ret(spy_prices, 5)
                features["market_spy_20d_ret"] = _roll_ret(spy_prices, 20)
                features["market_spy_vol_20d"] = _roll_vol(spy_rets, 20)

                # Beta: covariance(ticker, spy) / variance(spy) over last 60 days
                if len(tk_rets) >= 60 and len(spy_rets) >= 60:
                    tk_60 = tk_rets.iloc[-60:].values
                    sp_60 = spy_rets.iloc[-60:].values
                    n = min(len(tk_60), len(sp_60))
                    cov = float(np.cov(tk_60[-n:], sp_60[-n:])[0, 1])
                    var_sp = float(np.var(sp_60[-n:]))
                    features["market_ticker_beta"] = cov / var_sp if var_sp != 0 else 1.0

            # RSI(14)
            if len(tk_close) >= 15:
                delta = tk_close.diff().dropna().iloc[-14:]
                gains = delta.clip(lower=0)
                losses = (-delta).clip(lower=0)
                avg_gain = gains.mean()
                avg_loss = losses.mean()
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                features["market_rsi_14"] = float(rsi)

        except Exception as exc:
            logger.warning("Market feature extraction failed for %s: %s", ticker, exc)

        return features

    # ------------------------------------------------------------------
    # Feature vector assembly
    # ------------------------------------------------------------------

    def build_feature_vector(
        self,
        ticker: str,
        signal_score: float,
        confidence: float,
        lexicon_score: float,
        bullish_count: int,
        bearish_count: int,
        headline_count: int,
        confident_count: int,
        macro: MacroSnapshot,
        commodities: CommoditySnapshot,
    ) -> dict[str, float]:
        """Assemble a complete feature dict for one ticker/timepoint.

        Args:
            ticker:          Equity ticker.
            signal_score:    FinBERT composite signal.
            confidence:      Mean prediction confidence.
            lexicon_score:   Financial lexicon score.
            bullish_count:   Number of bullish lexicon phrases.
            bearish_count:   Number of bearish lexicon phrases.
            headline_count:  Total headlines.
            confident_count: Confident headlines.
            macro:           Macroeconomic snapshot.
            commodities:     Commodity snapshot.

        Returns:
            dict[str, float]: All feature columns as a flat dict.
        """
        events = self._analyser.analyse(macro, commodities)
        exposure = self._analyser.get_sector_exposure(ticker, events)
        chain_features = self._analyser.to_feature_dict(events)

        sentiment_features = {
            "sent_signal_score":   signal_score,
            "sent_confidence":     confidence,
            "sent_lexicon_score":  lexicon_score,
            "sent_bullish_count":  float(bullish_count),
            "sent_bearish_count":  float(bearish_count),
            "sent_headline_count": float(headline_count),
            "sent_confident_count": float(confident_count),
        }

        market_features = self._fetch_market_features(ticker)

        exposure_features = {
            "chain_net_exposure":      exposure["net_exposure"],
            "chain_pos_exposure":      exposure["positive_exposure"],
            "chain_neg_exposure":      exposure["negative_exposure"],
        }

        return {
            **sentiment_features,
            **market_features,
            **macro.to_feature_dict(),
            **commodities.to_feature_dict(),
            **chain_features,
            **exposure_features,
        }

    def build_training_data(
        self,
        news_df: pd.DataFrame,
        price_df: pd.DataFrame,
        macro: MacroSnapshot,
        commodities: CommoditySnapshot,
        forward_return_days: int = 5,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build a labelled training dataset from historical news.

        Each row corresponds to a headline date × ticker.  The label is
        ``1`` if the forward return is positive, ``-1`` if negative, and
        ``0`` if within ±0.5%.

        Args:
            news_df:              Combined news DataFrame with columns
                                  ``ticker``, ``date``, ``label``
                                  (0=negative, 1=neutral, 2=positive),
                                  ``confidence``, ``signal_score``.
            price_df:             MultiIndex price DataFrame from
                                  ``backtest.fetch_price_data()``.
            macro:                Current macro snapshot (used as context).
            commodities:          Current commodity snapshot.
            forward_return_days:  Days ahead to compute label return.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Feature matrix X and label y.
        """
        if news_df.empty:
            logger.warning("Empty news_df passed to build_training_data")
            return pd.DataFrame(), pd.Series(dtype=int)

        required_cols = {"ticker", "date"}
        if not required_cols.issubset(set(news_df.columns)):
            logger.warning("news_df missing required columns: %s", required_cols - set(news_df.columns))
            return pd.DataFrame(), pd.Series(dtype=int)

        rows: list[dict] = []
        labels: list[int] = []

        tickers = news_df["ticker"].unique()
        for ticker in tickers:
            ticker_news = news_df[news_df["ticker"] == ticker].copy()
            ticker_news["date"] = pd.to_datetime(ticker_news["date"])

            # Get price data for this ticker
            if isinstance(price_df.columns, pd.MultiIndex):
                if ticker not in price_df.columns.get_level_values(1):
                    continue
                prices = price_df.xs(ticker, level=1, axis=1)["Close"].dropna()
            elif ticker in price_df.columns:
                prices = price_df[ticker].dropna()
            else:
                continue

            prices.index = pd.to_datetime(prices.index).tz_localize(None)

            for _, row in ticker_news.iterrows():
                date = row["date"]
                if pd.isna(date):
                    continue

                # Forward return label
                future_date_candidates = prices.index[prices.index > date]
                if len(future_date_candidates) <= forward_return_days:
                    continue

                future_price = float(prices.loc[future_date_candidates[forward_return_days]])
                # Find closest price at/after event date
                base_candidates = prices.index[prices.index >= date]
                if base_candidates.empty:
                    continue
                base_price = float(prices.loc[base_candidates[0]])
                if base_price == 0:
                    continue

                fwd_ret = (future_price - base_price) / base_price
                if fwd_ret > 0.005:
                    label = 1
                elif fwd_ret < -0.005:
                    label = -1
                else:
                    label = 0

                signal_score = float(row.get("signal_score", 0.0))
                confidence = float(row.get("confidence", 0.5))

                feature_vec = self.build_feature_vector(
                    ticker=ticker,
                    signal_score=signal_score,
                    confidence=confidence,
                    lexicon_score=0.0,
                    bullish_count=0,
                    bearish_count=0,
                    headline_count=1,
                    confident_count=1 if confidence > config.CONFIDENCE_THRESHOLD else 0,
                    macro=macro,
                    commodities=commodities,
                )
                rows.append(feature_vec)
                labels.append(label)

        if not rows:
            return pd.DataFrame(), pd.Series(dtype=int)

        X = pd.DataFrame(rows).fillna(0.0)
        y = pd.Series(labels, name="label")
        logger.info("Built training data: %d samples, %d features", len(X), X.shape[1])
        return X, y


# ---------------------------------------------------------------------------
# SimpleAlphaModel (XGBoost / MLP)
# ---------------------------------------------------------------------------


class SimpleAlphaModel:
    """XGBoost-based enhanced alpha model.

    Falls back to a sklearn RandomForestClassifier if xgboost is not
    installed.

    Args:
        model_path: Path to save/load the trained model pickle.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = model_path or (config.MODEL_DIR / "enhanced_model.pkl")
        self._model = None
        self._feature_names: list[str] = []

    def _get_estimator(self):
        """Return an XGBoost classifier, falling back to RandomForest."""
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=config.RANDOM_SEED,
                verbosity=0,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            logger.info("xgboost not available — using GradientBoostingClassifier")
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=config.RANDOM_SEED,
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SimpleAlphaModel":
        """Train the model on feature matrix X with labels y.

        Args:
            X: Feature DataFrame (output of :meth:`FeatureBuilder.build_training_data`).
            y: Integer labels (1=positive, 0=neutral, -1=negative).

        Returns:
            self (for chaining).
        """
        if X.empty or len(y) == 0:
            logger.warning("Empty training data — model not trained")
            return self

        # Map labels: -1 → 0, 0 → 1, 1 → 2 for multi-class
        y_mapped = y.map({-1: 0, 0: 1, 1: 2}).fillna(1).astype(int)
        self._feature_names = list(X.columns)

        estimator = self._get_estimator()
        estimator.fit(X.values, y_mapped.values)
        self._model = estimator

        # Save
        try:
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._model_path, "wb") as f:
                pickle.dump({"model": self._model, "features": self._feature_names}, f)
            logger.info("Enhanced model saved to %s", self._model_path)
        except Exception as exc:
            logger.warning("Failed to save enhanced model: %s", exc)

        return self

    def load(self) -> bool:
        """Load a previously saved model from disk.

        Returns:
            bool: True if loaded successfully.
        """
        if not self._model_path.exists():
            return False
        try:
            with open(self._model_path, "rb") as f:
                payload = pickle.load(f)
            self._model = payload["model"]
            self._feature_names = payload["features"]
            logger.info("Enhanced model loaded from %s", self._model_path)
            return True
        except Exception as exc:
            logger.warning("Failed to load enhanced model: %s", exc)
            return False

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for features X.

        Args:
            X: Feature DataFrame aligned with training columns.

        Returns:
            np.ndarray: Shape (n_samples, 3) — [P(neg), P(neu), P(pos)].
        """
        if self._model is None:
            return np.array([[1 / 3, 1 / 3, 1 / 3]])
        # Align columns
        aligned = X.reindex(columns=self._feature_names, fill_value=0.0)
        return self._model.predict_proba(aligned.values)

    def get_feature_importances(self) -> pd.Series:
        """Return feature importances as a sorted Series.

        Returns:
            pd.Series: Feature name → importance, descending.
        """
        if self._model is None or not self._feature_names:
            return pd.Series(dtype=float)
        try:
            imps = self._model.feature_importances_
            return pd.Series(imps, index=self._feature_names).sort_values(ascending=False)
        except AttributeError:
            return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Signal adjustment helper
# ---------------------------------------------------------------------------


def adjust_signal_with_context(
    signal_score: float,
    ticker: str,
    macro: MacroSnapshot,
    commodities: CommoditySnapshot,
    analyser: TransmissionChainAnalyser | None = None,
    max_adjustment: float = 0.25,
) -> tuple[float, list[str]]:
    """Adjust an existing FinBERT signal score with macro/commodity context.

    This is a lightweight heuristic adjustment (no trained model required)
    that modifies the base signal score based on:

    1. Macro regime directional bias.
    2. Commodity shock sector exposure.
    3. VIX-based risk-off dampening.

    Args:
        signal_score:    Original FinBERT composite signal.
        ticker:          Equity ticker.
        macro:           Current macro snapshot.
        commodities:     Current commodity snapshot.
        analyser:        Transmission chain analyser instance.
        max_adjustment:  Maximum absolute delta applied to signal_score.

    Returns:
        tuple[float, list[str]]: Adjusted signal score and list of
        adjustment reasons.
    """
    if analyser is None:
        analyser = TransmissionChainAnalyser()

    events = analyser.analyse(macro, commodities)
    exposure = analyser.get_sector_exposure(ticker, events)

    adjustment = 0.0
    reasons: list[str] = []

    # 1. Macro regime bias
    regime_bias: dict[MacroRegime, float] = {
        MacroRegime.RISK_ON:         +0.08,
        MacroRegime.EASING:          +0.06,
        MacroRegime.RISK_OFF:        -0.12,
        MacroRegime.TIGHTENING:      -0.06,
        MacroRegime.INFLATION_SHOCK: -0.08,
        MacroRegime.GROWTH_SLOWDOWN: -0.10,
        MacroRegime.NEUTRAL:          0.0,
    }
    regime_adj = regime_bias.get(macro.regime, 0.0)
    if abs(regime_adj) > 0:
        adjustment += regime_adj
        reasons.append(f"Macro regime ({macro.regime.value}): {regime_adj:+.2f}")

    # 2. Sector exposure from transmission chain
    net_exp = exposure["net_exposure"]
    if abs(net_exp) > 0.2:
        exp_adj = net_exp * 0.10
        adjustment += exp_adj
        reasons.append(f"Sector transmission exposure: {exp_adj:+.2f}")

    # 3. VIX risk-off dampening — reduce signal magnitude in high-vol environments
    if macro.vix > 28:
        vix_damp = 0.85
        original = signal_score + adjustment
        adjustment = original * vix_damp - signal_score
        reasons.append(f"VIX dampening ({macro.vix:.0f}): signal ×{vix_damp:.0%}")

    # 4. Commodity stress penalty
    if commodities.commodity_stress > 0.5:
        stress_adj = -0.04 * commodities.commodity_stress
        adjustment += stress_adj
        reasons.append(f"Commodity stress ({commodities.commodity_stress:.2f}): {stress_adj:+.2f}")

    # Clip total adjustment
    adjustment = float(np.clip(adjustment, -max_adjustment, max_adjustment))
    adjusted_score = float(np.clip(signal_score + adjustment, -1.0, 1.0))

    logger.debug(
        "[%s] Signal adjusted: %.4f → %.4f | reasons: %s",
        ticker, signal_score, adjusted_score, reasons,
    )
    return round(adjusted_score, 6), reasons
