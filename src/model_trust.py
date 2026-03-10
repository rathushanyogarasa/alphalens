"""Model trust and quality scoring for AlphaLens.

Converts model evaluation diagnostics and backtest metrics into a
structured numeric quality report with four composite sub-scores:

* **Predictive Score** — classification accuracy, F1, IC, hit rate.
* **Risk Score**       — drawdown, Calmar, volatility of returns.
* **Robustness Score** — signal decay, rolling Sharpe stability.
* **Overall Quality**  — weighted combination, scaled to 0–10.

Usage::

    from src.model_trust import ModelTrustScorer, run_trust_scoring

    scorer = ModelTrustScorer()
    report = scorer.compute(
        eval_metrics=eval_metrics,
        backtest_metrics=bt_finbert,
        validation_metrics=signal_validation_metrics,
    )
    print(f"Quality: {report.overall_score:.1f} / 10")
    print(report.summary())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ModelQualityReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelQualityReport:
    """Numeric quality report for the AlphaLens model.

    All sub-scores are in [0, 10]; ``overall_score`` is the weighted
    composite.

    Attributes:
        predictive_score: Classification + IC quality (0–10).
        risk_score:       Drawdown + Calmar + volatility score (0–10).
        robustness_score: Signal decay + rolling stability (0–10).
        overall_score:    Weighted composite score (0–10).
        verdict:          Qualitative label: Excellent / Good / Fair / Poor.
        metrics_used:     Dict of raw metric values fed to the scorer.
        component_details: Per-component diagnostic breakdown.
    """

    predictive_score: float = 0.0
    risk_score: float = 0.0
    robustness_score: float = 0.0
    overall_score: float = 0.0
    verdict: str = "Unscored"
    metrics_used: dict = field(default_factory=dict)
    component_details: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a formatted multi-line summary string."""
        lines = [
            f"Model Quality Report",
            f"{'─' * 36}",
            f"  Predictive Score : {self.predictive_score:.1f} / 10",
            f"  Risk Score       : {self.risk_score:.1f} / 10",
            f"  Robustness Score : {self.robustness_score:.1f} / 10",
            f"  {'─' * 32}",
            f"  Overall Quality  : {self.overall_score:.1f} / 10  [{self.verdict}]",
        ]
        if self.component_details:
            lines.append("")
            lines.append("  Diagnostics:")
            for key, val in self.component_details.items():
                lines.append(f"    {key:<28} {val}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "predictive_score":  round(self.predictive_score, 2),
            "risk_score":        round(self.risk_score, 2),
            "robustness_score":  round(self.robustness_score, 2),
            "overall_score":     round(self.overall_score, 2),
            "verdict":           self.verdict,
            **{f"metric_{k}": v for k, v in self.metrics_used.items()},
        }


# ---------------------------------------------------------------------------
# ModelTrustScorer
# ---------------------------------------------------------------------------


class ModelTrustScorer:
    """Converts evaluation and backtest diagnostics into quality scores.

    Sub-score weights (sum to 1.0):

    * predictive: 0.40
    * risk:       0.35
    * robustness: 0.25
    """

    _WEIGHTS = {
        "predictive":  0.40,
        "risk":        0.35,
        "robustness":  0.25,
    }

    # ------------------------------------------------------------------
    # Predictive score
    # ------------------------------------------------------------------

    def _compute_predictive_score(
        self,
        accuracy: float,
        macro_f1: float,
        weighted_f1: float,
        ic: float,
        hit_rate: float,
    ) -> tuple[float, dict]:
        """Score predictive quality from classification + signal metrics.

        Scoring bands (each 0–10):

        * Accuracy (40%)
        * Macro F1 (30%)
        * IC absolute value (20%)
        * Hit rate (10%)

        Args:
            accuracy:    Model accuracy (0–1).
            macro_f1:    Macro-averaged F1 (0–1).
            weighted_f1: Weighted F1 (0–1).
            ic:          Information coefficient (Spearman, −1 to 1).
            hit_rate:    Directional hit rate (0–1).

        Returns:
            tuple[float, dict]: Sub-score in [0, 10] and details.
        """
        def _score_accuracy(a: float) -> float:
            # Baseline for 3-class random is 0.33; 0.70+ is excellent
            if a >= 0.70:
                return 10.0
            if a >= 0.60:
                return 7.0 + (a - 0.60) / 0.10 * 3.0
            if a >= 0.45:
                return 4.0 + (a - 0.45) / 0.15 * 3.0
            if a >= 0.33:
                return 1.0 + (a - 0.33) / 0.12 * 3.0
            return 0.0

        def _score_f1(f: float) -> float:
            if f >= 0.65:
                return 10.0
            if f >= 0.50:
                return 6.0 + (f - 0.50) / 0.15 * 4.0
            if f >= 0.35:
                return 2.0 + (f - 0.35) / 0.15 * 4.0
            return max(0.0, f / 0.35 * 2.0)

        def _score_ic(ic_val: float) -> float:
            abs_ic = abs(ic_val)
            if abs_ic >= 0.10:
                return 10.0
            if abs_ic >= 0.05:
                return 5.0 + (abs_ic - 0.05) / 0.05 * 5.0
            return abs_ic / 0.05 * 5.0

        def _score_hit_rate(h: float) -> float:
            if h >= 0.60:
                return 10.0
            if h >= 0.55:
                return 7.0 + (h - 0.55) / 0.05 * 3.0
            if h >= 0.50:
                return 4.0 + (h - 0.50) / 0.05 * 3.0
            return max(0.0, h / 0.50 * 4.0)

        acc_s = _score_accuracy(accuracy)
        f1_s = _score_f1((macro_f1 + weighted_f1) / 2)
        ic_s = _score_ic(ic)
        hr_s = _score_hit_rate(hit_rate)

        composite = acc_s * 0.40 + f1_s * 0.30 + ic_s * 0.20 + hr_s * 0.10
        details = {
            "accuracy_score":   f"{acc_s:.1f}/10 (acc={accuracy:.3f})",
            "f1_score":         f"{f1_s:.1f}/10 (macro_f1={macro_f1:.3f})",
            "ic_score":         f"{ic_s:.1f}/10 (IC={ic:.4f})",
            "hit_rate_score":   f"{hr_s:.1f}/10 (hit={hit_rate:.3f})",
        }
        return round(composite, 2), details

    # ------------------------------------------------------------------
    # Risk score
    # ------------------------------------------------------------------

    def _compute_risk_score(
        self,
        sharpe: float,
        max_drawdown: float,
        calmar: float,
        annualised_return: float,
        win_rate: float,
    ) -> tuple[float, dict]:
        """Score risk-adjusted performance.

        Higher Sharpe, lower drawdown, higher Calmar = better score.

        Args:
            sharpe:            Annualised Sharpe ratio.
            max_drawdown:      Maximum drawdown (positive fraction, e.g. 0.25).
            calmar:            Calmar ratio (return / max_drawdown).
            annualised_return: Annualised strategy return.
            win_rate:          Fraction of winning trades.

        Returns:
            tuple[float, dict]: Sub-score in [0, 10] and details.
        """
        def _score_sharpe(s: float) -> float:
            if s >= 2.0:  return 10.0
            if s >= 1.5:  return 8.0 + (s - 1.5) / 0.5 * 2.0
            if s >= 1.0:  return 6.0 + (s - 1.0) / 0.5 * 2.0
            if s >= 0.5:  return 3.0 + (s - 0.5) / 0.5 * 3.0
            if s >= 0.0:  return s / 0.5 * 3.0
            return 0.0

        def _score_drawdown(dd: float) -> float:
            # Lower drawdown = better
            if dd <= 0.05:  return 10.0
            if dd <= 0.10:  return 8.0 - (dd - 0.05) / 0.05 * 2.0
            if dd <= 0.20:  return 5.0 - (dd - 0.10) / 0.10 * 3.0
            if dd <= 0.40:  return 2.0 - (dd - 0.20) / 0.20 * 2.0
            return 0.0

        def _score_calmar(c: float) -> float:
            if c >= 3.0:  return 10.0
            if c >= 1.5:  return 6.0 + (c - 1.5) / 1.5 * 4.0
            if c >= 0.5:  return 2.0 + (c - 0.5) / 1.0 * 4.0
            return max(0.0, c / 0.5 * 2.0)

        sh_s = _score_sharpe(sharpe)
        dd_s = _score_drawdown(abs(max_drawdown))
        cal_s = _score_calmar(calmar)

        composite = sh_s * 0.45 + dd_s * 0.35 + cal_s * 0.20
        details = {
            "sharpe_score":    f"{sh_s:.1f}/10 (Sharpe={sharpe:.2f})",
            "drawdown_score":  f"{dd_s:.1f}/10 (MaxDD={max_drawdown:.1%})",
            "calmar_score":    f"{cal_s:.1f}/10 (Calmar={calmar:.2f})",
        }
        return round(composite, 2), details

    # ------------------------------------------------------------------
    # Robustness score
    # ------------------------------------------------------------------

    def _compute_robustness_score(
        self,
        ic_decay_1d: float,
        ic_decay_5d: float,
        ic_decay_10d: float,
        rolling_sharpe_std: float,
        signal_correlation: float,
    ) -> tuple[float, dict]:
        """Score signal robustness and stability.

        Args:
            ic_decay_1d:        IC at 1-day horizon.
            ic_decay_5d:        IC at 5-day horizon.
            ic_decay_10d:       IC at 10-day horizon.
            rolling_sharpe_std: Std dev of rolling 60-day Sharpe (lower = more stable).
            signal_correlation: Correlation between consecutive signal scores (lower = less crowded).

        Returns:
            tuple[float, dict]: Sub-score in [0, 10] and details.
        """
        # IC decay shape: good signal persists (decay should be gradual)
        ic_decay_score = 0.0
        if ic_decay_1d != 0:
            decay_ratio = abs(ic_decay_5d) / max(abs(ic_decay_1d), 1e-6)
            # Ideal: 5d IC still > 50% of 1d IC
            ic_decay_score = min(10.0, decay_ratio * 10.0)

        # Rolling Sharpe stability — lower std = more consistent
        if rolling_sharpe_std <= 0.3:
            stability_score = 10.0
        elif rolling_sharpe_std <= 0.8:
            stability_score = 10.0 - (rolling_sharpe_std - 0.3) / 0.5 * 5.0
        else:
            stability_score = max(0.0, 5.0 - (rolling_sharpe_std - 0.8) * 5.0)

        # Signal autocorrelation — very high means signals are too sticky
        if abs(signal_correlation) <= 0.3:
            corr_score = 10.0
        elif abs(signal_correlation) <= 0.7:
            corr_score = 10.0 - (abs(signal_correlation) - 0.3) / 0.4 * 5.0
        else:
            corr_score = max(0.0, 5.0 - (abs(signal_correlation) - 0.7) * 10.0)

        composite = ic_decay_score * 0.40 + stability_score * 0.35 + corr_score * 0.25
        details = {
            "ic_decay_score":    f"{ic_decay_score:.1f}/10 (IC_5d/IC_1d={abs(ic_decay_5d) / max(abs(ic_decay_1d), 1e-6):.2f})",
            "stability_score":   f"{stability_score:.1f}/10 (Sharpe_std={rolling_sharpe_std:.2f})",
            "corr_score":        f"{corr_score:.1f}/10 (signal_corr={signal_correlation:.2f})",
        }
        return round(composite, 2), details

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------

    @staticmethod
    def _get_verdict(score: float) -> str:
        if score >= 7.5:
            return "Excellent"
        if score >= 5.5:
            return "Good"
        if score >= 3.5:
            return "Fair"
        return "Poor"

    # ------------------------------------------------------------------
    # Main compute method
    # ------------------------------------------------------------------

    def compute(
        self,
        eval_metrics: dict | None = None,
        backtest_metrics: dict | None = None,
        validation_metrics: dict | None = None,
    ) -> ModelQualityReport:
        """Compute all quality scores and return a :class:`ModelQualityReport`.

        Args:
            eval_metrics:       Output of :func:`~src.evaluate.run_evaluation`.
                                Expected keys: ``finbert.accuracy``,
                                ``finbert.macro_f1``, ``finbert.weighted_f1``.
            backtest_metrics:   Output of :func:`~src.backtest.run_backtest`.
                                Expected keys: ``sharpe_ratio``,
                                ``max_drawdown``, ``calmar_ratio``,
                                ``annualised_return``, ``win_rate``.
            validation_metrics: Output of signal validation.  Expected keys:
                                ``ic``, ``hit_rate``, ``ic_decay_1``,
                                ``ic_decay_5``, ``ic_decay_10``.

        Returns:
            ModelQualityReport: Populated quality report.
        """
        eval_metrics = eval_metrics or {}
        backtest_metrics = backtest_metrics or {}
        validation_metrics = validation_metrics or {}

        # --- Extract with safe defaults ---
        fb = eval_metrics.get("finbert", {})
        accuracy     = float(fb.get("accuracy", 0.40))
        macro_f1     = float(fb.get("macro_f1", 0.35))
        weighted_f1  = float(fb.get("weighted_f1", 0.40))

        sharpe       = float(backtest_metrics.get("sharpe_ratio", 0.5))
        max_dd       = float(backtest_metrics.get("max_drawdown", 0.20))
        calmar       = float(backtest_metrics.get("calmar_ratio", 0.5))
        ann_ret      = float(backtest_metrics.get("annualised_return", 0.05))
        win_rate     = float(backtest_metrics.get("win_rate", 0.50))

        # --- Normalise validation_metrics: handle run_signal_validation output ---
        # hit_rate may be a dict {"overall": float, "buy": float, ...}
        _hit_rate_raw = validation_metrics.get("hit_rate", 0.50)
        if isinstance(_hit_rate_raw, dict):
            _hit_rate_raw = _hit_rate_raw.get("overall", 0.50)
        # ic_decay may be a DataFrame with columns [lag, ic, ...]
        _ic_decay_df = validation_metrics.get("ic_decay")
        _ic_map: dict = {}
        if _ic_decay_df is not None and hasattr(_ic_decay_df, "set_index"):
            try:
                _ic_map = _ic_decay_df.set_index("lag")["ic"].to_dict()
            except Exception:
                pass

        _ic_default = float(validation_metrics.get("ic", _ic_map.get(1, 0.02)))

        ic           = _ic_default
        hit_rate     = float(_hit_rate_raw)
        ic_1d        = float(validation_metrics.get("ic_decay_1", _ic_map.get(1, ic)))
        ic_5d        = float(validation_metrics.get("ic_decay_5", _ic_map.get(5, ic * 0.7)))
        ic_10d       = float(validation_metrics.get("ic_decay_10", _ic_map.get(10, ic * 0.4)))
        rolling_sh_std = float(validation_metrics.get("rolling_sharpe_std", 0.5))
        sig_corr     = float(validation_metrics.get("signal_autocorrelation", 0.3))

        # --- Compute sub-scores ---
        pred_score, pred_details = self._compute_predictive_score(
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            ic=ic,
            hit_rate=hit_rate,
        )
        risk_score, risk_details = self._compute_risk_score(
            sharpe=sharpe,
            max_drawdown=max_dd,
            calmar=calmar,
            annualised_return=ann_ret,
            win_rate=win_rate,
        )
        rob_score, rob_details = self._compute_robustness_score(
            ic_decay_1d=ic_1d,
            ic_decay_5d=ic_5d,
            ic_decay_10d=ic_10d,
            rolling_sharpe_std=rolling_sh_std,
            signal_correlation=sig_corr,
        )

        # --- Weighted composite ---
        overall = (
            pred_score * self._WEIGHTS["predictive"]
            + risk_score * self._WEIGHTS["risk"]
            + rob_score * self._WEIGHTS["robustness"]
        )
        overall = round(min(10.0, max(0.0, overall)), 2)

        verdict = self._get_verdict(overall)

        metrics_used = {
            "accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "sharpe": sharpe, "max_drawdown": max_dd, "calmar": calmar,
            "annualised_return": ann_ret, "win_rate": win_rate,
            "ic": ic, "hit_rate": hit_rate,
        }

        report = ModelQualityReport(
            predictive_score=pred_score,
            risk_score=risk_score,
            robustness_score=rob_score,
            overall_score=overall,
            verdict=verdict,
            metrics_used=metrics_used,
            component_details={**pred_details, **risk_details, **rob_details},
        )

        logger.info(
            "ModelTrustScore | overall=%.1f (%s) | pred=%.1f | risk=%.1f | robust=%.1f",
            overall, verdict, pred_score, risk_score, rob_score,
        )

        # Persist to metrics dir
        try:
            out_path = config.METRICS_DIR / "model_quality_report.json"
            with open(out_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
        except Exception as exc:
            logger.debug("Quality report save failed: %s", exc)

        return report


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_trust_scoring(
    eval_metrics: dict | None = None,
    backtest_metrics: dict | None = None,
    validation_metrics: dict | None = None,
) -> ModelQualityReport:
    """Pipeline-stage entry point.

    Loads saved metric files from disk if arguments are not provided,
    then delegates to :class:`ModelTrustScorer`.

    Args:
        eval_metrics:       Evaluation metrics dict (optional).
        backtest_metrics:   Backtest metrics dict (optional).
        validation_metrics: Validation metrics dict (optional).

    Returns:
        ModelQualityReport: Populated quality report.
    """
    # Auto-load from saved files if not provided
    if eval_metrics is None:
        eval_path = config.METRICS_DIR / "eval_metrics.csv"
        if eval_path.exists():
            try:
                df = pd.read_csv(eval_path, index_col=0)
                eval_metrics = {
                    "finbert": {
                        "accuracy":     float(df.loc["finbert", "accuracy"]) if "finbert" in df.index else 0.40,
                        "macro_f1":     float(df.loc["finbert", "macro_f1"]) if "finbert" in df.index else 0.35,
                        "weighted_f1":  float(df.loc["finbert", "weighted_f1"]) if "finbert" in df.index else 0.40,
                    }
                }
            except Exception as exc:
                logger.debug("Could not load eval_metrics.csv: %s", exc)

    if backtest_metrics is None:
        bt_path = config.METRICS_DIR / "backtest_finbert.json"
        if bt_path.exists():
            try:
                with open(bt_path) as f:
                    backtest_metrics = json.load(f)
            except Exception as exc:
                logger.debug("Could not load backtest_finbert.json: %s", exc)

    scorer = ModelTrustScorer()
    return scorer.compute(
        eval_metrics=eval_metrics,
        backtest_metrics=backtest_metrics,
        validation_metrics=validation_metrics,
    )
