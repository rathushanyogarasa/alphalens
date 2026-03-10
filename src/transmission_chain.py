"""Causal transmission chain modelling for AlphaLens.

Converts macroeconomic and commodity shock events into structured causal
feature columns rather than free-text reasoning.  Each rule in
:data:`TRANSMISSION_RULES` maps an event trigger to:

* ``first_order``   — immediate economic impact.
* ``second_order``  — secondary market effect.
* ``positive_sectors`` — sector ETF proxies expected to benefit.
* ``negative_sectors`` — sector ETF proxies expected to suffer.
* ``risk_level``    — severity of the transmission (0–3).

Example rule::

    "oil_supply_shock": {
        "first_order":        "inflation_pressure",
        "second_order":       "yields_rise",
        "positive_sectors":   ["XLE", "XOM"],
        "negative_sectors":   ["JETS", "XLY"],
        "risk_level":         2,
    }

Usage::

    from src.macro_data import MacroDataCollector
    from src.commodity_data import CommodityDataCollector
    from src.transmission_chain import TransmissionChainAnalyser

    macro = MacroDataCollector().get_macro_snapshot()
    commodities = CommodityDataCollector().get_commodity_snapshot()

    analyser = TransmissionChainAnalyser()
    events = analyser.analyse(macro, commodities)
    exposure = analyser.get_sector_exposure("NVDA", events)
    features = analyser.to_feature_dict(events)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.macro_data import MacroSnapshot, MacroRegime
from src.commodity_data import CommoditySnapshot, ShockType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sector → ticker mapping (GICS-based)
# ---------------------------------------------------------------------------

# Maps sector ETF symbols to a representative list of member tickers
_SECTOR_MEMBERS: dict[str, list[str]] = {
    "XLE":  ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "PXD", "VLO", "MPC", "OXY"],
    "XOM":  ["XOM"],
    "XLB":  ["LIN", "APD", "ECL", "NEM", "FCX", "NUE", "CF", "PPG", "ALB", "IFF"],
    "FCX":  ["FCX"],
    "JETS": ["DAL", "UAL", "AAL", "LUV", "ALK", "JBLU", "SAVE", "RYAAY", "HA", "ALGT"],
    "DAL":  ["DAL"],
    "UAL":  ["UAL"],
    "XLY":  ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "GM", "F"],
    "XLF":  ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "USB", "PNC"],
    "JPM":  ["JPM"],
    "GS":   ["GS"],
    "BAC":  ["BAC"],
    "XLK":  ["AAPL", "MSFT", "NVDA", "AVGO", "META", "ORCL", "CSCO", "ADBE", "AMD", "CRM"],
    "NVDA": ["NVDA"],
    "AAPL": ["AAPL"],
    "MSFT": ["MSFT"],
    "XLV":  ["UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN"],
    "XLU":  ["NEE", "DUK", "SO", "AEP", "EXC", "XEL", "SRE", "PCG", "ED", "WEC"],
    "GLD":  ["GLD", "IAU", "NEM", "GOLD", "AEM", "WPM", "KGC", "HL", "CDE", "AG"],
    "TLT":  ["TLT", "IEF", "SHY", "BND", "AGG"],
}


# ---------------------------------------------------------------------------
# Transmission rules knowledge base
# ---------------------------------------------------------------------------

TRANSMISSION_RULES: dict[str, dict] = {
    # ---------- Oil shocks ----------
    "oil_supply_shock": {
        "trigger":            "Brent crude supply disruption (>+5% 5-day move, elevated vol)",
        "first_order":        "inflation_pressure",
        "second_order":       "yields_rise_consumer_spending_falls",
        "macro_description":  "Supply-side oil shock raises headline CPI; central banks face "
                              "stagflation dilemma; real consumer spending contracts.",
        "positive_sectors":   ["XLE", "XOM"],
        "negative_sectors":   ["JETS", "XLY", "XLB"],
        "risk_level":         2,
    },
    "oil_demand_shock": {
        "trigger":            "Brent crude demand surge (>+5% 5-day, growth-driven)",
        "first_order":        "global_growth_acceleration",
        "second_order":       "risk_assets_re-rate_higher",
        "macro_description":  "Demand-driven oil rally signals strong global industrial "
                              "activity; cyclical sectors benefit; inflation risk moderate.",
        "positive_sectors":   ["XLE", "XLB", "XLY"],
        "negative_sectors":   ["TLT"],
        "risk_level":         1,
    },
    "oil_price_crash": {
        "trigger":            "Brent crude sharp decline (< -5% 5-day)",
        "first_order":        "deflation_risk_energy_sector_stress",
        "second_order":       "consumer_surplus_offsets_energy_losses",
        "macro_description":  "Oil collapse pressures energy producers and high-yield credit; "
                              "lower fuel costs benefit transport and consumer sectors.",
        "positive_sectors":   ["JETS", "XLY"],
        "negative_sectors":   ["XLE"],
        "risk_level":         2,
    },
    # ---------- Natural gas shocks ----------
    "gas_supply_shock": {
        "trigger":            "Natural gas supply shock (>+5% 5-day)",
        "first_order":        "utility_cost_inflation",
        "second_order":       "industrial_margins_squeezed",
        "macro_description":  "Gas supply shock raises industrial energy costs, compressing "
                              "margins across energy-intensive manufacturing.",
        "positive_sectors":   ["XLE"],
        "negative_sectors":   ["XLB", "XLY"],
        "risk_level":         1,
    },
    # ---------- Copper shocks ----------
    "copper_demand_shock": {
        "trigger":            "Copper demand surge (>+5% 5-day, elevated vol)",
        "first_order":        "global_industrial_expansion",
        "second_order":       "infrastructure_spend_accelerating",
        "macro_description":  "Copper rally signals robust global manufacturing and "
                              "construction; EV and grid investment a key driver.",
        "positive_sectors":   ["XLB", "FCX", "XLE"],
        "negative_sectors":   [],
        "risk_level":         0,
    },
    "copper_price_crash": {
        "trigger":            "Copper price crash (< -5% 5-day)",
        "first_order":        "global_growth_slowdown_signal",
        "second_order":       "risk_off_rotation_to_bonds",
        "macro_description":  "Copper collapse historically leads recessions by 3–6 months; "
                              "markets reprice growth expectations downward.",
        "positive_sectors":   ["TLT", "GLD"],
        "negative_sectors":   ["XLB", "XLY", "XLK"],
        "risk_level":         3,
    },
    # ---------- Gold shocks ----------
    "gold_demand_shock": {
        "trigger":            "Gold demand surge (>+5% 5-day, flight-to-safety)",
        "first_order":        "risk_off_safe_haven_bid",
        "second_order":       "equity_de-rating_duration_assets_bid",
        "macro_description":  "Gold surge signals systemic risk aversion; investors rotate "
                              "from equities to safe-haven assets; yields may fall.",
        "positive_sectors":   ["GLD", "TLT", "XLV"],
        "negative_sectors":   ["XLY", "XLK"],
        "risk_level":         2,
    },
    # ---------- Macro regimes ----------
    "tightening_regime": {
        "trigger":            "Fed tightening cycle (fed funds rising, above-trend CPI)",
        "first_order":        "higher_discount_rates",
        "second_order":       "growth_equities_de-rate_value_outperforms",
        "macro_description":  "Rising rates compress multiples for long-duration growth "
                              "stocks; financials benefit from wider net-interest-margins.",
        "positive_sectors":   ["XLF"],
        "negative_sectors":   ["XLK", "XLY", "TLT"],
        "risk_level":         2,
    },
    "easing_regime": {
        "trigger":            "Fed easing cycle (fed funds falling)",
        "first_order":        "lower_discount_rates",
        "second_order":       "growth_equities_re-rate_higher",
        "macro_description":  "Rate cuts expand P/E multiples for growth stocks; "
                              "tech and consumer discretionary lead; financials face NIM "
                              "compression.",
        "positive_sectors":   ["XLK", "XLY", "TLT"],
        "negative_sectors":   ["XLF"],
        "risk_level":         1,
    },
    "inflation_shock_regime": {
        "trigger":            "Inflation shock (CPI >4% YoY with positive surprise)",
        "first_order":        "real_rates_fall_nominal_yields_rise",
        "second_order":       "value_and_commodities_outperform_bonds",
        "macro_description":  "Inflation shock forces central bank hawkishness; TIPS and "
                              "commodities outperform; long-duration bonds lose real value.",
        "positive_sectors":   ["XLE", "XLB", "GLD"],
        "negative_sectors":   ["TLT", "XLK"],
        "risk_level":         3,
    },
    "growth_slowdown_regime": {
        "trigger":            "Growth slowdown (GDP falling, unemployment rising)",
        "first_order":        "earnings_revisions_lower",
        "second_order":       "defensive_sectors_outperform_cyclicals",
        "macro_description":  "Slowing growth triggers earnings downgrades across cyclicals; "
                              "healthcare and utilities outperform; credit spreads widen.",
        "positive_sectors":   ["XLV", "XLU", "GLD"],
        "negative_sectors":   ["XLY", "XLK", "XLE"],
        "risk_level":         2,
    },
    "risk_off_regime": {
        "trigger":            "Risk-off environment (VIX >28 or inverted yield curve)",
        "first_order":        "volatility_spike_liquidity_premium_rises",
        "second_order":       "safe_haven_bid_equity_selling",
        "macro_description":  "Elevated volatility signals systemic stress; investors "
                              "de-risk into treasuries and gold.",
        "positive_sectors":   ["TLT", "GLD", "XLV", "XLU"],
        "negative_sectors":   ["XLY", "XLK", "XLF"],
        "risk_level":         3,
    },
    "risk_on_regime": {
        "trigger":            "Risk-on environment (low VIX, positive growth)",
        "first_order":        "beta_expansion",
        "second_order":       "growth_assets_outperform_defensives",
        "macro_description":  "Low volatility and positive growth tailwinds support "
                              "high-beta equities; investors reach for yield.",
        "positive_sectors":   ["XLK", "XLY", "XLF"],
        "negative_sectors":   ["TLT", "GLD", "XLU"],
        "risk_level":         0,
    },
}


# ---------------------------------------------------------------------------
# TransmissionEvent dataclass
# ---------------------------------------------------------------------------


@dataclass
class TransmissionEvent:
    """A single identified causal transmission event.

    Attributes:
        rule_key:          Key from :data:`TRANSMISSION_RULES`.
        trigger:           Human-readable trigger description.
        first_order:       Immediate economic effect.
        second_order:      Secondary market effect.
        macro_description: Prose explanation of the chain.
        positive_sectors:  Sector proxies expected to benefit.
        negative_sectors:  Sector proxies expected to suffer.
        risk_level:        Severity 0 (none) – 3 (extreme).
    """

    rule_key: str
    trigger: str
    first_order: str
    second_order: str
    macro_description: str
    positive_sectors: list[str] = field(default_factory=list)
    negative_sectors: list[str] = field(default_factory=list)
    risk_level: int = 0


# ---------------------------------------------------------------------------
# TransmissionChainAnalyser
# ---------------------------------------------------------------------------


class TransmissionChainAnalyser:
    """Identifies active transmission events and computes sector exposures.

    Inspects a :class:`~src.macro_data.MacroSnapshot` and a
    :class:`~src.commodity_data.CommoditySnapshot` against
    :data:`TRANSMISSION_RULES` and returns a list of active
    :class:`TransmissionEvent` objects.
    """

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyse(
        self,
        macro: MacroSnapshot,
        commodities: CommoditySnapshot,
    ) -> list[TransmissionEvent]:
        """Identify all active transmission events.

        Args:
            macro:       Current macroeconomic snapshot.
            commodities: Current commodity snapshot.

        Returns:
            list[TransmissionEvent]: Ordered by descending risk level.
        """
        active: list[TransmissionEvent] = []

        def _add(key: str) -> None:
            rule = TRANSMISSION_RULES[key]
            active.append(
                TransmissionEvent(
                    rule_key=key,
                    trigger=rule["trigger"],
                    first_order=rule["first_order"],
                    second_order=rule["second_order"],
                    macro_description=rule["macro_description"],
                    positive_sectors=rule.get("positive_sectors", []),
                    negative_sectors=rule.get("negative_sectors", []),
                    risk_level=rule.get("risk_level", 0),
                )
            )

        # --- Commodity events ---
        if commodities.oil_shock == ShockType.SUPPLY_SHOCK:
            _add("oil_supply_shock")
        elif commodities.oil_shock == ShockType.DEMAND_SHOCK:
            _add("oil_demand_shock")
        elif commodities.oil_shock == ShockType.PRICE_CRASH:
            _add("oil_price_crash")

        if commodities.gas_shock == ShockType.SUPPLY_SHOCK:
            _add("gas_supply_shock")

        if commodities.copper_shock == ShockType.DEMAND_SHOCK:
            _add("copper_demand_shock")
        elif commodities.copper_shock == ShockType.PRICE_CRASH:
            _add("copper_price_crash")

        if commodities.gold_shock == ShockType.DEMAND_SHOCK:
            _add("gold_demand_shock")

        # --- Macro regime events ---
        regime_map: dict[MacroRegime, str] = {
            MacroRegime.TIGHTENING:      "tightening_regime",
            MacroRegime.EASING:          "easing_regime",
            MacroRegime.INFLATION_SHOCK: "inflation_shock_regime",
            MacroRegime.GROWTH_SLOWDOWN: "growth_slowdown_regime",
            MacroRegime.RISK_OFF:        "risk_off_regime",
            MacroRegime.RISK_ON:         "risk_on_regime",
        }
        regime_key = regime_map.get(macro.regime)
        if regime_key:
            _add(regime_key)

        active.sort(key=lambda e: e.risk_level, reverse=True)
        logger.info(
            "TransmissionChain: %d active events | regime=%s",
            len(active),
            macro.regime.value,
        )
        return active

    # ------------------------------------------------------------------
    # Sector exposure
    # ------------------------------------------------------------------

    def get_sector_exposure(
        self,
        ticker: str,
        events: list[TransmissionEvent],
    ) -> dict[str, float]:
        """Compute a net directional sector exposure score for *ticker*.

        For each active event the ticker's membership in positive/negative
        sector lists is used to compute a net score in [-1, +1].

        Args:
            ticker: Equity ticker to evaluate.
            events: Active transmission events from :meth:`analyse`.

        Returns:
            dict: Keys ``net_exposure`` (float), ``positive_exposure``
            (float), ``negative_exposure`` (float), ``active_rules``
            (list of rule keys that matched).
        """
        ticker_upper = ticker.upper()

        # Find which sector proxies the ticker belongs to
        ticker_sectors: set[str] = set()
        for sector, members in _SECTOR_MEMBERS.items():
            if ticker_upper in [m.upper() for m in members]:
                ticker_sectors.add(sector)
        # Always include the ticker itself as its own sector proxy
        ticker_sectors.add(ticker_upper)

        pos_score = 0.0
        neg_score = 0.0
        matching_rules: list[str] = []

        for event in events:
            weight = (event.risk_level + 1) / 4.0  # normalise 0–1
            pos_sectors = {s.upper() for s in event.positive_sectors}
            neg_sectors = {s.upper() for s in event.negative_sectors}

            if ticker_sectors & pos_sectors:
                pos_score += weight
                if event.rule_key not in matching_rules:
                    matching_rules.append(event.rule_key)
            if ticker_sectors & neg_sectors:
                neg_score += weight
                if event.rule_key not in matching_rules:
                    matching_rules.append(event.rule_key)

        # Normalise to [-1, 1]
        total = pos_score + neg_score
        if total == 0:
            net = 0.0
        else:
            net = (pos_score - neg_score) / total

        return {
            "net_exposure":      round(net, 4),
            "positive_exposure": round(pos_score, 4),
            "negative_exposure": round(neg_score, 4),
            "active_rules":      matching_rules,
        }

    # ------------------------------------------------------------------
    # Feature encoding
    # ------------------------------------------------------------------

    def to_feature_dict(self, events: list[TransmissionEvent]) -> dict[str, float]:
        """Encode active events as a flat numeric feature dict.

        Produces one-hot columns per rule key, a count of active events,
        and the maximum risk level present.

        Args:
            events: Active transmission events.

        Returns:
            dict[str, float]: Feature columns for model input.
        """
        features: dict[str, float] = {
            "chain_active_count":    float(len(events)),
            "chain_max_risk_level":  float(max((e.risk_level for e in events), default=0)),
            "chain_avg_risk_level":  float(
                sum(e.risk_level for e in events) / len(events) if events else 0.0
            ),
        }
        for key in TRANSMISSION_RULES:
            features[f"chain_{key}"] = float(any(e.rule_key == key for e in events))
        return features

    def get_risk_flags(
        self,
        ticker: str,
        events: list[TransmissionEvent],
    ) -> list[str]:
        """Generate human-readable risk flag strings for *ticker*.

        Args:
            ticker: Equity ticker symbol.
            events: Active transmission events from :meth:`analyse`.

        Returns:
            list[str]: Up to 5 concise risk flag descriptions.
        """
        exposure = self.get_sector_exposure(ticker, events)
        flags: list[str] = []

        for event in events:
            if event.rule_key in exposure["active_rules"] and event.risk_level >= 1:
                neg_sectors_upper = {s.upper() for s in event.negative_sectors}
                ticker_sectors: set[str] = set()
                for sector, members in _SECTOR_MEMBERS.items():
                    if ticker.upper() in [m.upper() for m in members]:
                        ticker_sectors.add(sector)
                ticker_sectors.add(ticker.upper())

                if ticker_sectors & neg_sectors_upper:
                    flags.append(f"{event.first_order.replace('_', ' ').title()} — "
                                 f"{event.trigger.split('(')[0].strip()}")

        # Add high-level regime flags
        for event in events:
            if event.risk_level >= 2 and event.rule_key not in [
                f.split("—")[0] for f in flags
            ]:
                flags.append(event.second_order.replace("_", " ").replace("-", " "))

        return flags[:5]
