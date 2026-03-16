"""Financial news sentiment scoring via a domain-specific lexicon.

This module implements a rules-based sentiment scoring layer that complements
VADER and FinBERT models with a curated financial-domain lexicon, combination
bonuses, and negation handling aligned with professional quant-fund methodology.

Typical usage::

    from src.financial_lexicon import score_headline, score_corpus, get_top_signals

    result = score_headline("Apple beats expectations and raises guidance")
    df     = score_corpus(headlines)
    sigs   = get_top_signals(headlines, n=20)
"""

from __future__ import annotations

import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config path bootstrap
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402  (after sys.path manipulation)

# ---------------------------------------------------------------------------
# Keyword Dictionaries
# ---------------------------------------------------------------------------

BULLISH_KEYWORDS: dict[str, float] = {
    # --- Earnings & Financial Performance (+3 to +5) ---
    "beat expectations": 3,
    "earnings beat": 3,
    "revenue growth": 3,
    "record profits": 4,
    "record revenue": 4,
    "strong guidance": 3,
    "raised outlook": 4,
    "raised guidance": 4,
    "raises guidance": 4,
    "margin expansion": 3,
    "accelerating growth": 3,
    "record earnings": 4,
    "exceeded expectations": 3,
    "above expectations": 3,
    "topped estimates": 3,
    "blowout earnings": 5,
    "profit surge": 3,
    # --- Business Expansion (+2 to +3) ---
    "strategic partnership": 2,
    "major contract": 3,
    "market expansion": 2,
    "new product launch": 2,
    "acquisition synergy": 2,
    "market share gains": 3,
    "scaling operations": 2,
    "major deal": 2,
    "landmark deal": 3,
    "exclusive contract": 3,
    "signed agreement": 2,
    # --- Investor & Capital Signals (+2 to +4) ---
    "share buyback": 3,
    "stock buyback": 3,
    "buyback program": 3,
    "dividend increase": 3,
    "dividend hike": 3,
    "special dividend": 3,
    "insider buying": 2,
    "institutional buying": 2,
    "strong demand": 2,
    "oversubscribed": 3,
    # --- Regulatory & Industry Tailwinds (+2 to +4) ---
    "government approval": 3,
    "regulatory approval": 4,
    "fda approval": 4,
    "regulatory clearance": 3,
    "tax incentives": 2,
    "policy support": 2,
    "subsidies": 2,
    "approved": 2,
    # --- Very Bullish — Extreme signals (+4 to +5) ---
    "breakthrough": 4,
    "first of its kind": 5,
    "patent approval": 4,
    "blockbuster": 4,
    "transformative": 3,
    "game changer": 4,
    "game-changing": 4,
    "revolutionary": 3,
    # --- Price action verbs (+2 to +3) ---
    "surges": 2, "surged": 2, "surge": 2,
    "soars": 3, "soared": 3, "soar": 3,
    "jumps": 2, "jumped": 2, "jump": 2,
    "rallies": 2, "rallied": 2, "rally": 2,
    "climbs": 2, "climbed": 2,
    "skyrockets": 3, "skyrocketed": 3,
    "outperforms": 2, "outperformed": 2, "outperforming": 2,
    "beats": 2, "beat": 2,
    # --- Analyst actions — bullish (+2 to +3) ---
    "raises price target": 3, "raised price target": 3,
    "raises pt": 2, "raised pt": 2,
    "price target raised": 3, "price target raise": 3,
    "target on": 2,  # catches "raises target on X"
    "raises target": 3, "raised target": 3,
    "upgrades": 2, "upgraded": 2, "upgrade": 2,
    "initiates with buy": 3, "initiated with buy": 3,
    "strong buy": 3, "outperform rating": 2,
    "overweight rating": 2, "overweight": 2,
    "buy rating": 2, "reiterate buy": 2, "reiterates buy": 2,
    "price target increase": 2, "target raised": 2,
    "upped target": 2, "boosts target": 2,
    # --- Capital allocation — bullish (+2 to +3) ---
    "raises dividend": 3, "raised dividend": 3, "dividend raised": 3,
    "raised its dividend": 3, "raises its dividend": 3,
    "buyback": 2, "repurchase": 2,
    "ipo": 2, "goes public": 2,
    "massive investment": 2, "committed": 2,
    # --- Growth signals (+1 to +3) ---
    "record high": 3, "all time high": 3, "all-time high": 3,
    "profit rises": 3, "earnings rise": 3, "revenue rises": 3,
    "strong results": 3, "strong earnings": 3,
    "earnings growth": 3, "profit growth": 3,
    "strong quarter": 3, "solid quarter": 2,
    "beats estimates": 3, "topped forecasts": 3,
    "doubles down": 2, "expands": 1,
    "accelerates": 2, "acceleration": 2,
    "demand surge": 3, "boom": 2,
    "milestone": 2, "record quarter": 4,
    "best ever": 4, "best quarter": 4,
}

BEARISH_KEYWORDS: dict[str, float] = {
    # --- Earnings Weakness (-3 to -5) ---
    "missed expectations": -3,
    "missed estimates": -3,
    "below expectations": -3,
    "earnings miss": -3,
    "revenue decline": -3,
    "profit warning": -4,
    "lower guidance": -4,
    "cuts guidance": -4,
    "reduced outlook": -4,
    "margin pressure": -3,
    "earnings shortfall": -4,
    "weak guidance": -3,
    "disappointing results": -3,
    "missed revenue": -3,
    # --- Financial Risk (-3 to -5) ---
    "debt concerns": -3,
    "liquidity issues": -4,
    "credit downgrade": -4,
    "cash burn": -3,
    "covenant breach": -5,
    "going concern": -5,
    "default risk": -5,
    "debt crisis": -4,
    "credit crisis": -4,
    "rating downgrade": -4,
    "rating downgraded": -3,
    # --- Legal & Regulatory Problems (-3 to -5) ---
    "sec investigation": -5,
    "sec probe": -5,
    "antitrust lawsuit": -4,
    "regulatory fine": -3,
    "compliance violation": -3,
    "class action": -4,
    "fraud investigation": -5,
    "accounting fraud": -5,
    "restatement": -5,
    "securities fraud": -5,
    "under investigation": -4,
    "investigated": -3,
    "lawsuit": -2,
    "fined": -3,
    "penalty": -2,
    # --- Business Weakness (-2 to -5) ---
    "layoffs": -3,
    "workforce reduction": -3,
    "job cuts": -3,
    "demand slowdown": -3,
    "supply chain disruption": -3,
    "customer losses": -3,
    "losing market share": -4,
    "declining sales": -3,
    "revenue miss": -3,
    "plant closure": -3,
    "store closures": -2,
    "restructuring charges": -2,
    # --- Very Bearish — Extreme signals (-4 to -5) ---
    "bankruptcy": -5,
    "chapter 11": -5,
    "delisting": -5,
    "delisted": -5,
    "accounting irregularities": -5,
    "ponzi": -5,
    "fraud": -4,
    "recall": -3,
    "product recall": -4,
    "suspended": -3,
    # --- Price action verbs (-2 to -4) ---
    "tumbles": -3, "tumbled": -3, "tumble": -3,
    "plummets": -3, "plummeted": -3, "plummet": -3,
    "plunges": -3, "plunged": -3, "plunge": -3,
    "dives": -3, "dived": -3, "dive": -3,
    "sinks": -2, "sank": -2, "sink": -2,
    "sinking": -2, "slumps": -3, "slumped": -3, "slump": -3,
    "crashes": -4, "crashed": -4, "crash": -4,
    "drops": -2, "dropped": -2,
    "declines": -2, "declined": -2,
    "falls sharply": -3, "sharp decline": -3,
    "sells off": -2, "sell-off": -2, "selloff": -2,
    "rout": -3, "bloodbath": -4,
    # --- Analyst actions — bearish (-2 to -3) ---
    "cuts price target": -3, "cut price target": -3,
    "cuts pt": -2, "cut pt": -2,
    "price target cut": -3, "price target reduced": -3,
    "downgrades": -2, "downgraded": -2, "downgrade": -2,
    "sell rating": -3, "underweight rating": -2,
    "underweight": -2, "underperform": -2,
    "avoid": -1, "reduce rating": -2,
    "slashes target": -3, "slashed target": -3,
    # --- Risk & concern language (-1 to -3) ---
    "bubble fears": -3, "bubble": -2,
    "fears": -1, "fear": -1, "worries": -2, "worry": -1,
    "concerns": -1, "concern": -1,
    "threatens": -2, "threaten": -2, "threatened": -2,
    "threatens to hurt": -3, "at risk": -2,
    "surging oil": -2, "surging costs": -2, "surging inflation": -2,
    "risks": -1, "warning": -2, "warns": -2,
    "disappoints": -3, "disappointing": -2,
    "struggles": -2, "struggling": -2,
    "pressure": -1, "headwinds": -2, "headwind": -2,
    "uncertainty": -1, "volatile": -1, "volatility": -1,
    # --- Workforce & operations (-2 to -3) ---
    "mass layoffs": -4, "announces layoffs": -3,
    "cuts jobs": -3, "cutting jobs": -3,
    "workforce cut": -3, "job losses": -3,
    "plant shutdown": -3, "factory closure": -3,
}

NEUTRAL_KEYWORDS: dict[str, float] = {
    "in line with expectations": 0,
    "meets expectations": 0,
    "maintains guidance": 0,
    "stable demand": 0,
    "operational update": 0,
    "strategic review": 0,
    "reorganisation": 0,
    "reorganization": 0,
    "executive transition": 0,
    "board appointment": 0,
    "as expected": 0,
    "in line with": 0,
}

# ---------------------------------------------------------------------------
# Combination Bonuses
# ---------------------------------------------------------------------------

COMBINATION_BONUSES: list[tuple[list[str], float, str]] = [
    # (required_phrases, bonus_score, description)
    (["earnings beat", "raised guidance"], +3.0, "beat+raised guidance"),
    (["earnings beat", "record revenue"], +2.0, "beat+record revenue"),
    (["raised guidance", "record"], +1.5, "raised guidance+record"),
    (["profit warning", "lower guidance"], -3.0, "warning+lower guidance"),
    (["missed expectations", "cuts guidance"], -3.0, "miss+cuts guidance"),
    (["sec investigation", "accounting fraud"], -3.0, "sec+fraud"),
    (["layoffs", "lower guidance"], -2.0, "layoffs+lower guidance"),
    (["buyback", "dividend"],                              +2.0, "buyback+dividend"),
    (["beats", "raised guidance"],                         +3.0, "beats+raised guidance"),
    (["record", "raised guidance"],                        +2.0, "record+raised guidance"),
    (["upgraded", "raises price target"],                  +2.0, "upgrade+target raise"),
    (["downgraded", "cuts price target"],                  -2.0, "downgrade+target cut"),
    (["tumbles", "profit warning"],                        -2.0, "tumbles+profit warning"),
    (["plummets", "lower guidance"],                       -2.0, "plummets+lower guidance"),
    (["bubble", "fears"],                                  -1.5, "bubble+fears"),
    (["layoffs", "revenue decline"],                       -2.0, "layoffs+revenue decline"),
    (["buyback", "record"],                                +2.0, "buyback+record"),
    (["dividend", "record"],                               +1.5, "dividend+record"),
]

# ---------------------------------------------------------------------------
# Negation configuration
# ---------------------------------------------------------------------------

_NEGATION_WORDS: list[str] = [
    "not",
    "no",
    "never",
    "fails",
    "failed",
    "unable",
    "without",
    "disappoints",
]

_NEGATION_WINDOW: int = 3  # tokens before matched phrase to scan
_NEGATION_MULTIPLIER: float = -0.8

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Pre-sort all dictionaries by phrase length (longest first) once at import.
_ALL_DICTS: list[tuple[dict[str, float], str]] = [
    (BULLISH_KEYWORDS, "bullish"),
    (BEARISH_KEYWORDS, "bearish"),
    (NEUTRAL_KEYWORDS, "neutral"),
]

_SORTED_PHRASES: list[tuple[str, float, str]] = sorted(
    [
        (phrase, score, label)
        for d, label in _ALL_DICTS
        for phrase, score in d.items()
    ],
    key=lambda t: len(t[0]),
    reverse=True,
)


def _tokenize(text: str) -> list[str]:
    """Split lowercased text into word tokens, stripping punctuation.

    Args:
        text: Raw input string.

    Returns:
        List of lowercase word tokens.
    """
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def _has_negation_before(
    text_lower: str,
    phrase: str,
    window: int = _NEGATION_WINDOW,
) -> bool:
    """Check whether a negation word appears within *window* tokens before *phrase*.

    The check operates on the token stream to avoid substring false-positives
    (e.g. "notable" matching "not").

    Args:
        text_lower: Fully lowercased source text.
        phrase: Keyword phrase that was matched.
        window: Number of tokens before the phrase start to scan.

    Returns:
        True if a negation word is found within the window.
    """
    tokens = _tokenize(text_lower)
    phrase_tokens = _tokenize(phrase)
    if not phrase_tokens:
        return False

    for i, token in enumerate(tokens):
        # Find where the phrase token sequence begins
        if (
            token == phrase_tokens[0]
            and tokens[i : i + len(phrase_tokens)] == phrase_tokens
        ):
            start_idx = i
            window_tokens = tokens[max(0, start_idx - window) : start_idx]
            if any(neg in window_tokens for neg in _NEGATION_WORDS):
                return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_headline(text: str) -> dict[str, Any]:
    """Score a single news headline using the financial lexicon.

    Applies longest-match-first scanning across bullish, bearish, and neutral
    dictionaries, applies negation flipping, sums combination bonuses, and
    normalises the result to [-1, 1] via tanh.

    Args:
        text: Raw headline string. May be mixed-case.

    Returns:
        A dictionary with the following keys:

        - ``raw_score`` (float): Sum of matched keyword scores and bonuses.
        - ``normalised_score`` (float): ``tanh(raw_score / 5.0)`` clipped to
          ``[-1.0, 1.0]``.
        - ``label`` (str): ``"bullish"``, ``"neutral"``, or ``"bearish"``.
        - ``confidence`` (float): ``abs(normalised_score)``, range ``[0, 1]``.
        - ``matched_bullish`` (list[str]): Bullish phrases detected.
        - ``matched_bearish`` (list[str]): Bearish phrases detected.
        - ``matched_neutral`` (list[str]): Neutral phrases detected.
        - ``combination_bonuses`` (list[str]): Combination bonus descriptions
          triggered.
    """
    if not text or not text.strip():
        logger.debug("score_headline received empty text; returning zero result.")
        return {
            "raw_score": 0.0,
            "normalised_score": 0.0,
            "label": "neutral",
            "confidence": 0.0,
            "matched_bullish": [],
            "matched_bearish": [],
            "matched_neutral": [],
            "combination_bonuses": [],
        }

    text_lower = text.lower()

    matched_bullish: list[str] = []
    matched_bearish: list[str] = []
    matched_neutral: list[str] = []
    raw_score: float = 0.0

    # Track character positions already consumed by a longer phrase match so
    # that shorter sub-phrases are not double-counted.
    consumed_spans: list[tuple[int, int]] = []

    for phrase, score, label in _SORTED_PHRASES:
        search_start = 0
        while True:
            idx = text_lower.find(phrase, search_start)
            if idx == -1:
                break
            end_idx = idx + len(phrase)

            # Boundary check: phrase must not be embedded in a longer word.
            before_ok = idx == 0 or not text_lower[idx - 1].isalpha()
            after_ok = end_idx == len(text_lower) or not text_lower[end_idx].isalpha()

            if not (before_ok and after_ok):
                search_start = idx + 1
                continue

            # Skip if this span is already covered by a longer match.
            overlap = any(
                s <= idx < e or s < end_idx <= e
                for s, e in consumed_spans
            )
            if overlap:
                search_start = idx + 1
                continue

            consumed_spans.append((idx, end_idx))

            # Negation handling
            effective_score = score
            if _has_negation_before(text_lower, phrase):
                effective_score = score * _NEGATION_MULTIPLIER
                logger.debug(
                    "Negation flipped score for '%s': %.1f -> %.1f",
                    phrase,
                    score,
                    effective_score,
                )

            raw_score += effective_score

            if label == "bullish":
                matched_bullish.append(phrase)
            elif label == "bearish":
                matched_bearish.append(phrase)
            else:
                matched_neutral.append(phrase)

            search_start = idx + 1

    # Combination bonuses — operate on the original lowercased text so that
    # partial keyword substring checks (e.g. "record" inside a phrase) work.
    triggered_combos: list[str] = []
    all_matched = matched_bullish + matched_bearish + matched_neutral

    for required_phrases, bonus, description in COMBINATION_BONUSES:
        if all(
            any(req in m for m in all_matched) or req in text_lower
            for req in required_phrases
        ):
            raw_score += bonus
            triggered_combos.append(description)
            logger.debug("Combination bonus triggered: %s (%.1f)", description, bonus)

    # Normalisation
    normalised_score: float = math.tanh(raw_score / 5.0)
    normalised_score = max(-1.0, min(1.0, normalised_score))

    if normalised_score > 0.15:
        label_out = "bullish"
    elif normalised_score < -0.15:
        label_out = "bearish"
    else:
        label_out = "neutral"

    confidence: float = min(abs(normalised_score), 1.0)

    return {
        "raw_score": raw_score,
        "normalised_score": normalised_score,
        "label": label_out,
        "confidence": confidence,
        "matched_bullish": matched_bullish,
        "matched_bearish": matched_bearish,
        "matched_neutral": matched_neutral,
        "combination_bonuses": triggered_combos,
    }


def score_corpus(texts: list[str]) -> pd.DataFrame:
    """Score a list of headlines and return results as a DataFrame.

    Args:
        texts: List of raw headline strings.

    Returns:
        A :class:`pandas.DataFrame` with one row per headline containing all
        fields from :func:`score_headline` plus a ``text`` column.
    """
    if not texts:
        logger.warning("score_corpus received an empty list; returning empty DataFrame.")
        return pd.DataFrame(
            columns=[
                "text",
                "raw_score",
                "normalised_score",
                "label",
                "confidence",
                "matched_bullish",
                "matched_bearish",
                "matched_neutral",
                "combination_bonuses",
            ]
        )

    logger.info("Scoring corpus of %d headlines.", len(texts))
    records: list[dict[str, Any]] = []
    for text in texts:
        result = score_headline(text)
        result["text"] = text
        records.append(result)

    df = pd.DataFrame(records)
    # Reorder columns for readability
    col_order = [
        "text",
        "raw_score",
        "normalised_score",
        "label",
        "confidence",
        "matched_bullish",
        "matched_bearish",
        "matched_neutral",
        "combination_bonuses",
    ]
    df = df[col_order]
    logger.info(
        "Corpus scoring complete. Bullish: %d, Neutral: %d, Bearish: %d.",
        (df["label"] == "bullish").sum(),
        (df["label"] == "neutral").sum(),
        (df["label"] == "bearish").sum(),
    )
    return df


def get_top_signals(texts: list[str], n: int = 20) -> dict[str, Any]:
    """Aggregate phrase frequencies and sentiment counts across a corpus.

    Args:
        texts: List of raw headline strings.
        n: Number of top phrases to return for each sentiment direction.

    Returns:
        A dictionary with keys:

        - ``top_bullish_phrases`` (list[tuple[str, int]]): Top *n* bullish
          phrases by frequency, as ``(phrase, count)`` pairs.
        - ``top_bearish_phrases`` (list[tuple[str, int]]): Top *n* bearish
          phrases by frequency.
        - ``bullish_count`` (int): Number of headlines labelled ``"bullish"``.
        - ``bearish_count`` (int): Number of headlines labelled ``"bearish"``.
        - ``neutral_count`` (int): Number of headlines labelled ``"neutral"``.
        - ``avg_score`` (float): Mean normalised score across the corpus.
    """
    if not texts:
        logger.warning("get_top_signals received an empty list.")
        return {
            "top_bullish_phrases": [],
            "top_bearish_phrases": [],
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "avg_score": 0.0,
        }

    df = score_corpus(texts)

    bullish_counter: Counter[str] = Counter()
    bearish_counter: Counter[str] = Counter()

    for phrases in df["matched_bullish"]:
        bullish_counter.update(phrases)
    for phrases in df["matched_bearish"]:
        bearish_counter.update(phrases)

    return {
        "top_bullish_phrases": bullish_counter.most_common(n),
        "top_bearish_phrases": bearish_counter.most_common(n),
        "bullish_count": int((df["label"] == "bullish").sum()),
        "bearish_count": int((df["label"] == "bearish").sum()),
        "neutral_count": int((df["label"] == "neutral").sum()),
        "avg_score": float(df["normalised_score"].mean()),
    }


def print_lexicon_report(results: pd.DataFrame) -> None:
    """Print a formatted ASCII sentiment report to the console.

    Displays score distribution, top bullish and bearish phrases, and the
    average normalised score. Uses only ASCII characters to ensure
    compatibility with Windows CP1252 console encoding.

    Args:
        results: DataFrame returned by :func:`score_corpus`.
    """
    if results.empty:
        print("No results to display.")
        return

    border = "+" + "-" * 58 + "+"
    header = "| {:^56} |".format("FINANCIAL LEXICON SENTIMENT REPORT")
    sep = "+" + "=" * 58 + "+"

    total = len(results)
    bullish_n = int((results["label"] == "bullish").sum())
    bearish_n = int((results["label"] == "bearish").sum())
    neutral_n = int((results["label"] == "neutral").sum())
    avg_score = float(results["normalised_score"].mean())

    # Build top-5 phrase lists
    bullish_counter: Counter[str] = Counter()
    bearish_counter: Counter[str] = Counter()
    for phrases in results["matched_bullish"]:
        bullish_counter.update(phrases)
    for phrases in results["matched_bearish"]:
        bearish_counter.update(phrases)

    top_bullish = bullish_counter.most_common(5)
    top_bearish = bearish_counter.most_common(5)

    lines: list[str] = [
        border,
        header,
        sep,
        "| {:56} |".format("Score Distribution"),
        border,
        "| {:<28} {:>27} |".format(
            "  Bullish", f"{bullish_n:>6} ({100*bullish_n/max(total,1):.1f}%)"
        ),
        "| {:<28} {:>27} |".format(
            "  Neutral", f"{neutral_n:>6} ({100*neutral_n/max(total,1):.1f}%)"
        ),
        "| {:<28} {:>27} |".format(
            "  Bearish", f"{bearish_n:>6} ({100*bearish_n/max(total,1):.1f}%)"
        ),
        "| {:<28} {:>27} |".format("  Total", f"{total:>6}"),
        border,
        "| {:56} |".format("Top 5 Bullish Phrases"),
        border,
    ]

    if top_bullish:
        for phrase, count in top_bullish:
            row = f"  {phrase}"
            cnt_str = f"x{count}"
            lines.append("| {:<46} {:>9} |".format(row[:46], cnt_str))
    else:
        lines.append("| {:56} |".format("  (none found)"))

    lines += [
        border,
        "| {:56} |".format("Top 5 Bearish Phrases"),
        border,
    ]

    if top_bearish:
        for phrase, count in top_bearish:
            row = f"  {phrase}"
            cnt_str = f"x{count}"
            lines.append("| {:<46} {:>9} |".format(row[:46], cnt_str))
    else:
        lines.append("| {:56} |".format("  (none found)"))

    lines += [
        border,
        "| {:<28} {:>27} |".format("  Avg Normalised Score", f"{avg_score:+.4f}"),
        border,
    ]

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    news_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
    df_news = pd.read_csv(news_path)
    real = df_news[
        df_news["source"].isin(
            [
                "guardian",
                "nyt",
                "yahoo_finance",
                "economic_times",
                "nikkei",
                "china_daily",
            ]
        )
    ]
    logger.info("Loaded %d real-source headlines from %s.", len(real), news_path)
    results = score_corpus(real["headline"].tolist())
    signals = get_top_signals(real["headline"].tolist())
    print_lexicon_report(results)
