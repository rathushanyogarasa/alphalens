"""Sentiment model definitions for AlphaLens.

Provides two sentiment classifiers:

- :class:`FinBERTClassifier`: Fine-tunable transformer model built on
  ``ProsusAI/finbert`` with a linear classification head.
- :class:`VADERBaseline`: Lightweight lexicon-based baseline using
  VADER's ``SentimentIntensityAnalyzer``.

Both expose an identical ``predict`` interface returning a list of
result dicts with keys ``text``, ``label``, ``label_name``,
``confidence``, and ``probabilities``.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label / signal mappings
# ---------------------------------------------------------------------------

LABEL2ID: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
"""Map from human-readable label name to integer class index."""

ID2LABEL: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
"""Map from integer class index to human-readable label name."""

SIGNAL_MAP: dict[str, int] = {
    "negative": -1,
    "neutral": 0,
    "positive": 1,
    "uncertain": 0,
}
"""Map from label name (including ``"uncertain"``) to a trading signal integer."""


# ---------------------------------------------------------------------------
# 1. FinBERTClassifier
# ---------------------------------------------------------------------------


class FinBERTClassifier(nn.Module):
    """Fine-tunable FinBERT-based three-class sentiment classifier.

    Wraps ``ProsusAI/finbert`` with an added linear classification head
    that maps the pooled ``[CLS]`` representation to three sentiment
    logits (negative / neutral / positive).

    Args:
        model_name: HuggingFace model identifier.  Defaults to
            ``config.MODEL_NAME`` (``"ProsusAI/finbert"``).

    Attributes:
        device (torch.device): The device the model is resident on.
        tokenizer: HuggingFace tokenizer loaded from *model_name*.
        bert: The pre-trained ``BertModel`` backbone.
        classifier (nn.Linear): Linear head mapping hidden size → 3 classes.
    """

    def __init__(self, model_name: str = config.MODEL_NAME) -> None:
        super().__init__()

        from transformers import AutoTokenizer, BertModel  # lazy import

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("FinBERTClassifier device: %s", self.device)

        logger.info("Loading tokenizer from '%s' …", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Loading BERT backbone from '%s' …", model_name)
        self.bert = BertModel.from_pretrained(model_name)

        hidden_size: int = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 3)
        logger.info(
            "Classification head: %d → 3 classes", hidden_size
        )

        self.to(self.device)
        logger.info("FinBERTClassifier ready")

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, texts: list[str]) -> dict:
        """Tokenise a batch of texts for BERT input.

        Applies padding to the longest sequence in the batch and
        truncates to ``config.MAX_LENGTH`` tokens.

        Args:
            texts: List of raw text strings to tokenise.

        Returns:
            dict: HuggingFace encoding dict with ``input_ids``,
            ``attention_mask``, and ``token_type_ids`` as tensors
            on :attr:`device`.
        """
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits for a batch of tokenised inputs.

        Args:
            input_ids: Token ID tensor of shape ``(batch, seq_len)``.
            attention_mask: Attention mask tensor of shape
                ``(batch, seq_len)``.

        Returns:
            torch.Tensor: Raw logits of shape ``(batch, 3)``.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # (batch, hidden_size)
        logits = self.classifier(pooled)  # (batch, 3)
        return logits

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, texts: list[str]) -> list[dict]:
        """Run batched inference and return structured predictions.

        Processes *texts* in batches of ``config.BATCH_SIZE``.  Each
        prediction includes the raw probabilities for all three classes.
        When the winning class probability does not exceed
        ``config.CONFIDENCE_THRESHOLD``, ``label_name`` is set to
        ``"uncertain"`` and ``label`` is set to the neutral class (1).

        Args:
            texts: List of raw financial text strings.

        Returns:
            list[dict]: One dict per input text with keys:

                - ``text`` (str): The original input string.
                - ``label`` (int): Predicted class index (0/1/2), or 1
                  if uncertain.
                - ``label_name`` (str): ``"negative"`` / ``"neutral"`` /
                  ``"positive"`` / ``"uncertain"``.
                - ``confidence`` (float): Probability of the winning class.
                - ``probabilities`` (dict): ``{label_name: prob}`` for
                  all three classes.
        """
        self.eval()
        results: list[dict] = []

        for batch_start in range(0, len(texts), config.BATCH_SIZE):
            batch = texts[batch_start : batch_start + config.BATCH_SIZE]
            encoding = self.tokenize(batch)

            with torch.no_grad():
                logits = self.forward(
                    encoding["input_ids"], encoding["attention_mask"]
                )

            probs = torch.softmax(logits, dim=-1).cpu()

            for i, text in enumerate(batch):
                prob_vec = probs[i].tolist()
                confidence = float(max(prob_vec))
                pred_idx = int(probs[i].argmax())

                if confidence >= config.CONFIDENCE_THRESHOLD:
                    label = pred_idx
                    label_name = ID2LABEL[pred_idx]
                else:
                    label = LABEL2ID["neutral"]
                    label_name = "uncertain"

                results.append(
                    {
                        "text": text,
                        "label": label,
                        "label_name": label_name,
                        "confidence": round(confidence, 4),
                        "probabilities": {
                            ID2LABEL[j]: round(float(prob_vec[j]), 4)
                            for j in range(3)
                        },
                    }
                )

        logger.info(
            "FinBERT predicted %d texts | label distribution: %s",
            len(results),
            {
                k: sum(1 for r in results if r["label_name"] == k)
                for k in list(ID2LABEL.values()) + ["uncertain"]
                if any(r["label_name"] == k for r in results)
            },
        )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the full model state (backbone + classifier head) to disk.

        The tokenizer is saved alongside the weights so the model can be
        reloaded without knowing the original ``model_name``.

        Args:
            path: Directory path where weights and tokenizer are saved.
                Created if it does not exist.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "weights.pt")
        self.tokenizer.save_pretrained(str(path))
        self.bert.config.save_pretrained(str(path))
        logger.info("FinBERTClassifier saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "FinBERTClassifier":
        """Load a previously saved :class:`FinBERTClassifier` from disk.

        Args:
            path: Directory path that was passed to :meth:`save`.

        Returns:
            FinBERTClassifier: Restored model in eval mode.

        Raises:
            FileNotFoundError: If ``weights.pt`` is not found at *path*.
        """
        path = Path(path)
        weights_file = path / "weights.pt"
        if not weights_file.exists():
            raise FileNotFoundError(
                f"No weights.pt found at {path}. "
                "Run training first or point to the correct model directory."
            )
        instance = cls(model_name=str(path))
        instance.load_state_dict(
            torch.load(weights_file, map_location=instance.device)
        )
        instance.eval()
        logger.info("FinBERTClassifier loaded ← %s", path)
        return instance


# ---------------------------------------------------------------------------
# 2. VADERBaseline
# ---------------------------------------------------------------------------


class VADERBaseline:
    """Lexicon-based sentiment baseline using VADER.

    Maps VADER compound scores to three classes:

    - compound > 0.05  → positive (2)
    - compound < -0.05 → negative (0)
    - otherwise        → neutral  (1)

    Confidence is the absolute compound score, capped at 1.0.
    No GPU or HuggingFace dependency is required.

    Attributes:
        analyzer: Initialised :class:`vaderSentiment.SentimentIntensityAnalyzer`.
    """

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("VADERBaseline initialised")

    def predict(self, texts: list[str]) -> list[dict]:
        """Score a list of texts with VADER and return structured predictions.

        Output schema is identical to :meth:`FinBERTClassifier.predict`
        for drop-in substitutability.  The ``probabilities`` field
        contains a single entry for the predicted class set to the
        confidence value; the other two classes are set to 0.0.

        Args:
            texts: List of raw text strings to score.

        Returns:
            list[dict]: One dict per input text with keys:

                - ``text`` (str): The original input string.
                - ``label`` (int): 0 (negative), 1 (neutral), or 2 (positive).
                - ``label_name`` (str): Human-readable label.
                - ``confidence`` (float): ``abs(compound)``, capped at 1.0.
                - ``probabilities`` (dict): ``{label_name: confidence}``
                  with 0.0 for the other two classes.
        """
        results: list[dict] = []

        for text in texts:
            scores = self.analyzer.polarity_scores(text)
            compound: float = scores["compound"]
            confidence: float = min(abs(compound), 1.0)

            if compound > 0.05:
                label = 2
                label_name = "positive"
            elif compound < -0.05:
                label = 0
                label_name = "negative"
            else:
                label = 1
                label_name = "neutral"

            probabilities = {name: 0.0 for name in ID2LABEL.values()}
            probabilities[label_name] = round(confidence, 4)

            results.append(
                {
                    "text": text,
                    "label": label,
                    "label_name": label_name,
                    "confidence": round(confidence, 4),
                    "probabilities": probabilities,
                }
            )

        logger.info(
            "VADER predicted %d texts | label distribution: %s",
            len(results),
            {
                k: sum(1 for r in results if r["label_name"] == k)
                for k in ID2LABEL.values()
                if any(r["label_name"] == k for r in results)
            },
        )
        return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    vader = VADERBaseline()
    results = vader.predict(
        [
            "Operating profit rose significantly",
            "The company missed revenue estimates",
            "Results were in line with expectations",
        ]
    )
    for r in results:
        print(r)
