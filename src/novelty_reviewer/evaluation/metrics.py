"""Metrics used in the paper: score classification metrics and the entailment-contradiction (E-C) NLI score."""

from __future__ import annotations

from typing import Optional, Sequence

from ..parsing import parse_novelty_score


def classification_metrics(y_true: Sequence[Optional[int]], y_pred: Sequence[Optional[int]]) -> dict:
    """Accuracy, macro precision/recall/F1 and rank correlations over the four novelty classes.

    Pairs with a missing true or predicted score are dropped, as in the paper; `coverage` reports the fraction kept.
    """
    from scipy.stats import kendalltau, pearsonr, spearmanr
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if not pairs:
        return {"n": 0, "coverage": 0.0}
    t, p = map(list, zip(*pairs))
    correlations = {}
    if len(set(t)) > 1 and len(set(p)) > 1:
        correlations = {"pearson": pearsonr(t, p)[0], "spearman": spearmanr(t, p)[0], "kendall": kendalltau(t, p)[0]}
    return {
        "n": len(pairs),
        "coverage": len(pairs) / len(y_true),
        "accuracy": accuracy_score(t, p),
        "precision": precision_score(t, p, average="macro", zero_division=0),
        "recall": recall_score(t, p, average="macro", zero_division=0),
        "f1": f1_score(t, p, average="macro", zero_division=0),
        **correlations,
    }


def score_predictions(references: Sequence[Optional[int]], generated_texts: Sequence[str]) -> dict:
    """`classification_metrics` for raw model outputs, parsing "Novelty Score: <n>" from each text."""
    return classification_metrics(references, [parse_novelty_score(t) for t in generated_texts])


class NLIScorer:
    """Entailment-contradiction score in [0, 1]: (mean(P(entail) - P(contradict)) + 1) / 2.

    The model output is the premise and the reference assessment the hypothesis.
    """

    def __init__(self, model_id: str = "roberta-large-mnli", device: Optional[str] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail, self.contradict = labels["entailment"], labels["contradiction"]

    def __call__(self, outputs: Sequence[str], references: Sequence[str], batch_size: int = 32) -> float:
        import torch

        outputs = [(x or "").strip() for x in outputs]
        references = [(x or "").strip() for x in references]
        diffs = []
        for i in range(0, len(outputs), batch_size):
            batch = self.tokenizer(outputs[i:i + batch_size], references[i:i + batch_size], padding=True,
                                   truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**batch).logits, dim=-1)
            diffs += (probs[:, self.entail] - probs[:, self.contradict]).tolist()
        return (sum(diffs) / len(diffs) + 1) / 2
