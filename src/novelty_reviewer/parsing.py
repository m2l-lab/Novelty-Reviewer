"""Parsing of model outputs: novelty scores, review sections and JSON objects."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from .constants import NOVELTY_LABELS, NOVELTY_SCORES

# Headers may be markdown-bold ("**Novelty Score:** 1") and scores may be bracketed ("[2]").
_B = r"\**"
_SCORE = re.compile(rf"(?i){_B}Novelty Score:{_B}\s*{_B}\[?\s*(-?\d+)")
_JUSTIFICATION = re.compile(
    rf"(?is){_B}Novelty Score:[^\n]*\n\s*(?:{_B}Score Justification:{_B})?\s*(.*?)(?=\n\s*{_B}Detailed Assessment:)"
)
_DETAILED = re.compile(rf"(?is){_B}Detailed Assessment:{_B}\s*(.*)")
_REVIEW = re.compile(rf"(?is){_B}Short Novelty Review:{_B}\s*(.*)")


@dataclass
class NoveltyPrediction:
    """A novelty score with its textual assessment."""

    score: Optional[int]
    review: str
    raw_text: str

    @property
    def label(self) -> Optional[str]:
        return NOVELTY_LABELS.get(self.score)

    def __str__(self) -> str:
        score = "n/a" if self.score is None else f"{self.score} ({self.label})"
        return f"Novelty Score: {score}\n\n{self.review}"


def parse_novelty_score(text: Optional[str]) -> Optional[int]:
    """Return the first "Novelty Score: <n>" in `text`, or None if absent or outside {-1, 0, 1, 2}."""
    if not text:
        return None
    m = _SCORE.search(text)
    if not m:
        return None
    score = int(m.group(1))
    return score if score in NOVELTY_SCORES else None


def parse_novelty_output(text: str) -> NoveltyPrediction:
    """Split a generated assessment into score and review text."""
    text = text or ""
    review = _REVIEW.search(text)
    if review:
        body = review.group(1)
    else:
        # The fine-tuned model repeats the score line (its training target was "Novelty Score: X\n" + an aggregated
        # assessment that itself starts with "Novelty Score: X"); drop all score lines.
        body = re.sub(r"(?im)^[ \t]*\**[ \t]*Novelty Score:.*$", "", text)
    return NoveltyPrediction(score=parse_novelty_score(text), review=body.strip(), raw_text=text)


def parse_aggregated_assessment(text: str) -> dict:
    """Split an aggregation output (stage 2) into score, justification and detailed assessment."""
    just, det = _JUSTIFICATION.search(text or ""), _DETAILED.search(text or "")
    return {
        "novelty_score": parse_novelty_score(text),
        "novelty_score_justification": just.group(1).strip() if just else None,
        "novelty_detailed_assessment": det.group(1).strip() if det else None,
    }


def parse_json_object(text: str, default: Optional[dict] = None) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response, tolerating code fences and surrounding prose."""
    text = (text or "").strip()
    if "```" in text:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if fenced:
            text = fenced.group(1)
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.S)
    if m:
        text = m.group(0)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else dict(default or {})
    except json.JSONDecodeError:
        return dict(default or {})
