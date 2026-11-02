"""Structured knowledge extraction: the tuple of core ideas, methods, contributions, data sources and experiments."""

from __future__ import annotations

from ..model import ChatModel
from ..parsing import parse_json_object
from ..prompts import KNOWLEDGE_EXTRACTION_PROMPT, OVERLAP_EXPLANATION_PROMPT

KNOWLEDGE_FIELDS = ("core_ideas", "methods", "contributions", "keywords", "data_sources", "experiments")
# Fields compared between papers (keywords are only used for search queries and reports).
COMPARED_FIELDS = ("core_ideas", "methods", "contributions", "data_sources", "experiments")


class KnowledgeExtractor:
    """Uses an instruction-tuned LLM to turn paper text into the structured knowledge tuple."""

    def __init__(self, chat_model: ChatModel):
        self.chat_model = chat_model

    def extract(self, text: str, title: str = "") -> dict:
        prompt = KNOWLEDGE_EXTRACTION_PROMPT.format(title=title, text=text)
        response = self.chat_model.generate([{"role": "user", "content": prompt}],
                                            max_new_tokens=1000, do_sample=True, temperature=0.3)
        parsed = parse_json_object(response)
        knowledge = {field: [str(x) for x in parsed.get(field) or []] for field in KNOWLEDGE_FIELDS}
        knowledge["title"] = title
        return knowledge

    def explain_overlap(self, manuscript: dict, paper: dict) -> tuple[float, str]:
        """LLM estimate of the conceptual overlap between two knowledge tuples, with a short explanation."""
        prompt = OVERLAP_EXPLANATION_PROMPT.format(paper1=_describe(manuscript, 1), paper2=_describe(paper, 2))
        response = self.chat_model.generate([{"role": "user", "content": prompt}],
                                            max_new_tokens=200, do_sample=True, temperature=0.3)
        parsed = parse_json_object(response)
        try:
            overlap = float(parsed.get("overlap_percentage", 0))
        except (TypeError, ValueError):
            overlap = 0.0
        return overlap, str(parsed.get("similarity_aspects") or parsed.get("similarity_reason") or "")


def _describe(knowledge: dict, index: int) -> str:
    labels = {"core_ideas": "Ideas", "methods": "Methods", "contributions": "Contributions",
              "keywords": "Keywords", "data_sources": "Data Sources", "experiments": "Experiments"}
    lines = [f"Paper {index}: {knowledge.get('title', '')}"]
    lines += [f"{labels[f]}: {'; '.join(knowledge.get(f, []))}" for f in KNOWLEDGE_FIELDS]
    return "\n            ".join(lines)


def summarize_for_context(knowledge: dict) -> str:
    """Compact description of a retrieved paper, as passed to the Novelty Reviewer."""
    join = lambda field, n: ", ".join(knowledge.get(field, [])[:n])
    return (f"- {knowledge.get('title', 'Unknown')}:\n"
            f"  Core Ideas: {join('core_ideas', 3)}\n"
            f"  Methods: {join('methods', 3)}\n"
            f"  Contributions: {join('contributions', 2)}\n"
            f"  Keywords: {join('keywords', 3)}")
