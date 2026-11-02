import string

import pytest

from novelty_reviewer import prompts
from novelty_reviewer.model import NoveltyReviewer
from novelty_reviewer.pipeline import OriginalityAnalyzer
from novelty_reviewer.retrieval.graph import MANUSCRIPT, KnowledgeGraph
from novelty_reviewer.retrieval.knowledge import _describe, summarize_for_context

EXPECTED_FIELDS = {
    "NOVELTY_REVIEW_PROMPT": {"paper_text"},
    "NOVELTY_REVIEW_WITH_CONTEXT_PROMPT": {"paper_text", "similar_papers"},
    "BASELINE_PROMPT": {"paper_text"},
    "REVIEW_EXCERPTS_USER_PROMPT": {"review_text"},
    "REVIEW_STANCE_USER_PROMPT": {"review"},
    "AGGREGATION_USER_PROMPT": {"reviews", "novelty_excerpts"},
    "KNOWLEDGE_EXTRACTION_PROMPT": {"title", "text"},
    "OVERLAP_EXPLANATION_PROMPT": {"paper1", "paper2"},
    "JUDGE_PROMPT": {"novelty_summary", "novelty_score", "model_output"},
}

KNOWLEDGE = {"title": "T", "core_ideas": ["i1", "i2"], "methods": ["m"], "contributions": ["c"],
             "keywords": ["k"], "data_sources": ["d"], "experiments": ["e"]}


@pytest.mark.parametrize("name, fields", EXPECTED_FIELDS.items())
def test_prompt_placeholders(name, fields):
    template = getattr(prompts, name)
    assert {f for _, f, _, _ in string.Formatter().parse(template) if f} == fields


def test_messages_without_and_with_related_work():
    plain = NoveltyReviewer.build_messages("PAPER")
    assert plain == [{"role": "user", "content": prompts.NOVELTY_REVIEW_PROMPT.format(paper_text="PAPER")}]
    grounded = NoveltyReviewer.build_messages("PAPER", ["- Prior work A"])[0]["content"]
    assert "PAPER" in grounded and "- Prior work A" in grounded


def test_knowledge_formatting():
    assert summarize_for_context(KNOWLEDGE).startswith("- T:\n  Core Ideas: i1, i2")
    assert _describe(KNOWLEDGE, 2).splitlines()[0] == "Paper 2: T"


def test_queries_cover_each_aspect():
    queries = OriginalityAnalyzer.build_queries(KNOWLEDGE)
    assert queries[0] == "i1 i2" and "i1 m" in queries and "i1 d" in queries
    assert len(queries) == len(set(queries))


def test_graph_ranking():
    g = KnowledgeGraph(edge_threshold=25)
    g.add_paper(MANUSCRIPT, {"title": "m"})
    for pid, sim in [("close", 80), ("mid", 40), ("weak", 30), ("unlinked", 10)]:
        g.add_paper(pid, {"title": pid})
        g.add_edge(MANUSCRIPT, pid, sim)
    ranked = g.rank(radius=35)
    assert [pid for pid, *_ in ranked] == ["close", "mid"]           # weak < radius, unlinked < threshold
    assert ranked[0][2] == pytest.approx(0.7 * 80 + 0.3 * 80 / 100)
