"""End-to-end originality analysis: knowledge extraction, related-work retrieval, graph ranking and novelty scoring."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .constants import BASE_MODEL_ID, MODEL_ID
from .model import ChatModel, NoveltyReviewer
from .parsing import NoveltyPrediction
from .retrieval import (KnowledgeExtractor, KnowledgeGraph, RankedPaper, SemanticScholarClient,
                        descriptor_similarity, summarize_for_context)
from .retrieval.graph import MANUSCRIPT


@dataclass
class OriginalityReport:
    """Result of `OriginalityAnalyzer.analyze`."""

    manuscript: dict
    related_work: list[RankedPaper]
    prediction: Optional[NoveltyPrediction]
    queries: list[str] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_markdown(self) -> str:
        m = self.manuscript
        lines = ["# Originality Report", "", f"_Generated {self.created}_", ""]
        if self.prediction is not None:
            lines += ["## Novelty assessment", "",
                      f"**Novelty score:** {self.prediction.score} ({self.prediction.label})", "",
                      self.prediction.review, ""]
        lines += ["## Manuscript", "",
                  f"- **Core ideas:** {', '.join(m['core_ideas'][:5])}",
                  f"- **Methods:** {', '.join(m['methods'][:4])}",
                  f"- **Keywords:** {', '.join(m['keywords'][:7])}", "",
                  "## Most similar prior work", ""]
        for rank, paper in enumerate(self.related_work, 1):
            k, meta = paper.knowledge, paper.metadata
            lines += [f"### {rank}. {k['title']}", "",
                      f"- **Similarity:** {paper.similarity:.1f} (rank score {paper.score:.1f})",
                      f"- **Year:** {meta.get('year', 'n/a')} | **Citations:** {meta.get('citationCount', 0)}",
                      f"- **URL:** {meta.get('url', 'n/a')}",
                      f"- **Core ideas:** {', '.join(k.get('core_ideas', [])[:4])}",
                      f"- **Methods:** {', '.join(k.get('methods', [])[:3])}"]
            if paper.explanation:
                lines.append(f"- **Overlap:** {paper.explanation}")
            lines.append("")
        return "\n".join(lines)

    def save(self, output_dir: str | Path) -> Path:
        """Write the report as markdown and JSON; returns the markdown path."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"originality_report_{self.created.replace(':', '').replace('-', '')}"
        (out / f"{stem}.json").write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        path = out / f"{stem}.md"
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path


class OriginalityAnalyzer:
    """Literature-grounded novelty assessment of a manuscript.

    1. Extract the manuscript's core ideas, methods, contributions, data sources and experiments.
    2. Query Semantic Scholar with each aspect and extract the same structure from the retrieved papers.
    3. Link papers whose descriptors are similar, and rank the manuscript's neighbours by similarity and centrality.
    4. Ask the Novelty Reviewer for a score and assessment, conditioned on the top-ranked related work.

    Args:
        reviewer: the novelty model; None skips step 4.
        extractor: chat model for knowledge extraction (default: Llama-3.1-8B-Instruct). Passing
            `reviewer.chat_model` avoids loading a second 8B model, at some cost in extraction quality.
        s2_api_key: Semantic Scholar API key (default: S2_API_KEY environment variable).
        encoder: sentence-transformers model for descriptor similarity.
        edge_threshold, radius, similarity_weight: graph construction and ranking parameters.
        max_candidates: number of retrieved papers (most cited first) that are analysed.
        explain_overlap: ask the extractor for a short overlap explanation per related paper.
    """

    def __init__(self, reviewer: Optional[NoveltyReviewer] = None, extractor: Optional[ChatModel] = None,
                 s2_api_key: Optional[str] = None, encoder: str = "all-MiniLM-L6-v2",
                 edge_threshold: float = 25.0, radius: float = 35.0, similarity_weight: float = 0.7,
                 max_candidates: int = 20, explain_overlap: bool = True):
        from sentence_transformers import SentenceTransformer

        self.reviewer = reviewer
        self.extractor = KnowledgeExtractor(extractor or ChatModel(BASE_MODEL_ID))
        self.s2 = SemanticScholarClient(api_key=s2_api_key)
        self.encoder = SentenceTransformer(encoder)
        self.edge_threshold, self.radius, self.similarity_weight = edge_threshold, radius, similarity_weight
        self.max_candidates, self.explain_overlap = max_candidates, explain_overlap

    @classmethod
    def from_pretrained(cls, model_id: str = MODEL_ID, share_model: bool = False, **kwargs) -> "OriginalityAnalyzer":
        reviewer = NoveltyReviewer.from_pretrained(model_id)
        extractor = reviewer.chat_model if share_model else None
        return cls(reviewer=reviewer, extractor=extractor, **kwargs)

    @staticmethod
    def build_queries(k: dict) -> list[str]:
        """One search query per aspect of the manuscript."""
        first = lambda f: k[f][0] if k.get(f) else ""
        queries = [
            " ".join(k["core_ideas"][:3]),
            " ".join(k["methods"][:3]),
            f"{first('core_ideas')} {first('methods')}" if k["core_ideas"] and k["methods"] else "",
            " ".join(k["contributions"][:2]),
            f"{first('core_ideas')} {first('data_sources')}" if k["data_sources"] else "",
            " ".join(k["experiments"][:2]),
            " ".join(k["keywords"][:5]),
        ]
        return list(dict.fromkeys(q.strip() for q in queries if q.strip()))

    def retrieve(self, queries: list[str]) -> list[dict]:
        """Search each query; keep the `max_candidates` most cited unique papers."""
        found = {p["paperId"]: p for q in queries for p in self.s2.search(q, limit=5) if p.get("paperId")}
        return sorted(found.values(), key=lambda p: p.get("citationCount") or 0, reverse=True)[:self.max_candidates]

    def analyze(self, manuscript_text: str, top_k: int = 5, verbose: bool = True) -> OriginalityReport:
        log = print if verbose else (lambda *a, **k: None)

        log("[1/4] Extracting manuscript knowledge")
        manuscript = self.extractor.extract(manuscript_text, "New Manuscript")
        queries = self.build_queries(manuscript)

        log(f"[2/4] Retrieving related work ({len(queries)} queries)")
        candidates = self.retrieve(queries)

        log(f"[3/4] Analysing {len(candidates)} candidates")
        graph = KnowledgeGraph(edge_threshold=self.edge_threshold)
        graph.add_paper(MANUSCRIPT, manuscript)
        papers = {}
        for paper in candidates:
            text = self.s2.full_text(paper) or paper.get("abstract") or paper.get("title", "")
            if len(text) < 200:
                continue
            knowledge = self.extractor.extract(text, paper["title"])
            similarity = descriptor_similarity(self.encoder, manuscript, knowledge)
            graph.add_paper(paper["paperId"], knowledge)
            graph.add_edge(MANUSCRIPT, paper["paperId"], similarity)
            papers[paper["paperId"]] = (knowledge, paper)

        ranked = []
        for paper_id, similarity, score in graph.rank(self.radius, self.similarity_weight)[:top_k]:
            knowledge, meta = papers[paper_id]
            explanation = self.extractor.explain_overlap(manuscript, knowledge)[1] if self.explain_overlap else ""
            ranked.append(RankedPaper(knowledge, similarity, score, explanation,
                                      {k: meta.get(k) for k in ("paperId", "year", "citationCount", "url")}))
        self.graph = graph

        prediction = None
        if self.reviewer is not None:
            log("[4/4] Assessing novelty")
            prediction = self.reviewer.predict(manuscript_text, [summarize_for_context(p.knowledge) for p in ranked])
        return OriginalityReport(manuscript, ranked, prediction, queries)
