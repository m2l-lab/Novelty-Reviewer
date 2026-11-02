"""Similarity graph over the manuscript and retrieved papers, and the ranking of related work."""

from __future__ import annotations

from dataclasses import dataclass, field

from .knowledge import COMPARED_FIELDS

MANUSCRIPT = "manuscript"


def descriptors(knowledge: dict) -> list[str]:
    return [d for f in COMPARED_FIELDS for d in knowledge.get(f, [])]


def descriptor_similarity(encoder, a: dict, b: dict) -> float:
    """Mean pairwise cosine similarity (x100) between the descriptors of two knowledge tuples."""
    from sentence_transformers import util

    da, db = descriptors(a), descriptors(b)
    if not da or not db:
        return 0.0
    return util.cos_sim(encoder.encode(da), encoder.encode(db)).mean().item() * 100


@dataclass
class RankedPaper:
    knowledge: dict
    similarity: float
    score: float
    explanation: str = ""
    metadata: dict = field(default_factory=dict)


class KnowledgeGraph:
    """Undirected graph whose edges connect papers with descriptor similarity >= `edge_threshold`."""

    def __init__(self, edge_threshold: float = 25.0):
        import networkx as nx

        self.graph = nx.Graph()
        self.edge_threshold = edge_threshold

    def add_paper(self, paper_id: str, knowledge: dict) -> None:
        self.graph.add_node(paper_id, title=knowledge.get("title", ""), knowledge=knowledge)

    def add_edge(self, a: str, b: str, similarity: float) -> None:
        if similarity >= self.edge_threshold:
            self.graph.add_edge(a, b, weight=similarity)

    def centrality(self) -> dict[str, float]:
        """Weighted degree of every node."""
        return {n: sum(d["weight"] for *_, d in self.graph.edges(n, data=True)) for n in self.graph.nodes}

    def rank(self, radius: float = 35.0, similarity_weight: float = 0.7) -> list[tuple[str, float, float]]:
        """Neighbours of the manuscript within `radius`, ranked by a mix of similarity and centrality.

        Returns (paper_id, similarity, score) tuples, best first.
        """
        if MANUSCRIPT not in self.graph:
            return []
        centrality = self.centrality()
        ranked = []
        for neighbour in self.graph.neighbors(MANUSCRIPT):
            similarity = self.graph[MANUSCRIPT][neighbour]["weight"]
            if similarity >= radius:
                score = similarity_weight * similarity + (1 - similarity_weight) * centrality[neighbour] / 100
                ranked.append((neighbour, similarity, score))
        return sorted(ranked, key=lambda x: x[2], reverse=True)

    def plot(self, path: str) -> None:
        import matplotlib.pyplot as plt
        import networkx as nx

        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=0)
        colors = ["tab:red" if n == MANUSCRIPT else "lightblue" for n in self.graph.nodes]
        nx.draw_networkx_nodes(self.graph, pos, node_color=colors, node_size=500, alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5,
                               width=[self.graph[u][v]["weight"] / 10 for u, v in self.graph.edges])
        nx.draw_networkx_labels(self.graph, pos, {n: d["title"][:30] for n, d in self.graph.nodes(data=True)},
                                font_size=8)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
