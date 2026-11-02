"""Minimal Semantic Scholar Graph API client."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from ..pdf import pdf_bytes_to_text

API = "https://api.semanticscholar.org/graph/v1"
SEARCH_FIELDS = "paperId,title,year,authors,abstract,citationCount,url"


class SemanticScholarClient:
    """Paper search and open-access full-text retrieval.

    An API key (https://www.semanticscholar.org/product/api) raises the rate limit; it is read from the
    S2_API_KEY environment variable if not given.
    """

    def __init__(self, api_key: Optional[str] = None, cache_dir: str | Path = ".cache/papers",
                 request_interval: float = 1.0):
        import requests

        self.session = requests.Session()
        api_key = api_key or os.environ.get("S2_API_KEY")
        if api_key:
            self.session.headers["x-api-key"] = api_key
        self.cache_dir = Path(cache_dir)
        self.request_interval = request_interval

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Relevance search; returns paper records with the fields in SEARCH_FIELDS."""
        response = self.session.get(f"{API}/paper/search", timeout=30,
                                    params={"query": query[:500], "limit": limit, "fields": SEARCH_FIELDS})
        time.sleep(self.request_interval)
        if response.status_code != 200:
            return []
        return response.json().get("data", [])

    def full_text(self, paper: dict) -> Optional[str]:
        """Plain text of the paper's open-access PDF (cached on disk), or None if unavailable."""
        cache = self.cache_dir / f"{paper['paperId']}.txt"
        if cache.exists():
            return cache.read_text(encoding="utf-8")

        urls = [paper["url"]] if str(paper.get("url", "")).endswith(".pdf") else []
        try:
            info = self.session.get(f"{API}/paper/{paper['paperId']}", params={"fields": "openAccessPdf"},
                                    timeout=10).json()
            if (info.get("openAccessPdf") or {}).get("url"):
                urls.append(info["openAccessPdf"]["url"])
        except Exception:
            pass

        for url in urls:
            try:
                response = self.session.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            except Exception:
                continue
            text = pdf_bytes_to_text(response.content) if response.status_code == 200 else None
            if text:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(text, encoding="utf-8")
                return text
        return None
