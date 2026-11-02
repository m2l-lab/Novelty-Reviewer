"""Download paper PDFs from OpenReview.

The dataset does not redistribute paper texts. Use this module to fetch a paper by its `paper_id`. OpenReview may
block anonymous downloads from some networks (e.g. compute clusters); with OPENREVIEW_USERNAME and
OPENREVIEW_PASSWORD set, the download is authenticated through the OpenReview API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

API = "https://api2.openreview.net"


def download_openreview_pdf(paper_id: str, dest_dir: str | Path = "papers", username: Optional[str] = None,
                            password: Optional[str] = None, timeout: int = 120) -> Path:
    """Download the PDF of `paper_id` to `dest_dir/<paper_id>.pdf` and return its path."""
    import requests

    dest = Path(dest_dir) / f"{paper_id}.pdf"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    username = username or os.environ.get("OPENREVIEW_USERNAME")
    password = password or os.environ.get("OPENREVIEW_PASSWORD")
    if username and password:
        token = requests.post(f"{API}/login", json={"id": username, "password": password}, timeout=60)
        token.raise_for_status()
        response = requests.get(f"{API}/pdf", params={"id": paper_id}, timeout=timeout,
                                headers={"Authorization": f"Bearer {token.json()['token']}"})
    else:
        response = requests.get("https://openreview.net/pdf", params={"id": paper_id}, timeout=timeout)

    if response.status_code != 200 or not response.content.startswith(b"%PDF"):
        raise RuntimeError(
            f"Could not download {paper_id} (HTTP {response.status_code}). OpenReview may block anonymous downloads "
            "from your network: set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD, or download the PDF manually from "
            f"https://openreview.net/forum?id={paper_id}."
        )
    dest.write_bytes(response.content)
    return dest
