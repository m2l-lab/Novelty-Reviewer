"""PDF to text conversion.

The model was trained on papers converted to markdown; `pdf_to_markdown` gives the closest input format.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional, Union


def pdf_to_markdown(pdf_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> str:
    """Convert a PDF to markdown with pymupdf4llm; optionally write it to `output_path`."""
    import pymupdf4llm

    markdown = pymupdf4llm.to_markdown(str(pdf_path))
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(markdown, encoding="utf-8")
    return markdown


def pdf_bytes_to_text(pdf_bytes: bytes, max_pages: int = 50) -> Optional[str]:
    """Plain-text extraction used for retrieved papers; returns None for unreadable or near-empty PDFs."""
    from PyPDF2 import PdfReader

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = " ".join(page.extract_text() or "" for page in reader.pages[:max_pages])
    except Exception:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) > 500 else None


def batch_pdf_to_markdown(pdf_dir: Union[str, Path], output_dir: Union[str, Path]) -> list[Path]:
    """Convert every PDF in `pdf_dir` to `output_dir/<name>.md`."""
    written = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        out = Path(output_dir) / f"{pdf.stem}.md"
        pdf_to_markdown(pdf, out)
        written.append(out)
    return written
