"""Command-line interface.

    novelty-reviewer predict  --pdf paper.pdf                  # novelty score + assessment
    novelty-reviewer predict  --paper-id 0wmfzWPAFu            # any OpenReview submission
    novelty-reviewer analyze  --pdf paper.pdf --output reports # full literature-grounded report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .constants import MODEL_ID


def _manuscript_text(args) -> str:
    from .pdf import pdf_to_markdown

    if args.text:
        return Path(args.text).read_text(encoding="utf-8")
    if args.paper_id:
        from .openreview import download_openreview_pdf

        args.pdf = download_openreview_pdf(args.paper_id, args.cache_dir)
    return pdf_to_markdown(args.pdf)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="novelty-reviewer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [("predict", "score a manuscript with the Novelty Reviewer model"),
                            ("analyze", "retrieve related work and write an originality report")]:
        p = sub.add_parser(name, help=help_text)
        source = p.add_mutually_exclusive_group(required=True)
        source.add_argument("--pdf", help="manuscript PDF")
        source.add_argument("--text", help="manuscript as markdown or plain text")
        source.add_argument("--paper-id", help="OpenReview submission id")
        p.add_argument("--model", default=MODEL_ID, help="model id or local path (default: %(default)s)")
        p.add_argument("--cache-dir", default=".cache/papers", help="where downloaded PDFs are stored")
    analyze = sub.choices["analyze"]
    analyze.add_argument("--top-k", type=int, default=5, help="number of related papers in the report")
    analyze.add_argument("--output", default="reports", help="output directory for the report")
    analyze.add_argument("--share-model", action="store_true",
                         help="use the Novelty Reviewer model for knowledge extraction (saves GPU memory)")
    args = parser.parse_args(argv)

    text = _manuscript_text(args)
    if args.command == "predict":
        from .model import NoveltyReviewer

        print(NoveltyReviewer.from_pretrained(args.model).predict(text))
    else:
        from .pipeline import OriginalityAnalyzer

        analyzer = OriginalityAnalyzer.from_pretrained(args.model, share_model=args.share_model)
        report = analyzer.analyze(text, top_k=args.top_k)
        print(report.to_markdown())
        print(f"\nReport saved to {report.save(args.output)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
