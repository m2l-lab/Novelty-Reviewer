"""Download papers from OpenReview and convert them to markdown, the input format of the models.

The dataset does not redistribute paper texts; this script recreates them for a split (by default the 500-paper
evaluation subset). Downloads are resumable. Set OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD if OpenReview blocks
anonymous downloads from your network.

Note: the paper's experiments used texts converted with the OpenReviewer PDF pipeline; pymupdf4llm gives similar
but not byte-identical markdown.

Example:
  python scripts/evaluation/prepare_paper_texts.py --output data/eval_paper_texts.parquet
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from novelty_reviewer import download_openreview_pdf, eval_subset, load_novelty_dataset
from novelty_reviewer.pdf import pdf_to_markdown


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--all", action="store_true", help="whole split instead of the evaluation subset")
    ap.add_argument("--dataset-path", default=None, help="local release folder instead of the Hugging Face Hub")
    ap.add_argument("--pdf-dir", default=".cache/papers")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ds = load_novelty_dataset(args.split, path=args.dataset_path)
    paper_ids = ds["paper_id"] if args.all else eval_subset(ds)["paper_id"]

    done = {}
    if Path(args.output).exists():
        done = dict(pd.read_parquet(args.output).itertuples(index=False, name=None))
    failed = []
    for paper_id in tqdm([p for p in paper_ids if p not in done], desc="papers"):
        try:
            done[paper_id] = pdf_to_markdown(download_openreview_pdf(paper_id, args.pdf_dir))
        except Exception as e:  # unavailable or unreadable PDF
            failed.append((paper_id, str(e)[:200]))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(done.items()), columns=["paper_id", "paper_text"]).to_parquet(args.output, index=False)
    print(f"wrote {len(done)} paper texts to {args.output}; {len(failed)} failed")
    for paper_id, error in failed[:10]:
        print(f"  {paper_id}: {error}")


if __name__ == "__main__":
    main()
