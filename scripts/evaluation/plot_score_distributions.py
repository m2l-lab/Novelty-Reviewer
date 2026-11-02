"""Plot the share of each predicted novelty score per model next to the ECNA labels (Figure 2 of the paper).

Example:
  python scripts/evaluation/plot_score_distributions.py results/predictions/*/model_results.parquet \
      --output results/score_distributions.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from novelty_reviewer import NOVELTY_LABELS, parse_novelty_score

from score_predictions import load_references

COLORS = {-1: "#750C10", 0: "#E57373", 1: "#FFB74D", 2: "#81C784"}


def shares(scores) -> dict:
    scores = [s for s in scores if s is not None and not pd.isna(s)]
    return {k: 100 * sum(s == k for s in scores) / max(len(scores), 1) for k in NOVELTY_LABELS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--output", default="score_distributions.pdf")
    args = ap.parse_args()

    rows, paper_ids = {}, set()
    for path in map(Path, args.predictions):
        pred = pd.read_parquet(path, columns=["paper_id", "generated_text"])
        rows[path.parent.name] = shares(pred.generated_text.map(parse_novelty_score))
        paper_ids |= set(pred.paper_id)
    refs = load_references(args.dataset_path)
    rows["ECNA (reference)"] = shares(refs[refs.paper_id.isin(paper_ids)].novelty_score)

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(rows) + 2))
    y, left = np.arange(len(rows)), np.zeros(len(rows))
    for score, label in NOVELTY_LABELS.items():
        values = np.array([r[score] for r in rows.values()])
        ax.barh(y, values, left=left, color=COLORS[score], edgecolor="white", label=f"{label} ({score})")
        left += values
    ax.set_yticks(y, list(rows))
    ax.set_xlabel("Share of papers (%)")
    ax.set_xlim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
