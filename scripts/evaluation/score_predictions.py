"""Score novelty predictions against the ECNA labels: accuracy, macro precision/recall/F1 and rank correlations.

Each prediction file is a parquet with columns paper_id and generated_text (as written by generate_predictions.py).
The model name is taken from the file's parent directory.

Example:
  python scripts/evaluation/score_predictions.py results/predictions/*/model_results.parquet \
      --output results/score_metrics.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from novelty_reviewer import load_novelty_dataset
from novelty_reviewer.evaluation import score_predictions


def load_references(dataset_path=None) -> pd.DataFrame:
    ds = load_novelty_dataset("test", path=dataset_path)
    return ds.select_columns(["paper_id", "novelty_score", "novelty_summary"]).to_pandas()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+", help="prediction parquet files")
    ap.add_argument("--dataset-path", default=None, help="local release folder instead of the Hugging Face Hub")
    ap.add_argument("--output", default=None, help="CSV file for the metrics table")
    args = ap.parse_args()

    references = load_references(args.dataset_path).set_index("paper_id").novelty_score
    rows = []
    for path in map(Path, args.predictions):
        pred = pd.read_parquet(path, columns=["paper_id", "generated_text"])
        truth = [None if pd.isna(s) else int(s) for s in references.reindex(pred.paper_id)]
        rows.append({"model": path.parent.name, **score_predictions(truth, pred.generated_text.tolist())})

    table = pd.DataFrame(rows).set_index("model").sort_values("f1", ascending=False)
    print(table.round(4).to_markdown())
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output)


if __name__ == "__main__":
    main()
