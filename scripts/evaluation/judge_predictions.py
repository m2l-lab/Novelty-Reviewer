"""Free-text evaluation of novelty assessments against the reference ECNA assessments.

  --nli    entailment-contradiction score with roberta-large-mnli (in [0, 1])
  --judge  LLM-as-a-judge score with Llama-3.1-8B-Instruct (in [0, 10])

Example:
  python scripts/evaluation/judge_predictions.py results/predictions/*/model_results.parquet --nli --judge \
      --output results/free_text_metrics.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from novelty_reviewer.evaluation import LLMJudge, NLIScorer

from score_predictions import load_references


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+", help="prediction parquet files")
    ap.add_argument("--nli", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--dataset-path", default=None, help="local release folder instead of the Hugging Face Hub")
    ap.add_argument("--output", default=None, help="CSV file for the metrics table")
    args = ap.parse_args()
    if not (args.nli or args.judge):
        ap.error("choose --nli and/or --judge")

    references = load_references(args.dataset_path).set_index("paper_id")
    nli = NLIScorer() if args.nli else None
    judge = LLMJudge() if args.judge else None

    rows = []
    for path in map(Path, args.predictions):
        pred = pd.read_parquet(path, columns=["paper_id", "generated_text"]).join(references, on="paper_id")
        pred = pred.dropna(subset=["novelty_summary"])
        row = {"model": path.parent.name, "n": len(pred)}
        if nli:
            row["nli_ec"] = nli(pred.generated_text.tolist(), pred.novelty_summary.tolist())
        if judge:
            scores = [judge.score(o, s, r) for o, s, r in tqdm(
                zip(pred.generated_text, pred.novelty_summary, pred.novelty_score), total=len(pred), desc=row["model"])]
            row["llm_judge"] = sum(scores) / len(scores)
        rows.append(row)

    table = pd.DataFrame(rows).set_index("model")
    print(table.round(4).to_markdown())
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output)


if __name__ == "__main__":
    main()
