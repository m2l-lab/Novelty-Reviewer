"""Stage 2: aggregate the reviews of each paper into the ECNA novelty score and assessment.

The model reads all reviews of a paper together with the per-review novelty stances from stage 1 (`--mode stance`)
and writes "Novelty Score / Score Justification / Detailed Assessment" (released as `novelty_summary`).
Decoding: temperature 0.3, top-p 0.9, as used for the released labels.

Input: the output of stage 1 in stance mode (columns paper_id, review_text, generated_text).
Output: one row per paper with columns paper_id, generated_text, novelty_score.

Example:
  python scripts/dataset/02_aggregate_paper_novelty.py --stances review_stances.parquet \
      --output paper_novelty.parquet
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from novelty_reviewer import BASE_MODEL_ID, ChatModel, parse_novelty_score
from novelty_reviewer.prompts import AGGREGATION_SYSTEM_PROMPT, AGGREGATION_USER_PROMPT

GENERATION = {"max_new_tokens": 350, "do_sample": True, "temperature": 0.3, "top_p": 0.9}


def build_messages(reviews: list[str], stances: list[str]) -> list[dict]:
    reviews_text = "\n\n".join(f"Review {i + 1}: {r}" for i, r in enumerate(reviews))
    excerpts_text = "\n\n".join(f"Novelty Excerpt {i + 1}: {s}" for i, s in enumerate(stances))
    return [{"role": "system", "content": AGGREGATION_SYSTEM_PROMPT},
            {"role": "user", "content": AGGREGATION_USER_PROMPT.format(reviews=reviews_text,
                                                                       novelty_excerpts=excerpts_text)}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stances", required=True, help="stage-1 output in stance mode")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=BASE_MODEL_ID)
    ap.add_argument("--start", type=int, default=0, help="first paper (for sharding across jobs)")
    ap.add_argument("--end", type=int, default=None, help="end paper, exclusive")
    args = ap.parse_args()

    stances = pd.read_parquet(args.stances)
    papers = (stances.groupby("paper_id", sort=True)
              .agg(reviews=("review_text", list), stances=("generated_text", list))
              .reset_index().iloc[args.start:args.end])
    model = ChatModel(args.model)

    outputs = [model.generate(build_messages(p.reviews, p.stances), **GENERATION)
               for p in tqdm(papers.itertuples(), total=len(papers), desc="aggregating")]

    result = pd.DataFrame({"paper_id": papers.paper_id, "generated_text": outputs})
    result["novelty_score"] = result.generated_text.map(parse_novelty_score)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"wrote {len(result)} papers to {args.output} ({result.novelty_score.isna().sum()} without valid score)")


if __name__ == "__main__":
    main()
