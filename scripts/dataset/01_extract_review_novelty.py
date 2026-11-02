"""Stage 1: extract novelty-related content from every review.

Two modes, both with Llama-3.1-8B-Instruct:
  excerpts  verbatim novelty passages as JSON (released as `novelty_excerpts`); greedy decoding.
  stance    a novelty stance in {-1, 0, 1} with 1-2 supporting quotes; the input of stage 2.

Input: a parquet file with one row per review and columns `paper_id` and `review_text`.
Output: the same rows with a `generated_text` column.

Example:
  python scripts/dataset/01_extract_review_novelty.py --reviews reviews.parquet --mode excerpts \
      --output review_excerpts.parquet --start 0 --end 20000
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from novelty_reviewer import BASE_MODEL_ID, ChatModel
from novelty_reviewer.prompts import (REVIEW_EXCERPTS_SYSTEM_PROMPT, REVIEW_EXCERPTS_USER_PROMPT,
                                      REVIEW_STANCE_SYSTEM_PROMPT, REVIEW_STANCE_USER_PROMPT)

MODES = {
    "excerpts": (REVIEW_EXCERPTS_SYSTEM_PROMPT, lambda r: REVIEW_EXCERPTS_USER_PROMPT.format(review_text=r),
                 {"max_new_tokens": 600, "do_sample": False, "repetition_penalty": 1.1}),
    "stance": (REVIEW_STANCE_SYSTEM_PROMPT, lambda r: REVIEW_STANCE_USER_PROMPT.format(review=r),
               {"max_new_tokens": 100, "do_sample": True, "temperature": 0.7, "top_p": 0.9}),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviews", required=True, help="parquet with columns paper_id, review_text")
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=BASE_MODEL_ID)
    ap.add_argument("--start", type=int, default=0, help="first row (for sharding across jobs)")
    ap.add_argument("--end", type=int, default=None, help="end row, exclusive")
    args = ap.parse_args()

    reviews = pd.read_parquet(args.reviews, columns=["paper_id", "review_text"]).iloc[args.start:args.end]
    system, user, generation = MODES[args.mode]
    model = ChatModel(args.model)

    outputs = []
    for review in tqdm(reviews.review_text, desc=f"{args.mode} ({len(reviews)} reviews)"):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user(review)}]
        outputs.append(model.generate(messages, **generation))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    reviews.assign(generated_text=outputs).to_parquet(args.output, index=False)
    print(f"wrote {len(reviews)} rows to {args.output}")


if __name__ == "__main__":
    main()
