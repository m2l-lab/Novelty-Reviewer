"""Build the tokenized fine-tuning data of the Novelty Reviewer model.

Each example is one user turn (the novelty-review prompt with the paper text) and one assistant turn
("Novelty Score: <ECNA score>" followed by the ECNA assessment). The loss is computed on the assistant turn only.
The output parquet (input_ids, attention_mask, labels) is read by the axolotl configs in scripts/training/configs/.

Paper texts are not part of the dataset; create them with scripts/evaluation/prepare_paper_texts.py --all.

Example:
  python scripts/evaluation/prepare_paper_texts.py --split train --all --output data/train_paper_texts.parquet
  python scripts/training/prepare_training_data.py --paper-texts data/train_paper_texts.parquet \
      --output data/train_tokenized.parquet
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from novelty_reviewer import BASE_MODEL_ID, load_novelty_dataset
from novelty_reviewer.prompts import NOVELTY_REVIEW_PROMPT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper-texts", required=True, help="parquet with columns paper_id, paper_text")
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--exclude-neurips-2025", action="store_true",
                    help="reproduce the training set of the paper (NeurIPS 2025 was added after the experiments)")
    ap.add_argument("--tokenizer", default=BASE_MODEL_ID)
    ap.add_argument("--dataset-path", default=None, help="local release folder instead of the Hugging Face Hub")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    ds = load_novelty_dataset(args.split, path=args.dataset_path)
    papers = ds.select_columns(["paper_id", "venue", "year", "novelty_score", "novelty_summary"]).to_pandas()
    if args.exclude_neurips_2025:
        papers = papers[~((papers.venue == "NeurIPS") & (papers.year == 2025))]
    papers = papers.dropna(subset=["novelty_score"]).merge(pd.read_parquet(args.paper_texts), on="paper_id")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = []
    for p in tqdm(papers.itertuples(), total=len(papers), desc="tokenizing"):
        messages = [{"role": "user", "content": NOVELTY_REVIEW_PROMPT.format(paper_text=p.paper_text)},
                    {"role": "assistant", "content": f"Novelty Score: {int(p.novelty_score)}\n {p.novelty_summary}"}]
        prompt_ids = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)
        input_ids = tokenizer.apply_chat_template(messages)
        rows.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
                     "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(f"wrote {len(rows)} training examples to {args.output}")


if __name__ == "__main__":
    main()
