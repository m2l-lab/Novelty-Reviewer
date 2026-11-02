"""Generate novelty assessments with a model, for the papers in a paper-text file.

Models are Hugging Face ids or local paths, or "openrouter:<model>" for API models (needs OPENROUTER_API_KEY).
The prompt is the Novelty Reviewer training prompt (`--prompt novelty_reviewer`, for the fine-tuned model) or the
baseline prompt of the paper (`--prompt baseline`). Decoding: temperature 0.1, top-p 0.5, 300 new tokens.

Output: <output-dir>/<model>/model_results.parquet with columns paper_id, generated_text, predicted_score.

Examples:
  python scripts/evaluation/generate_predictions.py --paper-texts data/eval_paper_texts.parquet \
      --model AbeerMostafa/Novelty_Reviewer --prompt novelty_reviewer
  python scripts/evaluation/generate_predictions.py --paper-texts data/eval_paper_texts.parquet \
      --model meta-llama/Llama-3.1-8B-Instruct openrouter:openai/gpt-oss-20b --prompt baseline
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from novelty_reviewer import ChatModel, parse_novelty_score
from novelty_reviewer.model import DEFAULT_GENERATION
from novelty_reviewer.prompts import BASELINE_PROMPT, NOVELTY_REVIEW_PROMPT

PROMPTS = {"novelty_reviewer": NOVELTY_REVIEW_PROMPT, "baseline": BASELINE_PROMPT}


def openrouter_generate(model: str):
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    def generate(messages):
        completion = client.chat.completions.create(
            model=model, messages=messages, max_tokens=DEFAULT_GENERATION["max_new_tokens"],
            temperature=DEFAULT_GENERATION["temperature"], top_p=DEFAULT_GENERATION["top_p"])
        return str(completion.choices[0].message.content)

    return generate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper-texts", required=True, help="parquet with columns paper_id, paper_text")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--prompt", choices=PROMPTS, default="baseline")
    ap.add_argument("--output-dir", default="results/predictions")
    ap.add_argument("--limit", type=int, default=None, help="only the first N papers")
    args = ap.parse_args()

    papers = pd.read_parquet(args.paper_texts).head(args.limit)
    template = PROMPTS[args.prompt]

    for model_name in args.model:
        if model_name.startswith("openrouter:"):
            generate = openrouter_generate(model_name.removeprefix("openrouter:"))
        else:
            chat_model = ChatModel(model_name, trust_remote_code=True)
            generate = lambda messages: chat_model.generate(messages, **DEFAULT_GENERATION)

        outputs = [generate([{"role": "user", "content": template.format(paper_text=text)}])
                   for text in tqdm(papers.paper_text, desc=model_name)]

        out_dir = Path(args.output_dir) / model_name.replace("/", "__").replace(":", "__")
        out_dir.mkdir(parents=True, exist_ok=True)
        result = pd.DataFrame({"paper_id": papers.paper_id, "generated_text": outputs})
        result["predicted_score"] = result.generated_text.map(parse_novelty_score)
        result.to_parquet(out_dir / "model_results.parquet", index=False)
        print(f"{model_name}: {result.predicted_score.notna().sum()}/{len(result)} parsable scores -> {out_dir}")
        del generate


if __name__ == "__main__":
    main()
