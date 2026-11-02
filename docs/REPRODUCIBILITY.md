# Reproducing the Paper

This guide maps every part of the paper to the code that produces it. All commands run from the repository root
after `pip install -e ".[all]"`. Steps marked (GPU) need a GPU.

| Paper | What | Code |
|---|---|---|
| §3.2 Extraction | Novelty excerpts and stances per review | `scripts/dataset/01_extract_review_novelty.py` (GPU) |
| §3.2 Aggregation | ECNA score and assessment per paper | `scripts/dataset/02_aggregate_paper_novelty.py` (GPU) |
| §3.1, §3.6 | Titles, abstracts, author lists (privacy check) | `scripts/dataset/03_fetch_openreview_metadata.py` |
| §3.3–3.7 | Release dataset: merge, splits, anonymization | `scripts/dataset/04_build_release_dataset.py` |
| §4.1 | Knowledge extraction | `novelty_reviewer.retrieval.knowledge` |
| §4.2 | Semantic retrieval and knowledge graph | `novelty_reviewer.retrieval`, `novelty_reviewer.pipeline` |
| §4.3 | Fine-tuning data and training | `scripts/training/` (GPU) |
| §4.4 | Originality report | `novelty_reviewer.pipeline.OriginalityAnalyzer` (GPU) |
| §5 | Score prediction metrics (main results table) | `scripts/evaluation/generate_predictions.py` (GPU), `score_predictions.py` |
| §5 | E–C NLI, LLM judge, BERTScore/AlignScore tables | `scripts/evaluation/judge_predictions.py` (GPU) |
| §5, appendix | Predicted score distributions | `scripts/evaluation/plot_score_distributions.py` |

All prompts are defined once, verbatim, in [`src/novelty_reviewer/prompts.py`](../src/novelty_reviewer/prompts.py).

## 1. Dataset construction

The release dataset can be used directly; these steps document how it was built. The raw OpenReview crawl
(papers, reviews, review forms) follows the data pipeline of OpenReviewer (Idahl & Ahmadi, NAACL 2025 System Demonstrations).

```bash
# Stage 1: per-review extraction with Llama-3.1-8B-Instruct (shard with --start/--end)
python scripts/dataset/01_extract_review_novelty.py --reviews reviews.parquet --mode excerpts --output excerpts.parquet
python scripts/dataset/01_extract_review_novelty.py --reviews reviews.parquet --mode stance --output stances.parquet

# Stage 2: paper-level ECNA labels
python scripts/dataset/02_aggregate_paper_novelty.py --stances stances.parquet --output ecna.parquet

# Metadata and release (needs an OpenReview account; the metadata contains author names, keep it private)
python scripts/dataset/03_fetch_openreview_metadata.py --openreviewer_dir <crawl> --credentials_file creds.json \
    --out_file openreview_metadata.parquet
python scripts/dataset/04_build_release_dataset.py --help
```

Decoding: extraction is greedy (repetition penalty 1.1); aggregation samples with temperature 0.3 and top-p 0.9.

## 2. Paper texts

Paper texts are not redistributed. Download and convert them from OpenReview:

```bash
python scripts/evaluation/prepare_paper_texts.py --output data/eval_paper_texts.parquet            # 500 eval papers
python scripts/evaluation/prepare_paper_texts.py --split train --all --output data/train_paper_texts.parquet
```

If OpenReview blocks anonymous downloads from your network, set `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD`.
The paper's experiments used markdown from the OpenReviewer PDF pipeline; pymupdf4llm output is similar but not
byte-identical, so re-generated predictions can differ slightly from the reported ones.

## 3. Training

```bash
python scripts/training/prepare_training_data.py --paper-texts data/train_paper_texts.parquet \
    --exclude-neurips-2025 --output data/train_tokenized.parquet
bash scripts/training/train.sh
```

Full fine-tuning of Llama-3.1-8B-Instruct: 3 epochs, AdamW, cosine schedule, learning rate 2e-5, 50 warm-up steps,
effective batch size 32 on 32 A100-80GB GPUs, DeepSpeed ZeRO-3, sequence length 32,120, bf16, Liger kernels
([config](../scripts/training/configs/llama31_8b_full_finetune.yaml)). `--exclude-neurips-2025` reproduces the
paper's training set of 25,951 papers; NeurIPS 2025 was added to the training split after the experiments.
Exact package versions are in [`requirements/full-environment.txt`](../requirements/full-environment.txt).

## 4. Evaluation

```bash
# Novelty Reviewer (training prompt) and baselines (baseline prompt of the paper)
python scripts/evaluation/generate_predictions.py --paper-texts data/eval_paper_texts.parquet \
    --model AbeerMostafa/Novelty_Reviewer --prompt novelty_reviewer
python scripts/evaluation/generate_predictions.py --paper-texts data/eval_paper_texts.parquet --prompt baseline \
    --model meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.1 Qwen/Qwen2.5-14B-Instruct-1M \
            SenthilKumarN/SciLlama-3.2-3B maxidl/Llama-OpenReviewer-8B weathon/paper_reviewer \
            openrouter:openai/gpt-oss-20b

python scripts/evaluation/score_predictions.py results/predictions/*/model_results.parquet
python scripts/evaluation/judge_predictions.py results/predictions/*/model_results.parquet --nli --judge
python scripts/evaluation/plot_score_distributions.py results/predictions/*/model_results.parquet
```

Metrics are macro-averaged over the four ECNA classes; predictions without a parsable score are excluded, and the
`coverage` column reports the fraction kept. Decoding for all models: temperature 0.1, top-p 0.5, 300 new tokens.

## 5. Full framework

```bash
export S2_API_KEY=...   # optional Semantic Scholar key
novelty-reviewer analyze --pdf paper.pdf --output reports
```

Graph parameters (`edge_threshold`, `radius`, `similarity_weight`, `max_candidates`) are arguments of
`OriginalityAnalyzer`.
