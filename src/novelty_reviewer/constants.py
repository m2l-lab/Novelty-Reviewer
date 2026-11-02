"""Identifiers and label definitions shared across the package."""

# Hugging Face repositories.
MODEL_ID = "AbeerMostafa/Novelty_Reviewer"
DATASET_ID = "AbeerMostafa/Novelty-Reviewer-Dataset"

# Base model of the released fine-tuned model; also used for knowledge extraction and as LLM judge.
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Expert-Consensus Novelty Alignment (ECNA) labels.
NOVELTY_LABELS = {
    -1: "Not novel",
    0: "Limited novelty",
    1: "Moderately novel",
    2: "Highly novel",
}
NOVELTY_SCORES = tuple(NOVELTY_LABELS)
