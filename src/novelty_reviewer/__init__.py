"""Novelty Reviewer: a dataset and models for reviewer-aligned novelty assessment of research papers."""

from .constants import BASE_MODEL_ID, DATASET_ID, MODEL_ID, NOVELTY_LABELS
from .data import eval_subset, load_novelty_dataset, openreview_url, reviews_frame
from .model import ChatModel, NoveltyReviewer
from .openreview import download_openreview_pdf
from .parsing import NoveltyPrediction, parse_novelty_output, parse_novelty_score
from .pipeline import OriginalityAnalyzer, OriginalityReport

__version__ = "1.0.0"

__all__ = [
    "BASE_MODEL_ID", "DATASET_ID", "MODEL_ID", "NOVELTY_LABELS",
    "eval_subset", "load_novelty_dataset", "openreview_url", "reviews_frame",
    "ChatModel", "NoveltyReviewer", "OriginalityAnalyzer", "OriginalityReport", "download_openreview_pdf",
    "NoveltyPrediction", "parse_novelty_output", "parse_novelty_score",
]
