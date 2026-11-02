"""Loading and reshaping the Novelty Reviewer dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .constants import DATASET_ID

PathLike = Union[str, Path]


def load_novelty_dataset(split: Optional[str] = None, config: str = "default", path: Optional[PathLike] = None,
                         **kwargs):
    """Load the dataset from the Hugging Face Hub, or from a local copy of the release folder.

    Args:
        split: "train", "test", or None for a DatasetDict with both splits.
        config: "default" (one record per paper) or "baseline_reviews".
        path: local release folder (containing data/ and baseline_reviews/); overrides the Hub.
        **kwargs: passed to `datasets.load_dataset` (e.g. `streaming=True`).
    """
    from datasets import load_dataset

    if path is None:
        return load_dataset(DATASET_ID, config, split=split, **kwargs)

    folder = Path(path) / ("data" if config == "default" else config)
    splits = sorted({f.name.split("-")[0] for f in folder.glob("*.parquet")})
    data_files = {s: str(folder / f"{s}-*.parquet") for s in splits}
    return load_dataset("parquet", data_files=data_files, split=split, **kwargs)


def eval_subset(dataset):
    """The 500 test papers used for the evaluation in the paper."""
    return dataset.filter(lambda row: row["in_eval_subset"])


def reviews_frame(dataset):
    """One row per review, with the paper's id, venue, year and novelty score (pandas DataFrame)."""
    import pandas as pd

    keep = ["paper_id", "venue", "year", "split", "novelty_score"]
    df = dataset.select_columns(keep + ["reviews"]).to_pandas()
    df = df.explode("reviews", ignore_index=True)
    reviews = pd.json_normalize(df.pop("reviews"))
    # Scores are missing for venues whose review form lacks them; make them numeric (NaN) rather than object columns.
    for column in reviews.columns.drop("review_text"):
        reviews[column] = pd.to_numeric(reviews[column], errors="coerce").astype(float)
    return pd.concat([df[keep], reviews], axis=1)


def openreview_url(paper_id: str) -> str:
    return f"https://openreview.net/forum?id={paper_id}"
