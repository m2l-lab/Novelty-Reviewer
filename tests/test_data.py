import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from novelty_reviewer import eval_subset, load_novelty_dataset, reviews_frame

REVIEW = {"review_text": "# Review\n## Rating\n6", "rating": 6, "confidence": 4, "originality": None}


@pytest.fixture
def release_dir(tmp_path):
    """A miniature release folder with the same layout as the published dataset."""
    rows = [
        {"paper_id": "a", "venue": "ICLR", "year": 2024, "split": "train", "in_eval_subset": False,
         "novelty_score": 1, "reviews": [REVIEW, REVIEW]},
        {"paper_id": "b", "venue": "NeurIPS", "year": 2023, "split": "test", "in_eval_subset": True,
         "novelty_score": -1, "reviews": [REVIEW]},
        {"paper_id": "c", "venue": "ICLR", "year": 2022, "split": "test", "in_eval_subset": False,
         "novelty_score": 2, "reviews": [REVIEW]},
    ]
    (tmp_path / "data").mkdir()
    for split in ("train", "test"):
        part = [r for r in rows if r["split"] == split]
        pq.write_table(pa.Table.from_pylist(part), tmp_path / "data" / f"{split}-00000-of-00001.parquet")
    return tmp_path


def test_load_local_release(release_dir):
    ds = load_novelty_dataset(path=release_dir)
    assert set(ds) == {"train", "test"}
    assert ds["train"].num_rows == 1 and ds["test"].num_rows == 2


def test_load_single_split(release_dir):
    assert load_novelty_dataset("test", path=release_dir).num_rows == 2


def test_eval_subset(release_dir):
    subset = eval_subset(load_novelty_dataset("test", path=release_dir))
    assert subset["paper_id"] == ["b"]


def test_reviews_frame(release_dir):
    df = reviews_frame(load_novelty_dataset("train", path=release_dir))
    assert len(df) == 2
    assert {"paper_id", "novelty_score", "rating", "review_text"} <= set(df.columns)
