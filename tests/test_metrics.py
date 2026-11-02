import pytest

from novelty_reviewer.evaluation.metrics import classification_metrics, score_predictions


def test_perfect_predictions():
    m = classification_metrics([-1, 0, 1, 2], [-1, 0, 1, 2])
    assert m["accuracy"] == m["f1"] == m["coverage"] == 1.0
    assert m["spearman"] == pytest.approx(1.0)


def test_missing_scores_are_dropped():
    m = classification_metrics([1, 1, None, 0], [1, None, 1, 0])
    assert m["n"] == 2 and m["coverage"] == 0.5 and m["accuracy"] == 1.0


def test_constant_predictions_have_no_correlation():
    m = classification_metrics([0, 1, 2], [1, 1, 1])
    assert "spearman" not in m
    assert m["accuracy"] == pytest.approx(1 / 3)


def test_score_predictions_parses_text():
    m = score_predictions([1, 0], ["Novelty Score: 1\n...", "**Novelty Score:** 0"])
    assert m["accuracy"] == 1.0


def test_no_valid_pairs():
    assert classification_metrics([None], [1]) == {"n": 0, "coverage": 0.0}
