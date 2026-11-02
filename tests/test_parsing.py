import pytest

from novelty_reviewer.parsing import (parse_aggregated_assessment, parse_json_object, parse_novelty_output,
                                      parse_novelty_score)


@pytest.mark.parametrize("text, expected", [
    ("Novelty Score: 1\nThe paper ...", 1),
    ("Novelty Score: -1", -1),
    ("**Novelty Score:** 2", 2),
    ("Novelty Score: [0]", 0),
    ("novelty score: 2", 2),
    ("Novelty Score: 3", None),      # outside the scale
    ("No score here", None),
    ("", None),
    (None, None),
])
def test_parse_novelty_score(text, expected):
    assert parse_novelty_score(text) == expected


def test_parse_output_of_fine_tuned_model():
    # The fine-tuned model repeats the score line before the aggregated assessment.
    text = "Novelty Score: 1\n Novelty Score: 1\n\nScore Justification:\nA.\n\nDetailed Assessment:\nB."
    prediction = parse_novelty_output(text)
    assert prediction.score == 1
    assert prediction.label == "Moderately novel"
    assert "Novelty Score" not in prediction.review
    assert prediction.review.startswith("Score Justification:")


def test_parse_output_with_short_review_section():
    prediction = parse_novelty_output("Novelty Score: -1\n\nShort Novelty Review: Derivative work.")
    assert (prediction.score, prediction.review) == (-1, "Derivative work.")


def test_parse_aggregated_assessment():
    text = "Novelty Score: 0\n\nScore Justification:\nMinor variation.\n\nDetailed Assessment:\nBuilds on X."
    assert parse_aggregated_assessment(text) == {
        "novelty_score": 0,
        "novelty_score_justification": "Minor variation.",
        "novelty_detailed_assessment": "Builds on X.",
    }


def test_parse_aggregated_assessment_without_justification_header():
    text = "Novelty Score: 2\nNew formulation.\nDetailed Assessment:\nFirst to do Y."
    parsed = parse_aggregated_assessment(text)
    assert parsed["novelty_score_justification"] == "New formulation."


@pytest.mark.parametrize("text", [
    '{"core_ideas": ["a"]}',
    'Here you go:\n```json\n{"core_ideas": ["a"]}\n```',
    'Sure! {"core_ideas": ["a"]} Hope this helps.',
])
def test_parse_json_object(text):
    assert parse_json_object(text) == {"core_ideas": ["a"]}


def test_parse_json_object_default():
    assert parse_json_object("not json", {"x": 1}) == {"x": 1}
