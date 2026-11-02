"""The prompts must stay byte-identical to the ones used for the dataset, the model and the paper's results.

Trailing spaces are part of the prompts. If a hash changes, the prompt no longer matches what the model was trained
or evaluated with.
"""

import hashlib

import pytest

from novelty_reviewer import prompts

EXPECTED_SHA256 = {
    "AGGREGATION_SYSTEM_PROMPT": "5be525ad94037490a20d3c660b8ff9550ed7ab87c808479633b870541c7049f9",
    "AGGREGATION_USER_PROMPT": "18b981b7291b25c542b9c451f029a29178eef30eecce1f96a76bd3bce58dabf1",
    "BASELINE_PROMPT": "e49ec1ee327bf1dce3005d9fa3c2bbf27255ded27846bfdbc1ce30503cebf076",
    "JUDGE_PROMPT": "92729813f74097a1f15b30aaa08633315cc0afbe9e51384309f4a72b90287464",
    "KNOWLEDGE_EXTRACTION_PROMPT": "af6c6074494a2efa4022f89b422d8d540b2d5b8a639f37986f7e2c80b1aab6f8",
    "NOVELTY_REVIEW_PROMPT": "d9429ab5206a1d4469ce171081cf04e0a74f290f1ab4bdbc855c3f8f6ca0e34b",
    "NOVELTY_REVIEW_WITH_CONTEXT_PROMPT": "0c168c493b38bd8d0f9ce89d57e82233eb07d12596508c501d03f76177ce5b57",
    "OVERLAP_EXPLANATION_PROMPT": "8eca210b6b3706bcbffb32f50e5ac004d2fb471e8065d5a1513ee7623cff623f",
    "REVIEW_EXCERPTS_SYSTEM_PROMPT": "2d875f21bc3a3f13a01c51d4ed91ea352ca565f97f5e33784fc672da462db140",
    "REVIEW_EXCERPTS_USER_PROMPT": "b241f38d47afa625346550dd33bac4f5191cef698fb9b43d3d92466ed23c6890",
    "REVIEW_STANCE_SYSTEM_PROMPT": "dddb4bce3b51365e4e549560bd51c4e648793654048e5be5184731be6f7a50e5",
    "REVIEW_STANCE_USER_PROMPT": "d76e7a8d6ba1cf7016715a72c789cbd926d56e345e1b5ddefff8519690570b38",
}


@pytest.mark.parametrize("name, digest", EXPECTED_SHA256.items())
def test_prompt_unchanged(name, digest):
    assert hashlib.sha256(getattr(prompts, name).encode()).hexdigest() == digest


def test_training_prompt_keeps_trailing_spaces():
    assert "for AI conferences. \n" in prompts.NOVELTY_REVIEW_PROMPT
    assert "Novelty Score: \n" in prompts.NOVELTY_REVIEW_PROMPT
