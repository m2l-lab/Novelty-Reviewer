"""LLM-as-a-judge: agreement of a generated assessment with the reference ECNA assessment, on a 0-10 scale."""

from __future__ import annotations

import re
from typing import Optional

from ..constants import BASE_MODEL_ID
from ..model import ChatModel
from ..prompts import JUDGE_PROMPT


class LLMJudge:
    def __init__(self, chat_model: Optional[ChatModel] = None):
        self.chat_model = chat_model or ChatModel(BASE_MODEL_ID)

    def score(self, model_output: str, reference_summary: str, reference_score: int) -> int:
        prompt = JUDGE_PROMPT.format(novelty_summary=reference_summary, novelty_score=reference_score,
                                     model_output=model_output)
        reply = self.chat_model.generate([{"role": "user", "content": prompt}],
                                         max_new_tokens=50, do_sample=True, temperature=0.1, top_p=0.5)
        match = re.search(r"\d+", reply)
        return min(int(match.group()), 10) if match else 0
