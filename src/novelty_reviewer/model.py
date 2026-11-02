"""The Novelty Reviewer model and a thin wrapper around Hugging Face chat models."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from .constants import MODEL_ID
from .parsing import NoveltyPrediction, parse_novelty_output
from .prompts import NOVELTY_REVIEW_PROMPT, NOVELTY_REVIEW_WITH_CONTEXT_PROMPT

# Decoding used for all evaluations in the paper.
DEFAULT_GENERATION = {"max_new_tokens": 300, "do_sample": True, "temperature": 0.1, "top_p": 0.5}


class ChatModel:
    """A causal language model with a chat template, loaded from the Hugging Face Hub or a local path."""

    def __init__(self, model_id: str, dtype: str = "bfloat16", device_map: str = "auto", **model_kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map=device_map, **model_kwargs
        ).eval()

    def generate(self, messages: Sequence[dict], **generation_kwargs) -> str:
        """Generate the assistant reply to a list of chat messages."""
        import torch

        input_ids = self.tokenizer.apply_chat_template(
            list(messages), add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs,
            )
        return self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)


class NoveltyReviewer:
    """Predicts a novelty score in {-1, 0, 1, 2} and a reviewer-style novelty assessment for a manuscript.

    Example:
        >>> reviewer = NoveltyReviewer.from_pretrained()
        >>> prediction = reviewer.predict(paper_markdown)
        >>> prediction.score, prediction.label
        (1, 'Moderately novel')
    """

    def __init__(self, chat_model: ChatModel):
        self.chat_model = chat_model

    @classmethod
    def from_pretrained(cls, model_id: str = MODEL_ID, **kwargs) -> "NoveltyReviewer":
        return cls(ChatModel(model_id, **kwargs))

    @staticmethod
    def build_messages(paper_text: str, related_work: Optional[Iterable[str]] = None) -> list[dict]:
        """Chat messages for a manuscript, optionally with summaries of retrieved related work."""
        if related_work:
            prompt = NOVELTY_REVIEW_WITH_CONTEXT_PROMPT.format(
                paper_text=paper_text, similar_papers="\n\n".join(related_work)
            )
        else:
            prompt = NOVELTY_REVIEW_PROMPT.format(paper_text=paper_text)
        return [{"role": "user", "content": prompt}]

    def predict(self, paper_text: str, related_work: Optional[Iterable[str]] = None,
                **generation_kwargs) -> NoveltyPrediction:
        """Assess the novelty of `paper_text` (the paper as markdown or plain text)."""
        kwargs = {**DEFAULT_GENERATION, **generation_kwargs}
        text = self.chat_model.generate(self.build_messages(paper_text, related_work), **kwargs)
        return parse_novelty_output(text)
