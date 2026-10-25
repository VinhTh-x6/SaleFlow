"""
OpenAI Chat Completion Generator.
"""

from __future__ import annotations
from typing import Iterator
from openai import OpenAI
from ..prompt.base import PromptResult
from .base import BaseGenerator, GenerationResult

_NEW_API_PREFIXES = ("o1", "o3", "o4", "gpt-5")

class OpenAIGenerator(BaseGenerator):
    """Generator sử dụng OpenAI Chat API."""
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        streaming: bool = True,
    ):
        super().__init__(model_name, temperature, max_tokens, streaming)

    def _is_reasoning_model(self) -> bool:
        return any(self.model_name.startswith(p) for p in _NEW_API_PREFIXES)

    def _build_kwargs(self, prompt_result: PromptResult | None = None) -> dict:
        kw: dict = {"model": self.model_name}
        if self._is_reasoning_model():
            kw["max_completion_tokens"] = self.max_tokens
        else:
            kw["max_tokens"] = self.max_tokens
            kw["temperature"] = self.temperature
        if prompt_result and "json" in prompt_result.full_prompt.lower() and not self._is_reasoning_model():
            kw["response_format"] = {"type": "json_object"}
        return kw

    def generate(self, prompt_result: PromptResult) -> GenerationResult:
        client = OpenAI()
        kw = self._build_kwargs(prompt_result)
        kw["messages"] = prompt_result.messages

        response = client.chat.completions.create(**kw)
        choice = response.choices[0]
        answer = (choice.message.content or "").strip()

        usage = response.usage
        return self._post_process(
            answer=answer,
            prompt_result=prompt_result,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )

    def stream(self, prompt_result: PromptResult) -> Iterator[str]:
        client = OpenAI()
        kw = self._build_kwargs(prompt_result)
        kw["messages"] = prompt_result.messages
        kw["stream"] = True

        for chunk in client.chat.completions.create(**kw):
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
