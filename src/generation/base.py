"""
Base class và Data models cho LLM Generation.
"""

from __future__ import annotations
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator
from ..prompt.base import PromptResult

@dataclass
class GenerationResult:
    """Kết quả sinh câu trả lời từ LLM."""
    answer: str = ""
    provider: str = "openai"
    model_name: str = "gpt-4.1-mini"
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    cited_sources: list[int] = field(default_factory=list)

class BaseGenerator(ABC):
    """Lớp cơ sở cho LLM Generator."""
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        streaming: bool = True,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming

    @abstractmethod
    def generate(self, prompt_result: PromptResult) -> GenerationResult:
        """Sinh câu trả lời từ PromptResult."""
        pass

    @abstractmethod
    def stream(self, prompt_result: PromptResult) -> Iterator[str]:
        """Stream từng token câu trả lời từ PromptResult."""
        pass

    def _post_process(
        self,
        answer: str,
        prompt_result: PromptResult,
        input_tokens: int = 0,
        output_tokens: int = 0,
        finish_reason: str = "stop",
    ) -> GenerationResult:
        cited = self._extract_cited_sources(answer)
        return GenerationResult(
            answer=answer,
            provider="openai",
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            cited_sources=cited,
        )

    @staticmethod
    def _extract_cited_sources(text: str) -> list[int]:
        pattern = r"\[NGUỒN\s+(\d+)\]"
        matches = re.findall(pattern, text, re.IGNORECASE)
        seen = set()
        result = []
        for m in matches:
            idx = int(m)
            if idx not in seen:
                seen.add(idx)
                result.append(idx)
        return sorted(result)
