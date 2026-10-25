from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class TransformResult:
    """Kết quả biến đổi query trước khi truy hồi."""
    original_query: str
    queries: list[str] = field(default_factory=list)
    metadata_filter: dict | None = None
    extra: dict = field(default_factory=dict)

    def all_queries(self) -> list[str]:
        seen = set()
        result = []
        for q in [self.original_query] + self.queries:
            k = q.strip().lower()
            if k and k not in seen:
                seen.add(k)
                result.append(q.strip())
        return result

class BaseTransformer(ABC):
    """Lớp cơ sở cho transformer."""
    def __init__(
        self,
        llm_model: str = "gpt-4.1-mini",
        llm_provider: str = "openai",
        language: str = "both",
    ):
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.language = language

    @abstractmethod
    def transform(self, query: str, history_text: str = "") -> TransformResult:
        """Biến đổi câu truy vấn."""
