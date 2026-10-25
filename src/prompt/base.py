from __future__ import annotations
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from langchain_core.documents import Document

@dataclass
class PromptResult:
    """Kết quả xây dựng prompt chuẩn bị đưa vào LLM."""
    messages: list[dict] = field(default_factory=list)
    full_prompt: str = ""
    context_docs: list[Document] = field(default_factory=list)
    n_sources: int = 0
    template_name: str = "safe_rag"

class BasePromptBuilder(ABC):
    def __init__(self, max_context_chars: int = 0):
        self.max_context_chars = max_context_chars

    @abstractmethod
    def build(
        self,
        query: str,
        docs: list[Document],
        history: list[dict] | None = None,
    ) -> PromptResult:
        pass

    def _format_context(self, docs: list[Document]) -> str:
        if not docs:
            return "(Không tìm thấy tài liệu phù hợp trong cơ sở tri thức)"

        parts: list[str] = []
        total_len = 0

        for i, doc in enumerate(docs, start=1):
            source_name = doc.metadata.get("filename") or doc.metadata.get("source") or "Tài liệu"
            chunk_text = f"--- [NGUỒN {i}] ({source_name}) ---\n{doc.page_content.strip()}\n"

            if self.max_context_chars > 0 and (total_len + len(chunk_text)) > self.max_context_chars:
                break

            parts.append(chunk_text)
            total_len += len(chunk_text)

        return "\n".join(parts)

    @staticmethod
    def extract_cited_indices(answer: str) -> list[int]:
        pattern = r"\[(?:NGUỒN|SOURCE)\s*(\d+)\]"
        matches = re.findall(pattern, answer, re.IGNORECASE)
        return sorted(list(set(int(m) for m in matches)))
