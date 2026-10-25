"""
Prompt templates và Factory khởi tạo PromptBuilder cho hệ thống RAG.
"""

from __future__ import annotations
from typing import Any
from .base import BasePromptBuilder, PromptResult
from .safe_rag_prompt import SafeRAGPromptBuilder

def get_prompt_builder(template: str = "safe_rag", **kwargs: Any) -> BasePromptBuilder:
    """Khởi tạo prompt builder dựa trên tên template."""
    template_lower = template.lower()
    if template_lower in ("safe_rag", "sales", "default"):
        return SafeRAGPromptBuilder(
            max_context_chars=kwargs.get("max_context_chars", 0),
        )
    raise ValueError(f"Không hỗ trợ template prompt: '{template}'. Chọn 'safe_rag'.")
