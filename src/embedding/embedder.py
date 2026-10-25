"""Embedder factory và loader."""

from __future__ import annotations
from typing import Any
from .base import BaseEmbedder
from .openai_embedder import OpenAIEmbedder

def get_embedder(provider: str = "openai", **kwargs: Any) -> BaseEmbedder:
    """Tạo embedder theo provider."""
    if provider.lower() == "openai":
        return OpenAIEmbedder(
            model_name=kwargs.get("model_name", "text-embedding-3-small"),
            dimensions=kwargs.get("dimensions", None),
        )
    raise ValueError(f"Provider '{provider}' không được hỗ trợ. Chọn 'openai'.")
