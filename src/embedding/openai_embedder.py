"""OpenAI Embedding provider."""

from __future__ import annotations
from langchain_core.embeddings import Embeddings
from .base import BaseEmbedder

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI Embeddings có hỗ trợ Matryoshka dimensions."""
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimensions: int | None = None,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.dimensions = dimensions

    def _build(self) -> Embeddings:
        from langchain_openai import OpenAIEmbeddings

        init_kwargs = {"model": self.model_name, **self.kwargs}
        if self.dimensions and "3" in self.model_name:
            init_kwargs["dimensions"] = self.dimensions
        return OpenAIEmbeddings(**init_kwargs)
