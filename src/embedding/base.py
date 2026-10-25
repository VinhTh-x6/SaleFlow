from __future__ import annotations
from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings

class BaseEmbedder(ABC):
    """Lớp bọc đối tượng Embeddings của LangChain."""
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self._embedder: Embeddings | None = None

    @abstractmethod
    def _build(self) -> Embeddings:
        """Tạo đối tượng Embeddings."""

    @property
    def embedder(self) -> Embeddings:
        if self._embedder is None:
            self._embedder = self._build()
        return self._embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embedder.embed_query(query)
