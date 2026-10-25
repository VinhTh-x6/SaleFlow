from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from ..pre_retrieval.base import TransformResult

def extract_query_and_filter(result: Union[TransformResult, str]) -> tuple[str, dict | None]:
    """Trích xuất query string và metadata filter từ TransformResult hoặc str thuần."""
    if isinstance(result, str):
        return result, None
    query = result.queries[0] if (hasattr(result, "queries") and result.queries) else getattr(result, "original_query", "")
    meta_filter = getattr(result, "metadata_filter", None)
    return query, meta_filter

class BaseRetriever(ABC):
    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    @abstractmethod
    def retrieve(self, result: Union[TransformResult, str]) -> list[Document]:
        pass
