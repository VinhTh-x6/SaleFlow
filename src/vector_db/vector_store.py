"""Vector store factory and registry."""

from __future__ import annotations
from typing import Any
from .base import BaseVectorStore
from .chroma_store import ChromaVectorStore

def get_vector_store(provider: str = "chroma", **kwargs: Any) -> BaseVectorStore:
    """Tạo Vector Store theo cấu hình (mặc định Chroma)."""
    if provider.lower() == "chroma":
        return ChromaVectorStore(
            collection_name=kwargs.get("collection_name", "rag_collection"),
            persist_dir=kwargs.get("persist_dir", "./storage"),
            force_reindex=kwargs.get("force_reindex", False),
        )
    raise ValueError(f"Vector DB provider '{provider}' không được hỗ trợ. Chọn 'chroma'.")
