"""Retriever factory and registry."""

from __future__ import annotations
from typing import Any
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .base import BaseRetriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever

def get_retriever(
    strategy: str,
    vector_store: VectorStore,
    documents: list[Document] | None = None,
    **kwargs: Any,
) -> BaseRetriever:
    strat = strategy.lower()
    top_k = kwargs.get("top_k", 15)
    score_threshold = kwargs.get("search_score_threshold", 0.0)

    if strat == "hybrid":
        return HybridRetriever(
            vector_store=vector_store,
            documents=documents or [],
            top_k=top_k,
            rrf_k=kwargs.get("rrf_k", 60),
            score_threshold=score_threshold,
        )
    elif strat == "dense":
        return DenseRetriever(
            vector_store=vector_store,
            top_k=top_k,
            score_threshold=score_threshold,
        )
    else:
        raise ValueError(f"Retrieval strategy '{strategy}' không được hỗ trợ. Chọn 'hybrid' hoặc 'dense'.")
