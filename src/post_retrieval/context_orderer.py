"""Context orderer (Sandwich ordering chống Lost-in-the-Middle)."""

from __future__ import annotations
from langchain_core.documents import Document
from .base import BasePostProcessor

class ContextOrderer(BasePostProcessor):
    def __init__(self, ordering: str = "sandwich"):
        self.ordering = ordering

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        if len(docs) <= 2 or self.ordering == "original":
            return docs

        def get_score(d: Document) -> float:
            for k in ("rerank_score", "rrf_score", "score", "bm25_score"):
                if k in d.metadata:
                    return float(d.metadata[k])
            return 0.0

        sorted_docs = sorted(docs, key=get_score, reverse=True)

        if self.ordering == "relevance":
            return sorted_docs

        if self.ordering == "sandwich":
            head: list[Document] = []
            tail: list[Document] = []
            for i, doc in enumerate(sorted_docs):
                if i % 2 == 0:
                    head.append(doc)
                else:
                    tail.append(doc)
            tail.reverse()
            return head + tail

        return sorted_docs
