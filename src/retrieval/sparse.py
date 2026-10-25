"""Sparse BM25 retrieval trên corpus chunks."""

from __future__ import annotations
from typing import Union
from langchain_core.documents import Document
from ..pre_retrieval.base import TransformResult
from ..lexical.bm25_index import BM25Index
from .base import extract_query_and_filter

class SparseRetriever:
    def __init__(self, documents: list[Document], top_k: int = 15):
        self.documents = documents
        self.top_k = top_k
        self._bm25_index = None
        self._init_bm25()

    def _init_bm25(self):
        if not self.documents:
            return
        corpus = [doc.page_content for doc in self.documents]
        self._bm25_index = BM25Index().fit(corpus)

    def retrieve(self, result: Union[TransformResult, str]) -> list[Document]:
        if not self._bm25_index or not self.documents:
            return []

        query, _ = extract_query_and_filter(result)
        scores = self._bm25_index.get_scores(query)

        scored_pairs = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[: self.top_k]

        docs: list[Document] = []
        for idx, score in scored_pairs:
            doc = self.documents[idx]
            new_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": score},
            )
            docs.append(new_doc)
        return docs

SparseBM25Retriever = SparseRetriever
