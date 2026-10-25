"""Hybrid retrieval: Kết hợp Dense (Chroma) và Sparse (BM25) qua RRF."""

from __future__ import annotations
from collections import defaultdict
from typing import Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from ..pre_retrieval.base import TransformResult
from .base import BaseRetriever
from .dense import DenseRetriever
from .sparse import SparseRetriever

def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = 60
) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.page_content.strip()
            scores[key] += 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc

    for key, doc in doc_map.items():
        doc.metadata["rrf_score"] = scores[key]

    return [doc_map[k] for k in sorted(scores, key=scores.__getitem__, reverse=True)]

class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: VectorStore,
        documents: list[Document],
        top_k: int = 15,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        rrf_k: int = 60,
        score_threshold: float = 0.0,
    ):
        super().__init__(vector_store, top_k)
        self.rrf_k = rrf_k
        d_k = dense_top_k or top_k * 2
        s_k = sparse_top_k or top_k * 2

        self.dense_retriever = DenseRetriever(
            vector_store=vector_store,
            top_k=d_k,
            score_threshold=score_threshold,
        )
        self.sparse_retriever = SparseRetriever(
            documents=documents,
            top_k=s_k,
        )

    def retrieve(self, result: Union[TransformResult, str]) -> list[Document]:
        dense_docs = self.dense_retriever.retrieve(result)
        sparse_docs = self.sparse_retriever.retrieve(result)
        fused = reciprocal_rank_fusion([dense_docs, sparse_docs], k=self.rrf_k)
        return fused[: self.top_k]
