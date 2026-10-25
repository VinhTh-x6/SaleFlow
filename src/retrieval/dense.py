"""Dense retrieval dựa trên độ tương đồng vector."""

from __future__ import annotations
import logging
from typing import Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from ..pre_retrieval.base import TransformResult
from .base import BaseRetriever, extract_query_and_filter

logger = logging.getLogger(__name__)

class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 15,
        score_threshold: float = 0.0,
    ):
        super().__init__(vector_store, top_k)
        self.score_threshold = score_threshold

    def retrieve(self, result: Union[TransformResult, str]) -> list[Document]:
        query, meta_filter = extract_query_and_filter(result)
        return self._search(
            query=query,
            k=self.top_k,
            filter=meta_filter,
            score_threshold=self.score_threshold,
        )

    def _search(
        self,
        query: str,
        k: int,
        filter: dict | None,
        score_threshold: float,
    ) -> list[Document]:
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k, filter=filter
            )
        except Exception:
            results = self.vector_store.similarity_search_with_score(
                query, k=k, filter=filter
            )

        docs: list[Document] = []
        for doc, score in results:
            if score_threshold > 0.0 and score < score_threshold:
                continue
            doc.metadata["retrieval_score"] = score
            docs.append(doc)
        return docs
