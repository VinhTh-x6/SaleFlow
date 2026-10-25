"""PostRetrievalPipeline kết hợp: Dedup -> Reranker -> ContextOrderer."""

from __future__ import annotations
import logging
from langchain_core.documents import Document

from .cross_encoder_reranker import CrossEncoderReranker
from .redundancy_filter import RedundancyFilter
from .context_orderer import ContextOrderer

logger = logging.getLogger(__name__)

class PostRetrievalPipeline:
    def __init__(
        self,
        reranker: str = "cross_encoder",
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 5,
        apply_redundancy: bool = True,
        redundancy_threshold: float = 0.92,
        context_ordering: str = "sandwich",
    ):
        self.top_n = top_n
        self.apply_redundancy = apply_redundancy
        self.context_ordering = context_ordering

        self.redundancy_filter = (
            RedundancyFilter(threshold=redundancy_threshold)
            if apply_redundancy
            else None
        )

        self.reranker = (
            CrossEncoderReranker(model_name=cross_encoder_model, top_n=top_n)
            if reranker.lower() == "cross_encoder"
            else None
        )

        self.orderer = (
            ContextOrderer(ordering=context_ordering)
            if context_ordering != "original"
            else None
        )

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        current = docs

        if self.redundancy_filter:
            current = self.redundancy_filter.process(query, current)

        if self.reranker:
            current = self.reranker.process(query, current)
        else:
            current = current[: self.top_n]

        if self.orderer:
            current = self.orderer.process(query, current)

        return current


RerankPipeline = PostRetrievalPipeline
