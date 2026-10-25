"""Semantic Chunker: Phát hiện ranh giới chuyển đổi chủ đề dựa trên embeddings."""

from __future__ import annotations
import logging
from langchain_core.documents import Document
from .base import BaseChunker

logger = logging.getLogger(__name__)

class SemanticChunker(BaseChunker):
    def __init__(
        self,
        embedding_model_name: str = "text-embedding-3-small",
        breakpoint_type: str = "percentile",
        breakpoint_threshold: float = 95.0,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.embedding_model_name = embedding_model_name
        self.breakpoint_type = breakpoint_type
        self.breakpoint_threshold = breakpoint_threshold

    def _build_embeddings(self):
        if "text-embedding" in self.embedding_model_name or "openai" in self.embedding_model_name.lower():
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=self.embedding_model_name)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    def split(self, docs: list[Document]) -> list[Document]:
        from langchain_experimental.text_splitter import (
            SemanticChunker as LCSemanticChunker,
        )

        embeddings = self._build_embeddings()
        splitter = LCSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=self.breakpoint_type,
            breakpoint_threshold_amount=self.breakpoint_threshold,
        )

        all_chunks: list[Document] = []
        for doc in docs:
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)

        return self._enrich(all_chunks)
