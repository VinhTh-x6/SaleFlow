"""Lọc bỏ các đoạn tương đồng gần trùng lặp ngữ nghĩa."""

from __future__ import annotations
import logging
from langchain_core.documents import Document
from .base import BasePostProcessor

logger = logging.getLogger(__name__)

class RedundancyFilter(BasePostProcessor):
    def __init__(
        self,
        threshold: float = 0.92,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.threshold = threshold
        self.embedding_model = embedding_model
        self._encoder = None

    def _load_encoder(self):
        if self._encoder is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(self.embedding_model)

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        if len(docs) <= 1:
            return docs

        try:
            self._load_encoder()
            import numpy as np

            embs = self._encoder.encode([d.page_content for d in docs], normalize_embeddings=True)
            kept: list[Document] = []
            kept_embs: list = []

            for doc, emb in zip(docs, embs):
                if not kept_embs:
                    kept.append(doc)
                    kept_embs.append(emb)
                    continue

                sims = np.dot(kept_embs, emb)
                if np.max(sims) < self.threshold:
                    kept.append(doc)
                    kept_embs.append(emb)

            return kept
        except Exception as exc:
            logger.warning("RedundancyFilter lỗi (%s), bỏ qua bước lọc trùng.", exc)
            return docs
