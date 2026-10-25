"""Cross-Encoder Reranker."""

from __future__ import annotations
import logging
from langchain_core.documents import Document
from .base import BasePostProcessor

logger = logging.getLogger(__name__)

class CrossEncoderReranker(BasePostProcessor):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 5,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.top_n = top_n
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        logger.info("Nạp Cross-Encoder '%s' trên %s...", self.model_name, self.device)
        self._model = CrossEncoder(self.model_name, device=self.device)

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []

        try:
            self._load()
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self._model.predict(pairs, batch_size=self.batch_size)

            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: float(x[1]), reverse=True)

            results: list[Document] = []
            for doc, score in scored_docs[: self.top_n]:
                doc.metadata["rerank_score"] = float(score)
                results.append(doc)
            return results
        except Exception as exc:
            logger.warning("Reranking với CrossEncoder gặp lỗi (%s), giữ thứ tự gốc.", exc)
            return docs[: self.top_n]
