"""BM25 index for keyword & hybrid search."""

from __future__ import annotations
import re

class BM25Index:
    """BM25 Okapi index bằng rank-bm25."""
    def __init__(self):
        self._bm25 = None
        self.corpus_size = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def fit(self, corpus: list[str]) -> "BM25Index":
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [self._tokenize(t) for t in corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_size = len(corpus)
        return self

    def get_scores(self, query: str) -> list[float]:
        if self._bm25 is None:
            raise RuntimeError("Cần gọi fit(corpus) trước khi tính điểm BM25.")
        query_tokens = self._tokenize(query)
        return list(self._bm25.get_scores(query_tokens))

# Alias for backward compatibility
BM25Embedder = BM25Index
