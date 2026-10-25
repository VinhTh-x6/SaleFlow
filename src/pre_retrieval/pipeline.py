"""Pre-retrieval pipeline điều phối các bộ biến đổi query."""

from __future__ import annotations
import logging
from .base import BaseTransformer, TransformResult
from .rewriter import QueryRewriter

logger = logging.getLogger(__name__)

class PreRetrievalPipeline:
    def __init__(
        self,
        transformations: list[str] | None = None,
        llm_model: str = "gpt-4.1-mini",
        llm_provider: str = "openai",
        language: str = "both",
    ):
        self.transformations = transformations or ["rewrite"]
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.language = language

        self._transformers: list[BaseTransformer] = []
        for name in self.transformations:
            name_lower = name.lower()
            if name_lower == "rewrite":
                self._transformers.append(
                    QueryRewriter(
                        llm_model=self.llm_model,
                        llm_provider=self.llm_provider,
                        language=self.language,
                    )
                )

    def transform(self, query: str, history_text: str = "") -> TransformResult:
        if not self._transformers or "none" in [t.lower() for t in self.transformations]:
            return TransformResult(original_query=query, queries=[query])

        current_res = TransformResult(original_query=query, queries=[])
        for t in self._transformers:
            res = t.transform(query, history_text=history_text)
            current_res.queries.extend(res.queries)
            if res.metadata_filter:
                current_res.metadata_filter = res.metadata_filter

        if not current_res.queries:
            current_res.queries = [query]

        return current_res


QueryPipeline = PreRetrievalPipeline
