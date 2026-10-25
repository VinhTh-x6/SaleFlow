"""Chunker factory and unified interface."""

from __future__ import annotations
from typing import Any
from .base import BaseChunker
from .format_aware import FormatAwareChunker
from .semantic import SemanticChunker

def get_chunker(strategy: str = "format_aware", **kwargs: Any) -> BaseChunker:
    strategy_clean = strategy.lower()
    if strategy_clean == "format_aware":
        return FormatAwareChunker(
            format_type=kwargs.get("format_type", "auto"),
            chunk_size=kwargs.get("chunk_size", 1000),
            chunk_overlap=kwargs.get("chunk_overlap", 150),
            split_large_sections=kwargs.get("split_large_sections", True),
        )
    elif strategy_clean == "semantic":
        return SemanticChunker(
            embedding_model_name=kwargs.get(
                "semantic_embedding_model", "text-embedding-3-small"
            ),
            breakpoint_type=kwargs.get("semantic_breakpoint_type", "percentile"),
            breakpoint_threshold=kwargs.get("semantic_breakpoint_threshold", 95.0),
            chunk_size=kwargs.get("chunk_size", 1000),
            chunk_overlap=kwargs.get("chunk_overlap", 150),
        )
    else:
        raise ValueError(
            f"Strategy '{strategy}' không được hỗ trợ. Chọn 'format_aware' hoặc 'semantic'."
        )
