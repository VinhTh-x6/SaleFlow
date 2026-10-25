"""Reranker factory and facade."""

from __future__ import annotations
from typing import Any
from .base import BasePostProcessor
from .cross_encoder_reranker import CrossEncoderReranker
from .pipeline import PostRetrievalPipeline

def get_reranker(strategy: str = "cross_encoder", **kwargs: Any) -> BasePostProcessor | None:
    if strategy.lower() == "cross_encoder":
        return CrossEncoderReranker(
            model_name=kwargs.get("cross_encoder_model", "BAAI/bge-reranker-v2-m3"),
            top_n=kwargs.get("top_n", 5),
            batch_size=kwargs.get("batch_size", 32),
            device=kwargs.get("device", "cpu"),
        )
    return None
