"""Pydantic and dataclass models for RAG pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, Field

# --- Core RAG Data Models ---

@dataclass
class Chunk:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentModel:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryModel:
    query: str
    session_id: str | None = None

@dataclass
class RetrievalResult:
    documents: list[Any] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

@dataclass
class QueryTransformationResult:
    queries: list[str] = field(default_factory=list)

@dataclass
class QueryResult:
    answer: str
    intent: str
    product_id: str | None
    query_used: str
    final_docs: list[Any]
    cited_sources: list[int] = field(default_factory=list)

# --- API Models ---

class QueryRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi cần tra cứu")
    session_id: str | None = Field(default=None, description="Mã session hội thoại")

class ClearSessionRequest(BaseModel):
    session_id: str = Field(..., description="Mã session cần xoá")

class SourceItem(BaseModel):
    index: int
    filename: str
    source: str
    content: str

class QueryResponse(BaseModel):
    query: str
    query_used: str
    session_id: str | None = None
    answer: str
    intent: str = "chat"
    product_id: str | None = None
    cited_sources: list[int] = []
    sources: list[SourceItem] = []

class IndexRequest(BaseModel):
    source: str | None = None
    reindex: bool = False

class IndexResponse(BaseModel):
    success: bool
    n_docs: int
    n_chunks: int
    message: str
