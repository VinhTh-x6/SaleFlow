from __future__ import annotations
from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BaseChunker(ABC):
    """Lớp cơ sở cho việc chia nhỏ tài liệu."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, docs: list[Document]) -> list[Document]:
        """Cắt danh sách Document thành các chunk nhỏ hơn."""

    @staticmethod
    def _enrich(chunks: list[Document]) -> list[Document]:
        """Gắn số thứ tự chunk_index và char_count vào metadata."""
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["char_count"] = len(chunk.page_content)
        return chunks
