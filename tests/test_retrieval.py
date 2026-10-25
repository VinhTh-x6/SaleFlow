"""
Unit test kiểm thử module retrieval và kiến trúc src RAG.
"""

from __future__ import annotations
import sys
import unittest
from pathlib import Path
from langchain_core.documents import Document

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseBM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.schemas.models import RetrievalResult

class TestRAGStructure(unittest.TestCase):
    def test_imports(self):
        """Kiểm tra import tất cả các module chính của src thành công."""
        from src.schemas import DocumentModel, QueryModel, RetrievalResult
        from src.loader import get_loader
        from src.chunking import get_chunker
        from src.embedding import get_embedder
        from src.vector_db import get_vector_store
        from src.retrieval import get_retriever
        from src.post_retrieval import get_reranker
        from src.prompt import get_prompt_builder
        from src.generation import get_generator
        from src.pipeline import IndexingPipeline, RAGPipeline
        self.assertTrue(True)

    def test_sparse_bm25_retriever(self):
        """Kiểm tra SparseBM25Retriever tìm kiếm từ khoá chính xác."""
        docs = [
            Document(page_content="Lập trình Python cơ bản dành cho người mới bắt đầu", metadata={"id": 1}),
            Document(page_content="Hướng dẫn xây dựng hệ thống RAG với ChromaDB và OpenAI", metadata={"id": 2}),
            Document(page_content="Tuyển tập câu hỏi phỏng vấn vị trí Backend Golang", metadata={"id": 3}),
        ]
        retriever = SparseBM25Retriever(documents=docs, top_k=1)
        results = retriever.retrieve("Python cơ bản")
        self.assertEqual(len(results), 1)
        self.assertIn("Python", results[0].page_content)

if __name__ == "__main__":
    unittest.main()
