"""ChromaDB persistent vector store."""

from __future__ import annotations
import functools
import logging
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .base import (
    BaseVectorStore,
    corpus_changed,
    corpus_fingerprint,
    save_fingerprint,
)

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=8)
def _get_or_create_chroma_client(persist_dir: str):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

class ChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_dir: str = "./storage",
        force_reindex: bool = False,
    ):
        super().__init__(collection_name, force_reindex)
        self.persist_dir = persist_dir

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma

        lc_embedder = self._to_langchain_embeddings(embedder)
        fp_path = str(Path(self.persist_dir) / f"{self.collection_name}_fp.json")

        if self.force_reindex and Path(self.persist_dir).exists():
            logger.info("Chroma: force_reindex, dọn dẹp thư mục storage...")
            shutil.rmtree(self.persist_dir, ignore_errors=True)

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        client = _get_or_create_chroma_client(self.persist_dir)

        if not self.force_reindex:
            if not corpus_changed(chunks, fp_path):
                existing_colls = [c.name for c in client.list_collections()]
                if self.collection_name in existing_colls:
                    logger.info("Chroma: Corpus không đổi, nạp collection hiện có.")
                    return Chroma(
                        client=client,
                        collection_name=self.collection_name,
                        embedding_function=lc_embedder,
                    )

        logger.info("Chroma: Tạo mới / cập nhật index cho %d chunks...", len(chunks))
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass

        store = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=lc_embedder,
        )

        batch_size = 200
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            store.add_documents(batch)

        if chunks:
            save_fingerprint(corpus_fingerprint(chunks), fp_path)

        return store
