from __future__ import annotations
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

def corpus_fingerprint(chunks: list[Document]) -> str:
    hasher = hashlib.md5()
    for chunk in chunks:
        hasher.update(chunk.page_content.encode("utf-8", errors="replace"))
        src = str(chunk.metadata.get("source", ""))
        hasher.update(src.encode("utf-8", errors="replace"))
    return hasher.hexdigest()

def save_fingerprint(fingerprint: str, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")

def load_fingerprint(path: str) -> str | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("fingerprint")
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def corpus_changed(chunks: list[Document], fingerprint_path: str) -> bool:
    current = corpus_fingerprint(chunks)
    previous = load_fingerprint(fingerprint_path)
    return current != previous

class _EmbedderAdapter(Embeddings):
    def __init__(self, inner):
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def __getattr__(self, item):
        return getattr(self._inner, item)

class BaseVectorStore(ABC):
    def __init__(self, collection_name: str = "rag_collection", force_reindex: bool = False):
        self.collection_name = collection_name
        self.force_reindex = force_reindex

    @abstractmethod
    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        pass

    @staticmethod
    def _to_langchain_embeddings(embedder) -> Embeddings:
        if isinstance(embedder, Embeddings):
            return embedder
        if hasattr(embedder, "embed_documents") and hasattr(embedder, "embed_query"):
            return _EmbedderAdapter(embedder)
        raise TypeError(f"Đối tượng {type(embedder)} không hỗ trợ embed_documents/embed_query.")
