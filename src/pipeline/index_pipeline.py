"""
Offline Document Indexing Pipeline.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import yaml

from ..loader.loader import get_loader
from ..chunking.chunker import get_chunker
from ..embedding.embedder import get_embedder
from ..vector_db.vector_store import get_vector_store

logger = logging.getLogger(__name__)

@dataclass
class IndexingResult:
    """Kết quả quá trình lập chỉ mục."""
    success: bool
    n_docs: int = 0
    n_chunks: int = 0
    error: str | None = None
    chunks_file: str | None = None

class IndexingPipeline:
    """Pipeline lập chỉ mục tài liệu vào ChromaDB."""
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

    @staticmethod
    def _load_config(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def run(self, source_path: str | None = None, force_reindex: bool = False) -> IndexingResult:
        try:
            cfg = self.config
            idx_cfg = cfg.get("indexing", {})
            data_cfg = cfg.get("data", {})

            # Hỗ trợ fallback: data/raw -> data
            default_dir = data_cfg.get("input_dir", "./data/raw")
            if source_path:
                src = source_path
            else:
                if Path(default_dir).exists():
                    src = default_dir
                elif Path("./data/raw").exists():
                    src = "./data/raw"
                else:
                    src = "./data"

            p_src = Path(src)
            if not p_src.exists():
                raise FileNotFoundError(f"Thư mục hoặc file nguồn không tồn tại: {src}")

            print(f"\n[1/4] Đang nạp tài liệu từ '{src}'...")
            loader_cfg = idx_cfg.get("loader", {})
            loader = get_loader(
                strategy=loader_cfg.get("pdf_strategy", "docling"),
                language=data_cfg.get("language", "both"),
                extract_tables=loader_cfg.get("extract_tables", True),
                extract_images=loader_cfg.get("extract_images", False),
                marker_device=loader_cfg.get("marker_device", "cpu"),
            )
            docs = loader.load(str(p_src))
            if not docs:
                print(" Không tìm thấy tài liệu nào để xử lý.")
                return IndexingResult(success=True, n_docs=0, n_chunks=0)
            print(f" Đã nạp thành công {len(docs)} tài liệu.")

            print(f"\n[2/4] Đang phân đoạn (chunking) theo chiến lược '{idx_cfg.get('chunking', {}).get('strategy', 'format_aware')}'...")
            chunk_cfg = idx_cfg.get("chunking", {})
            chunker = get_chunker(
                strategy=chunk_cfg.get("strategy", "format_aware"),
                chunk_size=chunk_cfg.get("chunk_size", 1000),
                chunk_overlap=chunk_cfg.get("chunk_overlap", 150),
                format_type=chunk_cfg.get("format_type", "auto"),
                semantic_embedding_model=chunk_cfg.get("semantic_embedding_model", "text-embedding-3-small"),
                semantic_breakpoint_type=chunk_cfg.get("semantic_breakpoint_type", "percentile"),
                semantic_breakpoint_threshold=chunk_cfg.get("semantic_breakpoint_threshold", 95.0),
            )
            chunks = chunker.split(docs)
            print(f" Đã tạo {len(chunks)} chunks.")

            print("\n[3/4] Đang khởi tạo Embedder (OpenAI)...")
            emb_cfg = idx_cfg.get("embedding", {})
            embedder = get_embedder(
                provider=emb_cfg.get("provider", "openai"),
                model_name=emb_cfg.get("model_name", "text-embedding-3-small"),
                dimensions=emb_cfg.get("dimensions", None),
            )

            print("\n[4/4] Đang lưu trữ vào Chroma Vector Store & Chunks Cache...")
            vdb_cfg = idx_cfg.get("vector_db", {})
            persist_dir = vdb_cfg.get("persist_dir", "./storage")
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

            store_builder = get_vector_store(
                provider=vdb_cfg.get("provider", "chroma"),
                collection_name=vdb_cfg.get("collection_name", "rag_collection"),
                persist_dir=persist_dir,
                force_reindex=force_reindex,
            )
            store_builder.get_or_create(chunks, embedder)

            chunks_cache_file = Path(persist_dir) / "indexed_chunks.json"
            chunks_data = [
                {"page_content": c.page_content, "metadata": c.metadata}
                for c in chunks
            ]
            with open(chunks_cache_file, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            print(f" Đã lưu {len(chunks)} chunks vào cache: {chunks_cache_file}")

            print("\n Quá trình Indexing hoàn tất xuất sắc!")
            return IndexingResult(
                success=True,
                n_docs=len(docs),
                n_chunks=len(chunks),
                chunks_file=str(chunks_cache_file),
            )

        except Exception as exc:
            logger.exception("IndexingPipeline thất bại: %s", exc)
            print(f"\n[!] Lỗi Indexing: {exc}")
            return IndexingResult(success=False, error=str(exc))
