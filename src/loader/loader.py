"""Document Loader hỗ trợ file đơn lẻ hoặc cả thư mục (PDF, Markdown, TXT)."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
from langchain_core.documents import Document

from .base import BaseLoader, Language, clean_text, content_hash
from .parsers import DoclingPDFLoader, MarkerPDFLoader

logger = logging.getLogger(__name__)

class DocumentLoader(BaseLoader):
    """Unified Loader cho hệ thống RAG."""
    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}

    def __init__(
        self,
        pdf_strategy: str = "docling",
        language: Language = "both",
        extract_tables: bool = True,
        extract_images: bool = False,
        marker_device: str = "cpu",
    ):
        super().__init__(language)
        self.pdf_strategy = pdf_strategy.lower()
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.marker_device = marker_device

        if self.pdf_strategy == "docling":
            self.pdf_loader = DoclingPDFLoader(
                language=self.language,
                extract_tables=self.extract_tables,
                extract_images=self.extract_images,
            )
        elif self.pdf_strategy == "marker":
            self.pdf_loader = MarkerPDFLoader(
                language=self.language,
                device=self.marker_device,
            )
        else:
            raise ValueError(
                f"PDF strategy '{pdf_strategy}' không được hỗ trợ. Chọn 'docling' hoặc 'marker'."
            )

    def load_file(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            return self.pdf_loader.load(str(path))
        elif ext in {".md", ".markdown", ".txt"}:
            return self._load_plain(path)
        else:
            logger.warning("Bỏ qua file không được hỗ trợ: %s", path)
            return []

    def _load_plain(self, path: Path) -> list[Document]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error("Lỗi khi đọc file %s: %s", path, e)
            return []

        cleaned = clean_text(text)
        doc = Document(
            page_content=cleaned,
            metadata={
                "source": str(path),
                "filename": path.name,
                "strategy": "plain_text",
                "content_hash": content_hash(cleaned),
                "file_type": path.suffix.lstrip("."),
                "char_count": len(cleaned),
            },
        )
        return self._stamp([doc])

    def load_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
    ) -> list[Document]:
        p = Path(dir_path).resolve()
        if not p.is_dir():
            raise NotADirectoryError(f"Không tìm thấy thư mục: {dir_path}")

        pattern = "**/*" if recursive else "*"
        all_files = [
            f for f in p.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        logger.info("Tìm thấy %d file trong %s", len(all_files), p)
        docs: list[Document] = []
        for file in sorted(all_files):
            try:
                sub_docs = self.load_file(file)
                docs.extend(sub_docs)
            except Exception as e:
                logger.error("Lỗi khi nạp file %s: %s", file, e)

        return docs

    def load(self, path: str | Path) -> list[Document]:
        p = Path(path).resolve()
        if p.is_file():
            return self.load_file(p)
        elif p.is_dir():
            return self.load_directory(p)
        raise FileNotFoundError(f"Đường dẫn không tồn tại: {path}")


def get_loader(strategy: str = "docling", **kwargs: Any) -> DocumentLoader:
    """Factory tạo DocumentLoader."""
    return DocumentLoader(pdf_strategy=strategy, **kwargs)
