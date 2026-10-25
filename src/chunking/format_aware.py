"""Format-aware chunking: Cắt theo ranh giới tự nhiên của Markdown, Code, HTML."""

from __future__ import annotations
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BaseChunker

_LANG_ENUM_MAP = {
    "python": "PYTHON",
    "javascript": "JS",
    "typescript": "TS",
    "java": "JAVA",
    "cpp": "CPP",
    "c": "C",
    "go": "GO",
    "rust": "RUST",
    "ruby": "RUBY",
    "sql": "SQL",
}

class FormatAwareChunker(BaseChunker):
    def __init__(
        self,
        format_type: str = "auto",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        split_large_sections: bool = True,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.format_type = format_type
        self.split_large_sections = split_large_sections
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split(self, docs: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []
        for doc in docs:
            fmt = self._detect(doc)
            if fmt == "markdown":
                chunks = self._split_markdown(doc)
            elif fmt == "code":
                chunks = self._split_code(doc)
            elif fmt == "html":
                chunks = self._split_html(doc)
            else:
                chunks = self.char_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        return self._enrich(all_chunks)

    def _detect(self, doc: Document) -> str:
        if self.format_type != "auto":
            return self.format_type
        ft = doc.metadata.get("file_type", "")
        if ft in ("markdown", "md"):
            return "markdown"
        if ft == "html":
            return "html"
        if ft == "code" or doc.metadata.get("programming_language"):
            return "code"
        if doc.page_content.lstrip().startswith("#"):
            return "markdown"
        return "text"

    def _maybe_split_large(
        self, sections: list[Document], extra_meta: dict
    ) -> list[Document]:
        if not self.split_large_sections:
            return [
                Document(
                    page_content=s.page_content.strip(),
                    metadata={**s.metadata, **extra_meta},
                )
                for s in sections
                if s.page_content.strip()
            ]

        chunks: list[Document] = []
        for s in sections:
            content = s.page_content.strip()
            if not content:
                continue
            base_meta = {**s.metadata, **extra_meta}
            if len(content) <= self.chunk_size:
                chunks.append(Document(page_content=content, metadata=base_meta))
            else:
                chunks.extend(
                    self.char_splitter.split_documents(
                        [Document(page_content=content, metadata=base_meta)]
                    )
                )
        return chunks

    def _split_markdown(self, doc: Document) -> list[Document]:
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        headers = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
        try:
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers, strip_headers=False
            )
            sections = splitter.split_text(doc.page_content)
            for s in sections:
                s.metadata = {**doc.metadata, **s.metadata}
            return self._maybe_split_large(sections, {"format": "markdown"})
        except Exception:
            return self.char_splitter.split_documents([doc])

    def _split_code(self, doc: Document) -> list[Document]:
        from langchain_text_splitters import Language

        prog_lang = doc.metadata.get("programming_language", "python")
        enum_name = _LANG_ENUM_MAP.get(prog_lang, "PYTHON")
        lc_language = Language[enum_name]

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lc_language,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents([doc])

    def _split_html(self, doc: Document) -> list[Document]:
        from langchain_text_splitters import HTMLHeaderTextSplitter

        headers = [("h1", "h1"), ("h2", "h2"), ("h3", "h3")]
        try:
            splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers)
            sections = splitter.split_text(doc.page_content)
            for s in sections:
                s.metadata = {**doc.metadata, **s.metadata}
            return self._maybe_split_large(sections, {"format": "html"})
        except Exception:
            return self.char_splitter.split_documents([doc])
