from __future__ import annotations
import hashlib
from abc import ABC, abstractmethod
from typing import Literal
from langchain_core.documents import Document

Language = Literal["vi", "en", "both"]

def clean_text(text: str) -> str:
    """Chuẩn hoá văn bản sau khi trích xuất: xoá null byte, chuẩn hoá dòng trống."""
    text = text.replace("\x00", "")
    lines = [line.rstrip() for line in text.splitlines()]

    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append(line)
        else:
            blank_run = 0
            cleaned.append(line)

    return "\n".join(cleaned).strip()

def content_hash(text: str) -> str:
    """MD5 fingerprint cho text."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()

class BaseLoader(ABC):
    """Lớp cơ sở cho Document Loader."""
    def __init__(self, language: Language = "both"):
        self.language = language

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        """Tải một file và trả về danh sách Document."""

    def _stamp(self, docs: list[Document]) -> list[Document]:
        """Gắn metadata mặc định."""
        for doc in docs:
            doc.metadata.setdefault("language", self.language)
        return docs
