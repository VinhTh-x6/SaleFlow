from __future__ import annotations
from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BasePostProcessor(ABC):
    @abstractmethod
    def process(self, query: str, docs: list[Document]) -> list[Document]:
        pass
