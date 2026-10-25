"""Parsers for PDF and documents (Docling & Marker)."""

from __future__ import annotations
import contextlib
import logging
import re
from pathlib import Path
from langchain_core.documents import Document

from .base import BaseLoader, Language, clean_text, content_hash

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def _force_eager_attention():
    orig = None
    owner = None
    try:
        from transformers.models.idefics3 import modeling_idefics3 as _m
        for name in ("Idefics3ForConditionalGeneration", "Idefics3VisionTransformer"):
            cls = getattr(_m, name, None)
            if cls is not None and hasattr(cls, "from_pretrained"):
                orig = cls.from_pretrained
                owner = cls
                break
    except Exception:
        pass

    if orig is None:
        yield
        return

    def _patched(pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("attn_implementation", "eager")
        return orig(pretrained_model_name_or_path, *args, **kwargs)

    try:
        setattr(owner, "from_pretrained", _patched)
        yield
    finally:
        setattr(owner, "from_pretrained", orig)


class DoclingPDFLoader(BaseLoader):
    """Trích xuất PDF bằng Docling (IBM Research)."""
    def __init__(
        self,
        language: Language = "both",
        extract_tables: bool = True,
        extract_images: bool = False,
    ):
        super().__init__(language)
        self.extract_tables = extract_tables
        self.extract_images = extract_images

    def load(self, file_path: str) -> list[Document]:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
        except ImportError as e:
            raise ImportError(
                "Docling chưa được cài đặt. Chạy: pip install docling"
            ) from e

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = self.extract_tables
        pipeline_options.generate_picture_images = self.extract_images

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        with _force_eager_attention():
            result = converter.convert(path)

        markdown = result.document.export_to_markdown()
        cleaned = clean_text(markdown)

        doc = Document(
            page_content=cleaned,
            metadata={
                "source": str(path),
                "filename": path.name,
                "strategy": "docling",
                "content_hash": content_hash(cleaned),
                "file_type": "pdf",
                "char_count": len(cleaned),
            },
        )
        return self._stamp([doc])


def repair_marker_math(markdown: str) -> tuple[str, int]:
    pattern = re.compile(r'(?<!\$)\$([^\$\n]+)\$(?!\$)')
    fixed = pattern.sub(r' $\1$ ', markdown)
    return fixed, 0


class MarkerPDFLoader(BaseLoader):
    """Trích xuất PDF bằng Marker."""
    _LANG_MAP = {"vi": ["vi"], "en": ["en"], "both": ["vi", "en"]}

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def __init__(
        self,
        language: Language = "both",
        langs: list[str] | None = None,
        device: str | None = None,
        workers: int = 1,
    ):
        super().__init__(language)
        self._langs = langs or self._LANG_MAP.get(language, ["vi", "en"])
        self.device = device or self._auto_device()
        self.workers = workers

    def load(self, file_path: str) -> list[Document]:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.config.parser import ConfigParser
        except ImportError as e:
            raise ImportError(
                "Marker chưa được cài đặt. Chạy: pip install marker-pdf"
            ) from e

        config_parser = ConfigParser({"output_format": "markdown"})
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        rendered = converter(str(path))
        raw_md = rendered.markdown if hasattr(rendered, "markdown") else str(rendered)
        repaired_md, _ = repair_marker_math(raw_md)
        cleaned = clean_text(repaired_md)

        doc = Document(
            page_content=cleaned,
            metadata={
                "source": str(path),
                "filename": path.name,
                "strategy": "marker",
                "content_hash": content_hash(cleaned),
                "file_type": "pdf",
                "char_count": len(cleaned),
            },
        )
        return self._stamp([doc])
