"""
Các hàm tiện ích chung cho dự án RAG.
"""

from __future__ import annotations
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import yaml

logger = logging.getLogger(__name__)

def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Tải và parse file YAML cấu hình."""
    p = Path(path)
    if not p.exists():
        logger.warning("File cấu hình không tồn tại: %s", path)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

@contextmanager
def timer(name: str):
    """Context manager đo thời gian thực thi của một tác vụ."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info("[%s] Hoàn thành sau %.3f giây", name, elapsed)
