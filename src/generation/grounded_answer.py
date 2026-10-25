"""
Grounded Answer Generator: Factory khởi tạo Generator cùng bộ bóc tách JSON và Streaming chuẩn.
"""

from __future__ import annotations
import json
import re
from typing import Any, Iterator
from .base import BaseGenerator, GenerationResult
from .openai_generator import OpenAIGenerator

def get_generator(provider: str = "openai", **kwargs: Any) -> BaseGenerator:
    """Khởi tạo Generator."""
    if provider.lower() == "openai":
        return OpenAIGenerator(
            model_name=kwargs.get("model_name", "gpt-4.1-mini"),
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 2048),
            streaming=kwargs.get("streaming", True),
        )
    raise ValueError(f"Provider '{provider}' không được hỗ trợ trong dự án này. Chọn 'openai'.")

def parse_sales_json(raw_text: str) -> tuple[str, str, str | None]:
    """Parse JSON trả về từ model để bóc tách answer, intent, và product_id."""
    clean = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if match:
        clean = match.group(1).strip()
    try:
        data = json.loads(clean)
        return data.get("answer", clean), data.get("intent", "chat"), data.get("product_id", None)
    except Exception:
        intent = "buy" if re.search(r'"intent"\s*:\s*"buy"', clean, re.I) else "chat"
        prod_match = re.search(r'"product_id"\s*:\s*"([^"]+)"', clean, re.I)
        product_id = prod_match.group(1) if prod_match else None
        ans_match = re.search(r'"answer"\s*:\s*"(.*?)(?:"\s*,\s*"intent")', clean, re.DOTALL)
        if ans_match:
            try:
                answer = json.loads(f'"{ans_match.group(1)}"')
            except Exception:
                answer = ans_match.group(1)
        else:
            answer = clean
        return answer, intent, product_id

def stream_sales_answer(raw_stream: Iterator[str]) -> tuple[Iterator[str], list[str]]:
    """Tách stream raw từ JSON thành luồng token text thuần của answer."""
    raw_collector = []

    def generator() -> Iterator[str]:
        buffer = ""
        in_answer = False
        finished_answer = False
        escape = False

        for chunk in raw_stream:
            raw_collector.append(chunk)
            if finished_answer:
                continue

            buffer += chunk

            if not in_answer:
                match = re.search(r'"answer"\s*:\s*"', buffer)
                if match:
                    in_answer = True
                    buffer = buffer[match.end():]
                elif not buffer.startswith("{") and len(buffer) > 20:
                    yield buffer
                    buffer = ""
                    in_answer = True

            if in_answer and not finished_answer:
                i = 0
                out_chars = []
                while i < len(buffer):
                    c = buffer[i]
                    if escape:
                        if c == "n":
                            out_chars.append("\n")
                        elif c == "t":
                            out_chars.append("\t")
                        elif c == '"':
                            out_chars.append('"')
                        elif c == "\\":
                            out_chars.append("\\")
                        else:
                            out_chars.append("\\" + c)
                        escape = False
                        i += 1
                    elif c == "\\":
                        escape = True
                        i += 1
                    elif c == '"':
                        finished_answer = True
                        i += 1
                        break
                    else:
                        out_chars.append(c)
                        i += 1

                if out_chars:
                    yield "".join(out_chars)

                if escape:
                    buffer = "\\"
                else:
                    buffer = buffer[i:]

        if buffer and not finished_answer:
            yield buffer

    return generator(), raw_collector
