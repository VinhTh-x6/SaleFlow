"""LLM caller client with retries and model support."""

from __future__ import annotations
from openai import OpenAI

_OPENAI_NEW_API_PREFIXES = ("gpt-5", "o1", "o3", "o4")

def call_llm(
    prompt: str,
    provider: str = "openai",
    model: str = "gpt-4.1-mini",
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Gọi LLM và trả về phản hồi dạng text."""
    if provider.lower() == "openai":
        client = OpenAI()
        is_new_api = any(model.startswith(p) for p in _OPENAI_NEW_API_PREFIXES)
        token_param = "max_completion_tokens" if is_new_api else "max_tokens"
        extra = {} if is_new_api else {"temperature": temperature}

        r = client.chat.completions.create(
            model=model,
            **{token_param: max_tokens},
            **extra,
            messages=[{"role": "user", "content": prompt}],
        )
        content = r.choices[0].message.content
        if content is None:
            reasoning = getattr(r.choices[0].message, "reasoning_content", None)
            if reasoning:
                return reasoning.strip()
            return ""
        return content.strip()

    raise ValueError(f"Provider '{provider}' không được hỗ trợ. Chọn 'openai'.")
