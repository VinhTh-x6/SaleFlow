"""Query Rewriting: Chuẩn hoá và làm rõ câu hỏi trước retrieval."""

from __future__ import annotations
from .base import BaseTransformer, TransformResult
from ..llm.llm_client import call_llm

class QueryRewriter(BaseTransformer):
    """Viết lại câu hỏi để tối ưu cho semantic search."""
    _PROMPT_EN = (
        "Rewrite the following search query to make it concise, unambiguous, "
        "and optimal for document retrieval. Fix typos, expand abbreviations, "
        "and clarify pronouns if any.\n\n"
        "Original query: {query}\n\n"
        "Rewritten query (output ONLY the rewritten query, nothing else):"
    )

    _PROMPT_VI = (
        "Hãy viết lại câu truy vấn sau để tối ưu hóa việc tìm kiếm thông tin trong tài liệu. "
        "Sửa lỗi chính tả, làm rõ đại từ mơ hồ nếu có, giữ câu ngắn gọn và giàu từ khóa.\n\n"
        "Câu truy vấn gốc: {query}\n\n"
        "Câu truy vấn viết lại (chỉ xuất nội dung câu viết lại, không thêm lời dẫn giải):"
    )

    _PROMPT_WITH_HISTORY_VI = (
        "Dưới đây là lịch sử cuộc trò chuyện gần đây:\n"
        "{history}\n\n"
        "Câu hỏi mới nhất của người dùng: {query}\n\n"
        "Nhiệm vụ: Hãy viết lại câu hỏi mới nhất thành một câu truy vấn tìm kiếm độc lập và đầy đủ ý nghĩa. "
        "Giải quyết các đại từ mơ hồ (như 'nó', 'trường hợp đó', 'cái này', 'ai'...) dựa vào ngữ cảnh đã trò chuyện. "
        "Nếu câu hỏi đã rõ ràng và độc lập, hãy giữ nguyên nội dung.\n"
        "Câu truy vấn viết lại (chỉ xuất câu viết lại, không giải thích):"
    )

    _PROMPT_WITH_HISTORY_EN = (
        "Given the recent conversation history:\n"
        "{history}\n\n"
        "Latest user query: {query}\n\n"
        "Task: Rewrite the latest query into a standalone, fully self-contained search query. "
        "Resolve ambiguous references and pronouns (e.g., 'it', 'they', 'that case', 'which one') using the chat context. "
        "If the query is already self-contained, keep it as is.\n"
        "Rewritten query (output ONLY the rewritten query, nothing else):"
    )

    def transform(self, query: str, history_text: str = "") -> TransformResult:
        if history_text.strip():
            tmpl = self._PROMPT_WITH_HISTORY_VI if self.language == "vi" else self._PROMPT_WITH_HISTORY_EN
            prompt_content = tmpl.format(history=history_text.strip(), query=query)
        else:
            tmpl = self._PROMPT_VI if self.language == "vi" else self._PROMPT_EN
            prompt_content = tmpl.format(query=query)

        try:
            rewritten = call_llm(
                prompt_content,
                provider=self.llm_provider,
                model=self.llm_model,
                max_tokens=128,
            ).strip().strip('"').strip("'")
            return TransformResult(
                original_query=query,
                queries=[rewritten] if rewritten else [query],
            )
        except Exception:
            return TransformResult(original_query=query, queries=[query])
