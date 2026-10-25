"""
Sliding window session memory theo session_id.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field

@dataclass
class ChatTurn:
    user_query: str
    ai_answer: str
    timestamp: float = field(default_factory=time.time)

class SessionMemoryManager:
    """Quản lý hội thoại dùng sliding window (mặc định lưu `max_turns` lượt gần nhất)."""
    def __init__(self, max_turns: int = 3, ttl_seconds: int = 3600, answer_max_chars: int = 300):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self.answer_max_chars = answer_max_chars
        self._sessions: dict[str, list[ChatTurn]] = {}
        self._last_accessed: dict[str, float] = {}

    def _cleanup_expired(self):
        now = time.time()
        expired = [
            sid for sid, last in self._last_accessed.items()
            if now - last > self.ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_accessed.pop(sid, None)

    def get_history_messages(self, session_id: str | None) -> list[dict[str, str]]:
        if not session_id:
            return []

        self._cleanup_expired()
        turns = self._sessions.get(session_id, [])
        self._last_accessed[session_id] = time.time()

        messages: list[dict[str, str]] = []
        for turn in turns:
            messages.append({"role": "user", "content": turn.user_query})
            messages.append({"role": "assistant", "content": turn.ai_answer})

        return messages

    def get_history_text(self, session_id: str | None) -> str:
        if not session_id:
            return ""

        self._cleanup_expired()
        turns = self._sessions.get(session_id, [])
        if not turns:
            return ""
        self._last_accessed[session_id] = time.time()

        lines: list[str] = []
        for turn in turns:
            lines.append(f"Người dùng: {turn.user_query}")
            # Truncate context tránh tràn token
            answer = turn.ai_answer.strip().replace("\n", " ")
            if len(answer) > self.answer_max_chars:
                answer = answer[:self.answer_max_chars].rstrip() + "..."
            lines.append(f"AI: {answer}")

        return "\n".join(lines)

    def add_turn(self, session_id: str | None, user_query: str, ai_answer: str):
        if not session_id or not user_query.strip():
            return

        self._cleanup_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append(ChatTurn(user_query=user_query, ai_answer=ai_answer))
        self._last_accessed[session_id] = time.time()

        if len(self._sessions[session_id]) > self.max_turns:
            self._sessions[session_id] = self._sessions[session_id][-self.max_turns:]

    def clear_session(self, session_id: str) -> bool:
        found = session_id in self._sessions
        self._sessions.pop(session_id, None)
        self._last_accessed.pop(session_id, None)
        return found
