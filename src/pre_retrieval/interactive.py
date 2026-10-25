"""CLI query và interactive chat interface."""

from __future__ import annotations
import sys
from typing import Any

def ask_single_question(rag: Any, query: str):
    print(f"\nQ: {query}")
    print("A: ", end="", flush=True)

    stream, res = rag.stream(query)
    for token in stream:
        print(token, end="", flush=True)
    print("\n" + "-" * 40)
    print(f"[debug] intent={res.intent} | product_id={res.product_id} | docs={len(res.final_docs)}")

def interactive_chat(rag: Any):
    print("--- RAG Chat (gõ 'exit' để thoát, 'clear' để reset session) ---")
    session_id = "cli_chat_session"

    while True:
        try:
            query = input("\nBạn: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Tạm biệt!")
                break
            if query.lower() in ("clear", "reset", "cls"):
                rag.memory.clear_session(session_id)
                print("Đã xoá lịch sử chat.")
                continue

            print("\nAI: ", end="", flush=True)
            stream, _ = rag.stream(query, session_id=session_id)
            for token in stream:
                print(token, end="", flush=True)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nTạm biệt!")
            break
