"""
Online RAG Pipeline.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import yaml
from langchain_core.documents import Document

from ..embedding.embedder import get_embedder
from ..vector_db.chroma_store import ChromaVectorStore
from ..pre_retrieval.pipeline import QueryPipeline
from ..retrieval.retriever import get_retriever
from ..post_retrieval.pipeline import RerankPipeline
from ..prompt.prompt_templates import get_prompt_builder
from ..prompt.base import PromptResult
from ..generation.grounded_answer import get_generator, parse_sales_json, stream_sales_answer
from ..generation.base import GenerationResult
from ..utils.memory import SessionMemoryManager

logger = logging.getLogger(__name__)

@dataclass
class RAGResult:
    """Kết quả trả về của pipeline RAG."""
    answer: str
    query_used: str
    retrieved_docs: list[Document]
    final_docs: list[Document]
    intent: str = "chat"
    product_id: str | None = None
    cited_sources: list[int] = field(default_factory=list)
    generation_result: GenerationResult | None = None
    prompt_result: PromptResult | None = None

class RAGPipeline:
    """RAG Online Execution Pipeline."""
    def __init__(self, config_path: str = "config.yaml"):
        p_cfg = Path(config_path)
        if not p_cfg.is_absolute():
            proj_root = Path(__file__).resolve().parent.parent.parent
            if (proj_root / config_path).exists():
                p_cfg = proj_root / config_path
        self.config_path = str(p_cfg)
        self.config = self._load_config(self.config_path)

        cfg = self.config
        idx_cfg = cfg.get("indexing", {})
        q_cfg = cfg.get("query_pipeline", {})

        # 1. Embedder
        emb_cfg = idx_cfg.get("embedding", {})
        self.embedder = get_embedder(
            provider=emb_cfg.get("provider", "openai"),
            model_name=emb_cfg.get("model_name", "text-embedding-3-small"),
            dimensions=emb_cfg.get("dimensions", None),
        )

        # 2. Vector DB (Chroma)
        vdb_cfg = idx_cfg.get("vector_db", {})
        raw_persist = vdb_cfg.get("persist_dir", "./storage")
        if not Path(raw_persist).is_absolute():
            self.persist_dir = str(Path(self.config_path).parent / raw_persist)
        else:
            self.persist_dir = raw_persist
        self.collection_name = vdb_cfg.get("collection_name", "rag_collection")

        # Chunks cache cho hybrid BM25
        self.indexed_chunks = self._load_indexed_chunks()

        self.vector_store_builder = ChromaVectorStore(
            collection_name=self.collection_name,
            persist_dir=self.persist_dir,
            force_reindex=False,
        )
        self.vector_store = self.vector_store_builder.get_or_create(
            self.indexed_chunks, self.embedder
        )

        pre_cfg = q_cfg.get("pre_retrieval", {})
        self.pre_pipeline = QueryPipeline(
            transformations=pre_cfg.get("transformations", ["rewrite"]),
            llm_model=pre_cfg.get("transformation_llm", "gpt-4.1-mini"),
            language=cfg.get("data", {}).get("language", "both"),
        )

        ret_cfg = q_cfg.get("retrieval", {})
        self.retriever = get_retriever(
            strategy=ret_cfg.get("strategy", "hybrid"),
            vector_store=self.vector_store,
            documents=self.indexed_chunks,
            top_k=ret_cfg.get("top_k", 15),
            rrf_k=ret_cfg.get("rrf_k", 60),
            search_score_threshold=vdb_cfg.get("search_score_threshold", 0.0),
        )

        post_cfg = q_cfg.get("post_retrieval", {})
        self.post_pipeline = RerankPipeline(
            reranker=post_cfg.get("reranker", "cross_encoder"),
            cross_encoder_model=post_cfg.get("cross_encoder_model", "BAAI/bge-reranker-v2-m3"),
            top_n=post_cfg.get("top_n", 5),
            apply_redundancy=post_cfg.get("apply_redundancy", True),
            redundancy_threshold=post_cfg.get("redundancy_threshold", 0.92),
            context_ordering=post_cfg.get("context_ordering", "sandwich"),
        )

        prompt_cfg = q_cfg.get("prompt", {})
        self.prompt_builder = get_prompt_builder(
            template=prompt_cfg.get("template", "safe_rag"),
            max_context_chars=prompt_cfg.get("max_context_chars", 0),
        )

        gen_cfg = q_cfg.get("generation", {})
        self.generator = get_generator(
            provider=gen_cfg.get("provider", "openai"),
            model_name=gen_cfg.get("model_name", "gpt-4.1-mini"),
            temperature=gen_cfg.get("temperature", 0.0),
            max_tokens=gen_cfg.get("max_tokens", 2048),
            streaming=gen_cfg.get("streaming", True),
        )

        self.memory = SessionMemoryManager(max_turns=3)

    @staticmethod
    def _load_config(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_indexed_chunks(self) -> list[Document]:
        cache_file = Path(self.persist_dir) / "indexed_chunks.json"
        if not cache_file.exists():
            return []
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [
                Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
                for item in data
            ]
        except Exception as exc:
            logger.warning("Không thể đọc indexed_chunks.json (%s)", exc)
            return []

    def prepare_prompt(
        self, query: str, session_id: str | None = None
    ) -> tuple[PromptResult, str, list[Document], list[Document]]:
        """Chạy Pre-retrieval (kèm context memory) -> Retrieval -> Post-retrieval -> Build Prompt."""
        history_msgs = self.memory.get_history_messages(session_id) if session_id else []
        history_text = self.memory.get_history_text(session_id) if session_id else ""

        # 1. Pre-retrieval
        transform_res = self.pre_pipeline.transform(query, history_text=history_text)
        effective_query = transform_res.queries[0] if transform_res.queries else query

        # 2. Retrieval
        retrieved_docs = self.retriever.retrieve(transform_res)

        # 3. Post-retrieval
        final_docs = self.post_pipeline.process(query=effective_query, docs=retrieved_docs)

        # 4. Prompt Construction
        prompt_result = self.prompt_builder.build(
            query=query, docs=final_docs, history=history_msgs
        )

        return prompt_result, effective_query, retrieved_docs, final_docs

    def run(self, query: str, session_id: str | None = None) -> RAGResult:
        """Thực hiện toàn bộ quy trình RAG và lưu câu trả lời vào memory nếu có session_id."""
        prompt_res, eff_query, ret_docs, fin_docs = self.prepare_prompt(query, session_id=session_id)
        gen_res = self.generator.generate(prompt_res)

        # Parse JSON output để bóc tách answer, intent, product_id
        answer, intent, product_id = parse_sales_json(gen_res.answer)

        if session_id:
            self.memory.add_turn(session_id, user_query=query, ai_answer=answer)

        return RAGResult(
            answer=answer,
            query_used=eff_query,
            retrieved_docs=ret_docs,
            final_docs=fin_docs,
            intent=intent,
            product_id=product_id,
            cited_sources=gen_res.cited_sources,
            generation_result=gen_res,
            prompt_result=prompt_res,
        )

    def stream(self, query: str, session_id: str | None = None) -> tuple[Iterator[str], RAGResult]:
        """Thực hiện streaming câu trả lời và tự động lưu vào session memory khi hoàn tất."""
        prompt_res, eff_query, ret_docs, fin_docs = self.prepare_prompt(query, session_id=session_id)
        raw_stream = self.generator.stream(prompt_res)

        clean_stream, raw_acc = stream_sales_answer(raw_stream)

        placeholder_result = RAGResult(
            answer="",
            query_used=eff_query,
            retrieved_docs=ret_docs,
            final_docs=fin_docs,
            prompt_result=prompt_res,
        )

        def wrapped_stream() -> Iterator[str]:
            full_answer_parts = []
            for token in clean_stream:
                full_answer_parts.append(token)
                yield token

            full_raw = "".join(raw_acc)
            parsed_answer, intent, product_id = parse_sales_json(full_raw)
            final_answer = "".join(full_answer_parts) or parsed_answer

            placeholder_result.answer = final_answer
            placeholder_result.intent = intent
            placeholder_result.product_id = product_id

            if session_id:
                self.memory.add_turn(session_id, user_query=query, ai_answer=final_answer)

        return wrapped_stream(), placeholder_result
