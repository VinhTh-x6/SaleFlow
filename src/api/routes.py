"""
FastAPI Server cho RAG Pipeline.
"""

from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..schemas.models import (
    QueryRequest,
    ClearSessionRequest,
    SourceItem,
    QueryResponse,
    IndexRequest,
    IndexResponse,
)
from ..pipeline.index_pipeline import IndexingPipeline
from ..pipeline.query_pipeline import RAGPipeline
from dotenv import load_dotenv

env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file, override=True)
else:
    load_dotenv(override=True)

logger = logging.getLogger("rag_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Khởi tạo RAG Pipeline...")
    app.state.rag = RAGPipeline(config_path="config.yaml")
    logger.info("RAG Pipeline ready.")
    yield

app = FastAPI(
    title="RAG Pipeline API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
async def root():
    return {
        "status": "online",
        "service": "Standalone RAG API",
        "docs_url": "/docs",
    }

@app.get("/health", tags=["General"])
async def health():
    rag: RAGPipeline = app.state.rag
    indexed_count = len(rag.indexed_chunks)
    return {
        "status": "healthy",
        "indexed_chunks": indexed_count,
        "collection_name": rag.collection_name,
    }

@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(req: QueryRequest):
    """
    Truy vấn câu hỏi (Non-streaming).
    Có hỗ trợ `session_id` để AI nhớ ngữ cảnh các câu hỏi trước.
    Trả về câu trả lời cho khách cùng với phân loại `intent` ("chat"/"buy") và `product_id`.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    rag: RAGPipeline = app.state.rag
    try:
        result = await asyncio.to_thread(rag.run, req.query, session_id=req.session_id)

        sources: list[SourceItem] = []
        for i, doc in enumerate(result.final_docs, start=1):
            sources.append(
                SourceItem(
                    index=i,
                    filename=doc.metadata.get("filename", "unknown"),
                    source=str(doc.metadata.get("source", "")),
                    content=doc.page_content,
                )
            )

        return QueryResponse(
            query=req.query,
            query_used=result.query_used,
            session_id=req.session_id,
            answer=result.answer,
            intent=result.intent,
            product_id=result.product_id,
            cited_sources=result.cited_sources,
            sources=sources,
        )
    except Exception as exc:
        logger.exception("Lỗi khi xử lý query: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/query/stream", tags=["RAG"])
async def query_stream_endpoint(req: QueryRequest):
    """
    Truy vấn câu hỏi dạng Streaming (Server-Sent Events).
    Hỗ trợ `session_id` để nhớ ngữ cảnh và tự lưu câu trả lời vào memory khi kết thúc stream.
    Gửi metadata intent và product_id khi hoàn tất luồng token.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    rag: RAGPipeline = app.state.rag

    async def token_generator() -> AsyncIterator[str]:
        try:
            stream, placeholder = await asyncio.to_thread(rag.stream, req.query, session_id=req.session_id)

            for token in stream:
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.005)

            yield f"data: {json.dumps({'done': True, 'intent': placeholder.intent, 'product_id': placeholder.product_id}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

@app.post("/api/session/clear", tags=["Session"])
async def clear_session_endpoint(req: ClearSessionRequest):
    """
    Xoá lịch sử hội thoại của một session để bắt đầu chủ đề mới.
    """
    rag: RAGPipeline = app.state.rag
    cleared = rag.memory.clear_session(req.session_id)
    return {
        "session_id": req.session_id,
        "cleared": cleared,
        "message": "Đã xoá lịch sử trò chuyện của session thành công." if cleared else "Session không tồn tại hoặc đã trống.",
    }

@app.post("/api/index", response_model=IndexResponse, tags=["Indexing"])
async def index_endpoint(req: IndexRequest):
    """
    Thực hiện lập chỉ mục (Indexing) tài liệu mới.
    Sau khi index xong, hệ thống sẽ tự động cập nhật lại pipeline trong bộ nhớ.
    """
    try:
        indexer = IndexingPipeline(config_path="config.yaml")
        res = await asyncio.to_thread(indexer.run, source_path=req.source, force_reindex=req.reindex)

        if not res.success:
            raise HTTPException(status_code=500, detail=f"Indexing thất bại: {res.error}")

        app.state.rag = RAGPipeline(config_path="config.yaml")

        return IndexResponse(
            success=True,
            n_docs=res.n_docs,
            n_chunks=res.n_chunks,
            message="Lập chỉ mục thành công và đã nạp lại vào bộ nhớ API.",
        )
    except Exception as exc:
        logger.exception("Lỗi khi reindex qua API: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
