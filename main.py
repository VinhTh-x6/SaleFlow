"""
Unified CLI Entry Point cho RAG Project.
Các lệnh chính:
    python main.py index [--source PATH] [--reindex]
    python main.py query "câu hỏi của bạn"
    python main.py chat
    python main.py api [--host HOST] [--port PORT]
    python main.py eval [--dataset PATH] [--config PATH] [--output PATH]
    python main.py generate-testset [--data-dir PATH] [--size N]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Đảm bảo root và src có trong sys.path
root_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def main():
    parser = argparse.ArgumentParser(description="Standalone RAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Command: index
    idx_parser = subparsers.add_parser("index", help="Lập chỉ mục tài liệu vào ChromaDB")
    idx_parser.add_argument("--source", type=str, default=None, help="Đường dẫn file hoặc thư mục nguồn (mặc định ./data/raw)")
    idx_parser.add_argument("--config", type=str, default="config.yaml", help="File cấu hình")
    idx_parser.add_argument("--reindex", action="store_true", help="Xoá và index lại từ đầu")

    # Command: query
    q_parser = subparsers.add_parser("query", help="Đặt câu hỏi đơn lẻ")
    q_parser.add_argument("question", type=str, help="Câu hỏi tra cứu")
    q_parser.add_argument("--config", type=str, default="config.yaml", help="File cấu hình")

    # Command: chat
    chat_parser = subparsers.add_parser("chat", help="Chat tương tác qua dòng lệnh")
    chat_parser.add_argument("--config", type=str, default="config.yaml", help="File cấu hình")

    # Command: api
    api_parser = subparsers.add_parser("api", help="Khởi chạy REST API server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (mặc định: 0.0.0.0)")
    api_parser.add_argument("--port", type=int, default=8000, help="Port (mặc định: 8000)")

    # Command: eval
    eval_parser = subparsers.add_parser("eval", help="Đánh giá RAG bằng Ragas")
    eval_parser.add_argument("--dataset", type=str, default="data/eval/eval_dataset.json", help="File dataset (.json / .csv)")
    eval_parser.add_argument("--config", type=str, default="config.yaml", help="File cấu hình")
    eval_parser.add_argument("--output", type=str, default="data/eval/eval_results.csv", help="File xuất kết quả")

    # Command: generate-testset
    gen_parser = subparsers.add_parser("generate-testset", help="Sinh testset câu hỏi tự động qua Ragas")
    gen_parser.add_argument("--data-dir", type=str, default="./data/raw", help="Thư mục tài liệu (mặc định: ./data/raw)")
    gen_parser.add_argument("--size", type=int, default=5, help="Số lượng câu hỏi")
    gen_parser.add_argument("--output", type=str, default="data/eval/eval_dataset.json", help="File JSON kết quả")
    gen_parser.add_argument("--llm", type=str, default="gpt-4o-mini", help="Model LLM sinh câu hỏi")

    args = parser.parse_args()

    if args.command == "index":
        from src.pipeline.index_pipeline import IndexingPipeline
        pipe = IndexingPipeline(config_path=args.config)
        res = pipe.run(source_path=args.source, force_reindex=args.reindex)
        sys.exit(0 if res.success else 1)

    elif args.command == "query":
        from src.pipeline.query_pipeline import RAGPipeline
        from src.pre_retrieval.interactive import ask_single_question
        rag = RAGPipeline(config_path=args.config)
        ask_single_question(rag, args.question)

    elif args.command == "chat":
        from src.pipeline.query_pipeline import RAGPipeline
        from src.pre_retrieval.interactive import interactive_chat
        rag = RAGPipeline(config_path=args.config)
        interactive_chat(rag)

    elif args.command == "api":
        import uvicorn
        print(f"\n Khởi chạy REST API tại http://{args.host}:{args.port}")
        print(f" Tài liệu tương tác Swagger UI: http://localhost:{args.port}/docs\n")
        uvicorn.run("src.api.routes:app", host=args.host, port=args.port, reload=False)

    elif args.command == "eval":
        from src.eval.rag_metrics import run_evaluation
        run_evaluation(
            dataset_path=args.dataset,
            config_path=args.config,
            output_path=args.output,
        )

    elif args.command == "generate-testset":
        from src.eval.generator import generate_synthetic_testset
        generate_synthetic_testset(
            data_dir=args.data_dir,
            test_size=args.size,
            output_path=args.output,
            llm_model=args.llm,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
