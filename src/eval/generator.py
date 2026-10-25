"""
Tự động sinh synthetic testset từ tài liệu PDF bằng Ragas TestsetGenerator.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

from ..loader.loader import DocumentLoader

def generate_synthetic_testset(
    data_dir: str = "./data/raw",
    test_size: int = 5,
    output_path: str = "data/eval/eval_dataset.json",
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
):
    print(f"--- Sinh Testset với Ragas (size={test_size}, model={llm_model}) ---")

    if not os.environ.get("OPENAI_API_KEY"):
        print("Lỗi: Thiếu OPENAI_API_KEY trong .env")
        return

    # Fallback cho data_dir
    p_data = Path(data_dir)
    if not p_data.exists():
        if Path("./data/raw").exists():
            data_dir = "./data/raw"
        elif Path("./data").exists():
            data_dir = "./data"

    loader = DocumentLoader(pdf_strategy="docling")
    docs = loader.load_directory(data_dir)

    if not docs:
        print(f"Không tìm thấy tài liệu trong '{data_dir}'")
        return

    print(f"Đã nạp {len(docs)} documents từ {data_dir}")

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.testset import TestsetGenerator

    generator_llm = ChatOpenAI(model=llm_model, temperature=0.7)
    generator_embeddings = OpenAIEmbeddings(model=embedding_model)

    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        llm_context=(
            "Tất cả các câu hỏi (user_input/question) và câu trả lời mẫu (reference/ground_truth) "
            "BẮT BUỘC phải được tạo 100% bằng TIẾNG VIỆT tự nhiên, phù hợp với câu hỏi của khách hàng "
            "hỏi về sản phẩm, tài liệu kỹ thuật tiếng Việt. Tuyệt đối KHÔNG sinh câu hỏi bằng tiếng Anh."
        ),
    )

    print(f"Đang sinh {test_size} câu hỏi và đáp án mẫu...")
    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=test_size,
    )

    df = testset.to_pandas()
    print(f"Đã tạo thành công {len(df)} mẫu kiểm thử.")

    samples = []
    for _, row in df.iterrows():
        question = str(row.get("user_input") or row.get("question") or "").strip()
        ground_truth = str(row.get("reference") or row.get("ground_truth") or "").strip()
        contexts = row.get("reference_contexts") or []

        if question and ground_truth:
            samples.append({
                "question": question,
                "ground_truth": ground_truth,
                "reference_contexts": contexts if isinstance(contexts, list) else [str(contexts)],
            })

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu dataset JSON: {out_p.resolve()}")

    csv_out = out_p.with_suffix(".csv")
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"Đã lưu bản sao CSV: {csv_out.resolve()}")

    print("\n--- Mẫu câu hỏi vừa sinh ---")
    for idx, s in enumerate(samples[:3], start=1):
        print(f"[{idx}] Q: {s['question']}")
        print(f"    GT: {s['ground_truth'][:100]}...\n")

    print(f"Chạy đánh giá bằng lệnh:\n  python main.py eval --dataset {output_path}\n")
