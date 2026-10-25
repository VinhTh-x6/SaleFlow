"""
Đánh giá chất lượng RAG pipeline bằng Ragas (0.4+).
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

from ..pipeline.query_pipeline import RAGPipeline

def load_testset(file_path: str) -> list[dict]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File dataset không tồn tại: {file_path}")

    samples = []
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict) and "samples" in data:
                samples = data["samples"]
    elif p.suffix.lower() == ".csv":
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(row)
    else:
        raise ValueError("Chỉ hỗ trợ file .json hoặc .csv")

    standardized = []
    for s in samples:
        q = s.get("question") or s.get("user_input") or s.get("query")
        gt = s.get("ground_truth") or s.get("reference") or s.get("answer") or ""
        if q:
            standardized.append({"question": q.strip(), "ground_truth": gt.strip()})

    return standardized

def run_evaluation(
    dataset_path: str = "data/eval/eval_dataset.json",
    config_path: str = "config.yaml",
    output_path: str = "data/eval/eval_results.csv",
):
    print("--- Khởi chạy đánh giá RAG ---")
    rag = RAGPipeline(config_path=config_path)

    # Fallback path nếu dataset_path mặc định không tồn tại ở data/eval
    p_data = Path(dataset_path)
    if not p_data.exists():
        if Path("eval_dataset.json").exists():
            dataset_path = "eval_dataset.json"
        elif Path("data/eval/eval_dataset.json").exists():
            dataset_path = "data/eval/eval_dataset.json"

    test_samples = load_testset(dataset_path)
    print(f"Loaded {len(test_samples)} test samples từ {dataset_path}")

    if not test_samples:
        print("Dataset rỗng, dừng tiến trình.")
        return None

    rag_eval_items = []
    for idx, item in enumerate(test_samples, start=1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"[{idx}/{len(test_samples)}] Query: {q[:60]}...")

        res = rag.run(q)
        rag_eval_items.append({
            "user_input": q,
            "response": res.answer,
            "retrieved_contexts": [doc.page_content for doc in res.final_docs],
            "reference": gt,
            "intent": res.intent,
            "product_id": res.product_id,
        })

    print("\nTính toán metrics qua Ragas...")
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    eval_emb = OpenAIEmbeddings(model="text-embedding-3-small")

    samples_list = [
        SingleTurnSample(
            user_input=it["user_input"],
            response=it["response"],
            retrieved_contexts=it["retrieved_contexts"],
            reference=it["reference"],
        )
        for it in rag_eval_items
    ]

    eval_dataset = EvaluationDataset(samples=samples_list)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_emb,
    )

    df = results.to_pandas()
    df["intent"] = [it["intent"] for it in rag_eval_items]
    df["product_id"] = [it["product_id"] for it in rag_eval_items]

    print("\n" + "=" * 55)
    print("             KẾT QUẢ ĐÁNH GIÁ (RAGAS)             ")
    print("=" * 55)
    if "faithfulness" in df:
        print(f" • Faithfulness:        {df['faithfulness'].mean():.4f}")
    if "answer_relevancy" in df:
        print(f" • Answer Relevancy:   {df['answer_relevancy'].mean():.4f}")
    if "context_precision" in df:
        print(f" • Context Precision:  {df['context_precision'].mean():.4f}")
    if "context_recall" in df:
        print(f" • Context Recall:     {df['context_recall'].mean():.4f}")
    print("=" * 55)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu kết quả CSV: {out_p.resolve()}")

    json_out = out_p.with_suffix(".json")
    df.to_json(json_out, orient="records", force_ascii=False, indent=2)
    print(f"Đã lưu kết quả JSON: {json_out.resolve()}\n")

    return results
