"""
Evaluation script.

Runs both baseline and RAG systems on all evaluation questions,
records the outputs and retrieved evidence to a CSV for scoring.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from config import RESULTS_DIR, DEFAULT_TOP_K


QUESTIONS_PATH = RESULTS_DIR / "instructor_questions.json"
OUTPUT_PATH = RESULTS_DIR / "evaluation_results.csv"


def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list format and dict-with-key format
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["questions", "data"]:
            if key in data:
                return data[key]
    return data


def run_evaluation() -> None:
    """
    Run baseline and RAG systems on all evaluation questions and save results.
    """
    from baseline import baseline_answer
    from rag_chatbot import build_pipeline, rag_answer
    from chunking import format_chunk_for_display

    questions = load_questions()
    print(f"Loaded {len(questions)} evaluation questions.")

    # Build RAG pipeline once
    print("Building RAG pipeline...")
    retriever, chunks = build_pipeline()
    print(f"Pipeline ready. {len(chunks)} chunks indexed.\n")

    fieldnames = [
        "question_id",
        "question",
        "question_type",
        "question_source",
        "expected_focus",
        "system",
        "is_stylised",
        "retrieved_passages",
        "generated_response",
        "correctness_score",
        "grounding_score",
        "retrieval_relevance_score",
        "usefulness_score",
        "style_quality_score",
        "comments",
    ]

    rows: List[Dict[str, Any]] = []

    for i, q in enumerate(questions):
        qid = q.get("question_id") or q.get("id") or f"Q{i+1}"
        question = q.get("question", "")
        qtype = q.get("type") or q.get("question_type", "")
        qsource = q.get("source", "unknown")
        expected = q.get("expected_focus", "")

        print(f"[{qid}] {question}")

        # --- Baseline ---
        print(f"  Running baseline...")
        try:
            baseline_resp = baseline_answer(question)
        except Exception as e:
            baseline_resp = f"[ERROR] {e}"

        rows.append({
            "question_id": qid,
            "question": question,
            "question_type": qtype,
            "question_source": qsource,
            "expected_focus": expected,
            "system": "baseline",
            "is_stylised": "",
            "retrieved_passages": "N/A (no retrieval)",
            "generated_response": baseline_resp,
            "correctness_score": "",
            "grounding_score": "",
            "retrieval_relevance_score": "",
            "usefulness_score": "",
            "style_quality_score": "",
            "comments": "",
        })
        print(f"  Baseline done.")

        # --- RAG ---
        print(f"  Running RAG...")
        is_stylised = False
        try:
            result = rag_answer(question, retriever, top_k=DEFAULT_TOP_K)
            rag_resp = result["answer"]
            is_stylised = result.get("stylised", False)
            retrieved_str = "\n---\n".join(
                f"[Rank {r+1}, score={s:.3f}] {format_chunk_for_display(c)[:300]}"
                for r, (c, s) in enumerate(result["retrieved"])
            )
        except Exception as e:
            rag_resp = f"[ERROR] {e}"
            retrieved_str = f"[ERROR] {e}"

        rows.append({
            "question_id": qid,
            "question": question,
            "question_type": qtype,
            "question_source": qsource,
            "expected_focus": expected,
            "system": "rag",
            "is_stylised": is_stylised,
            "retrieved_passages": retrieved_str,
            "generated_response": rag_resp,
            "correctness_score": "",
            "grounding_score": "",
            "retrieval_relevance_score": "",
            "usefulness_score": "",
            "style_quality_score": "",
            "comments": "",
        })
        print(f"  RAG done.\n")

    # Write results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluation results saved to: {OUTPUT_PATH}")
    print(f"Total rows: {len(rows)} ({len(questions)} questions x 2 systems)")

    # Summary by type and source
    from collections import Counter
    type_counts = Counter(q.get("type", "unknown") for q in questions)
    source_counts = Counter(q.get("source", "unknown") for q in questions)
    print(f"\nQuestion types: {dict(type_counts)}")
    print(f"Question sources: {dict(source_counts)}")
    print("\nFill in the score columns (1-5) manually to complete evaluation.")


if __name__ == "__main__":
    run_evaluation()
