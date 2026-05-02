"""
RAG chatbot with local HuggingFace language model.

Pipeline:
1. Load Shakespeare dataset and create scene-level chunks.
2. Build embedding index with sentence-transformers.
3. For each user query: retrieve top-k chunks, build RAG prompt, generate answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import re

from config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    PROMPT_DIR,
    CHUNK_LEVEL,
)
from data_loader import load_all_plays
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever
from model import generate


Chunk = Dict[str, Any]


def load_system_prompt() -> str:
    prompt_path = PROMPT_DIR / "system_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def build_rag_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    Build a prompt for a RAG-based answer.
    """
    system_prompt = load_system_prompt()

    context_blocks = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        play = chunk.get("play", "Unknown")
        act = chunk.get("act", "?")
        scene = chunk.get("scene", "?")
        summary = chunk.get("scene_summary", "")
        text = chunk.get("text", "")
        # Truncate text to fit in context window
        if len(text) > 1500:
            text = text[:1500] + "..."
        block = f"[Source {rank}: {play}, Act {act}, Scene {scene} | relevance={score:.3f}]"
        if summary:
            block += f"\nSummary: {summary}"
        block += f"\n{text}"
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    return f"{system_prompt}\n\nRetrieved context:\n{context}\n\nUser question:\n{query}\n\nAnswer:"


STYLISED_KEYWORDS = [
    "shakespearean", "shakespeare style", "in the style of",
    "write as", "respond as", "speak as", "stylised", "stylized",
    "like shakespeare", "old english", "poetic response",
    "generate a short shakespearean", "generate a shakespearean",
]


def is_stylised_query(query: str) -> bool:
    """Detect if the user is requesting a Shakespearean-style creative response."""
    q = query.lower()
    return any(kw in q for kw in STYLISED_KEYWORDS)


def build_stylised_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """Build a prompt for stylised Shakespearean generation (≤150 words)."""
    context_blocks = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        play = chunk.get("play", "Unknown")
        act = chunk.get("act", "?")
        scene = chunk.get("scene", "?")
        text = chunk.get("text", "")
        if len(text) > 800:
            text = text[:800] + "..."
        context_blocks.append(f"[{play}, Act {act}, Scene {scene}]\n{text}")

    context = "\n\n".join(context_blocks)

    return (
        "You are a creative writing assistant who writes in Shakespearean style.\n"
        "Based on the context below, write a SHORT response (no more than 150 words) "
        "in the style of Shakespeare. Use Early Modern English vocabulary and poetic rhythm.\n"
        "IMPORTANT: This is CREATIVE output, NOT factual evidence. "
        "Make this clear in your response.\n\n"
        f"Context:\n{context}\n\n"
        f"Request: {query}\n\n"
        "Shakespearean-style response (≤150 words):"
    )


def generate_answer(prompt: str) -> str:
    """
    Generate an answer using the shared local language model
    conditioned on the RAG prompt with retrieved context.
    """
    return generate(prompt)


def build_pipeline():
    """
    Build and return the full RAG pipeline components.
    Returns (retriever, chunks) for reuse in evaluation.
    """
    records = load_all_plays(level=CHUNK_LEVEL)
    chunks = create_chunks(records)
    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    retriever.build_index(chunks)
    return retriever, chunks


def rag_answer(query: str, retriever: EmbeddingRetriever, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """
    Full RAG answer pipeline. Returns dict with answer, retrieved chunks, and prompt.
    Auto-detects if query requests stylised generation.
    """
    retrieved = retriever.retrieve(query, top_k=top_k)
    stylised = is_stylised_query(query)

    if stylised:
        prompt = build_stylised_prompt(query, retrieved)
    else:
        prompt = build_rag_prompt(query, retrieved)

    answer = generate_answer(prompt)

    if stylised:
        answer = (
            "[NOTE: The following is creative Shakespearean-style output, "
            "not factual evidence.]\n\n" + answer
        )

    return {
        "query": query,
        "answer": answer,
        "retrieved": retrieved,
        "prompt": prompt,
        "stylised": stylised,
    }


def main() -> None:
    retriever, chunks = build_pipeline()

    print(f"Shakespeare-aware RAG chatbot ready. ({len(chunks)} chunks indexed)")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in {"quit", "exit"}:
            break

        result = rag_answer(query, retriever)

        print("\nRetrieved evidence:")
        for rank, (chunk, score) in enumerate(result["retrieved"], start=1):
            print("-" * 80)
            print(f"Rank {rank} | Score: {score:.4f}")
            print(format_chunk_for_display(chunk))

        print("\nGenerated answer:")
        print(result["answer"])
        print("\n")


if __name__ == "__main__":
    main()
