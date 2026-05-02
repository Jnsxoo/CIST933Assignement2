"""
Baseline system: prompt-only generation without retrieval.

This baseline sends the user question directly to the same language model
used by the RAG system, but WITHOUT any retrieved context. This isolates
the effect of retrieval-augmented generation and provides a fair comparison.
"""

from __future__ import annotations

from model import generate


def baseline_answer(query: str) -> str:
    """
    Generate an answer using only the language model's parametric knowledge
    (no retrieved context). This is the prompt-only baseline.
    """
    prompt = (
        "You are a helpful assistant that answers questions about Shakespeare's plays. "
        "Answer in a beginner-friendly way. If you are unsure, say so.\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    return generate(prompt)


if __name__ == "__main__":
    question = "Who is Hamlet?"
    print("Question:", question)
    print("Answer:", baseline_answer(question))
