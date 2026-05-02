"""
Chunking utilities.

Implements a scene-level chunking strategy with summary enhancement.
Scene-level chunks preserve enough context for meaningful retrieval while
keeping chunk sizes manageable. Each chunk is enriched with the scene summary
and keywords to improve embedding quality and retrieval relevance.
"""

from __future__ import annotations

from typing import Any, Dict, List


Record = Dict[str, Any]
Chunk = Dict[str, Any]

MAX_CHUNK_CHARS = 4000


def _build_scene_chunk_text(record: Record) -> str:
    """
    Build retrieval text for a scene-level record.
    Prepends scene summary and keywords to the full scene text
    so that the embedding captures both semantic overview and details.
    """
    parts = []

    summary = record.get("scene_summary", "")
    if summary:
        parts.append(f"Summary: {summary}")

    keywords = record.get("keywords", [])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    location = record.get("location", "")
    if location:
        parts.append(f"Location: {location}")

    text = record.get("text", "")
    if text:
        parts.append(text.strip())

    return "\n".join(parts)


def create_chunks(records: List[Record]) -> List[Chunk]:
    """
    Convert scene-level records into retrieval chunks.

    Strategy: scene-level with summary enhancement.
    - Each scene becomes one chunk.
    - Summary, keywords, and location are prepended to improve retrieval.
    - Very long scenes are split into sub-chunks with overlap.
    """
    chunks: List[Chunk] = []

    for i, record in enumerate(records):
        full_text = _build_scene_chunk_text(record)
        if not full_text.strip():
            continue

        base_meta = {
            "play": record.get("play", record.get("play_key", "unknown")),
            "act": record.get("act"),
            "scene": record.get("scene"),
            "location": record.get("location", ""),
            "scene_summary": record.get("scene_summary", ""),
            "keywords": record.get("keywords", []),
        }
        chunk_id = record.get("source_id") or f"chunk_{i:06d}"

        if len(full_text) <= MAX_CHUNK_CHARS:
            chunks.append({
                "chunk_id": chunk_id,
                **base_meta,
                "text": full_text,
            })
        else:
            sub_chunks = _split_long_text(full_text, MAX_CHUNK_CHARS, overlap=400)
            for j, sub_text in enumerate(sub_chunks):
                chunks.append({
                    "chunk_id": f"{chunk_id}_part{j}",
                    **base_meta,
                    "text": sub_text,
                })

    return chunks


def _split_long_text(text: str, max_chars: int, overlap: int = 400) -> List[str]:
    """
    Split long text into overlapping sub-chunks at sentence boundaries.
    """
    sentences = text.replace("\n", " \n ").split(". ")
    sub_chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent) + 2
        if current_len + sent_len > max_chars and current:
            sub_chunks.append(". ".join(current) + ".")
            # Keep last few sentences for overlap
            overlap_chars = 0
            overlap_start = len(current)
            for k in range(len(current) - 1, -1, -1):
                overlap_chars += len(current[k]) + 2
                if overlap_chars >= overlap:
                    overlap_start = k
                    break
            current = current[overlap_start:]
            current_len = sum(len(s) + 2 for s in current)
        current.append(sent)
        current_len += sent_len

    if current:
        sub_chunks.append(". ".join(current))

    return sub_chunks


def format_chunk_for_display(chunk: Chunk) -> str:
    """
    Format a retrieved chunk for display to the user.
    """
    play = chunk.get("play", "Unknown play")
    act = chunk.get("act", "?")
    scene = chunk.get("scene", "?")
    location = chunk.get("location", "")

    header = f"{play}, Act {act}, Scene {scene}"
    if location:
        header += f" ({location})"

    summary = chunk.get("scene_summary", "")
    display = f"[{header}]"
    if summary:
        display += f"\nSummary: {summary}"

    text = chunk.get("text", "")
    # Show first 500 chars for display, full text is used for retrieval
    if len(text) > 500:
        display += f"\n{text[:500]}..."
    else:
        display += f"\n{text}"

    return display
