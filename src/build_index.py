"""
Build and test a simple retrieval index.

This script is a sanity check that:
1. the dataset can be loaded;
2. chunks can be created;
3. embeddings can be generated;
4. retrieval returns plausible passages.
"""

from config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, CHUNK_LEVEL
from data_loader import load_all_plays
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever


def main() -> None:
    records = load_all_plays(level=CHUNK_LEVEL)
    chunks = create_chunks(records)

    print(f"Loaded {len(records)} records ({CHUNK_LEVEL}-level).")
    print(f"Created {len(chunks)} retrieval chunks.")

    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    retriever.build_index(chunks)

    test_queries = [
        "Why does Macbeth kill Duncan?",
        "Who is Hamlet?",
        "What is the conflict between the Montagues and the Capulets?",
    ]

    for query in test_queries:
        results = retriever.retrieve(query, top_k=DEFAULT_TOP_K)
        print("\n" + "=" * 80)
        print(f"Query: {query}\n")
        for rank, (chunk, score) in enumerate(results, start=1):
            print(f"  Rank {rank} | Score: {score:.4f} | {chunk.get('play')}, "
                  f"Act {chunk.get('act')}, Scene {chunk.get('scene')}")
            print(f"    Summary: {chunk.get('scene_summary', 'N/A')}")


if __name__ == "__main__":
    main()
