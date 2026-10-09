from __future__ import annotations

from app.rag.vector_store import RetrievedChunk


def _word_set(text: str) -> frozenset[str]:
    return frozenset(text.lower().split())


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def select_context(
    ranked: list[RetrievedChunk],
    final_k: int,
    min_relevance: float,
    duplicate_threshold: float,
    max_chars: int,
) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    seen: list[frozenset[str]] = []
    used_chars = 0

    for chunk in ranked:
        score = chunk.rerank_score if chunk.rerank_score is not None else chunk.score
        if score < min_relevance:
            continue
        words = _word_set(chunk.text)
        if any(jaccard(words, prev) >= duplicate_threshold for prev in seen):
            continue
        if selected and used_chars + len(chunk.text) > max_chars:
            continue  # may still fit a shorter, lower-ranked chunk
        selected.append(chunk)
        seen.append(words)
        used_chars += len(chunk.text)
        if len(selected) >= final_k:
            break
    return selected
