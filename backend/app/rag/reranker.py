from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter

from app.config import Settings
from app.rag.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how i in is it its of on or that the "
    "their this to was were what when where which who why will with you your do does did "
    "can could should would about into than then there these those they them we our not no".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOPWORDS]


class BaseReranker(ABC):
    name: str = "base"

    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return the candidates with `rerank_score` set, best first."""


class NoOpReranker(BaseReranker):
    """RERANKER=none. Keeps the vector order. Useful only to compare quality."""
    name = "none"

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        for c in candidates:
            c.rerank_score = max(0.0, min(1.0, c.score))
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)


class LexicalReranker(BaseReranker):
    name = "lexical"

    def __init__(self, vector_weight: float = 0.5, k1: float = 1.5, b: float = 0.75):
        self.vector_weight = vector_weight
        self.k1 = k1
        self.b = b

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        q_terms = list(dict.fromkeys(content_terms(query)))  # unique, keep order
        if not q_terms:  # question was all stop-words: nothing lexical to add
            return NoOpReranker().rerank(query, candidates)

        docs = [tokenize(c.text) for c in candidates]
        n_docs = len(docs)
        avg_len = (sum(len(d) for d in docs) / n_docs) or 1.0
        # Document frequency measured over the candidate set.
        df = {t: sum(1 for d in docs if t in set(d)) for t in q_terms}
        idf = {t: math.log(1 + (n_docs - df[t] + 0.5) / (df[t] + 0.5)) for t in q_terms}
        q_bigrams = set(zip(q_terms, q_terms[1:]))

        bm25_scores: list[float] = []
        for doc in docs:
            tf = Counter(doc)
            length_norm = 1 - self.b + self.b * (len(doc) / avg_len)
            score = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if f:
                    score += idf[t] * (f * (self.k1 + 1)) / (f + self.k1 * length_norm)
            bm25_scores.append(score)
        max_bm25 = max(bm25_scores) or 1.0

        for cand, doc, bm25 in zip(candidates, docs, bm25_scores):
            doc_set = set(doc)
            coverage = sum(1 for t in q_terms if t in doc_set) / len(q_terms)
            doc_content = [t for t in doc if t not in _STOPWORDS]
            bigram_hits = len(q_bigrams & set(zip(doc_content, doc_content[1:])))
            bigram_score = bigram_hits / len(q_bigrams) if q_bigrams else 0.0

            lexical = 0.5 * (bm25 / max_bm25) + 0.35 * coverage + 0.15 * bigram_score
            vector = max(0.0, min(1.0, cand.score))
            cand.rerank_score = round(self.vector_weight * vector + (1 - self.vector_weight) * lexical, 4)
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)


class CrossEncoderReranker(BaseReranker):
    name = "cross_encoder"

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # heavy, imported lazily

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        raw = self.model.predict([(query, c.text) for c in candidates])
        for cand, logit in zip(candidates, raw):
            cand.rerank_score = round(1 / (1 + math.exp(-float(logit))), 4)  # sigmoid -> 0..1
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)


def build_reranker(settings: Settings) -> BaseReranker:
    choice = settings.reranker
    if choice == "none":
        return NoOpReranker()
    if choice == "cross_encoder":
        try:
            return CrossEncoderReranker(settings.cross_encoder_model)
        except Exception as exc:  # ImportError, download failure, ...
            logger.warning("Cross-encoder unavailable (%s). Falling back to the lexical reranker.", exc)
    return LexicalReranker()
