from __future__ import annotations

import hashlib
import math
import re

from app.rag.embedder import BaseEmbedder

_WORD = re.compile(r"[a-z0-9]+")
_DIM = 512


class HashEmbedder(BaseEmbedder):
    model_name = "test-hash-embedder"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    @staticmethod
    def _embed(text: str) -> list[float]:
        vec = [0.0] * _DIM
        for word in _WORD.findall(text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % _DIM
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class EchoGenerator:
    """Stands in for OpenRouter: returns the first lines of the provided context."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def complete(self, messages, temperature=None, max_tokens=None) -> str:
        self.calls.append(messages)
        user = messages[-1]["content"]
        if "Latest question:" in user:  # query-rewrite call: return the question unchanged
            return user.rsplit("Latest question:", 1)[1].strip()
        body = user.split("Content:", 1)[1].split("[Source", 1)[0].split("=== END", 1)[0].strip()
        return f"(offline stub answer) {body[:300]} [Source 1]"
