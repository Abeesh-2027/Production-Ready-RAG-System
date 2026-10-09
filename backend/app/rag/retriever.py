from __future__ import annotations

from app.rag.embedder import BaseEmbedder
from app.rag.vector_store import RetrievedChunk, VectorStore


class Retriever:
    def __init__(self, embedder: BaseEmbedder, store: VectorStore, top_k: int):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str, document_ids: list[str] | None = None) -> list[RetrievedChunk]:
        # Same embedder (=> same model) that embedded the documents at upload time.
        query_vector = self.embedder.embed_query(query)
        return self.store.search(query_vector, top_k=self.top_k, document_ids=document_ids)
