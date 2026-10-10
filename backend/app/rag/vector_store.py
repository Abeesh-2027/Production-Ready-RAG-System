from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.errors import DocumentNotFoundError, VectorStoreError
from app.rag.chunker import Chunk


@dataclass
class RetrievedChunk:
    """A chunk returned from search, with its scores."""
    chunk_id: str
    text: str
    metadata: dict
    score: float                
    rerank_score: float | None = None  # filled in by the reranker


class VectorStore:
    def __init__(self, path: Path, collection_name: str, embedding_model: str):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not open the vector database: {exc}") from exc

        stored_model = (self._collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != embedding_model and self._collection.count() > 0:
            raise VectorStoreError(
                f"The database was built with embedding model '{stored_model}' but EMBEDDING_MODEL is "
                f"'{embedding_model}'. Restore the old model or delete {path} and re-upload documents."
            )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Number of chunks and embeddings do not match.")
        if not chunks:
            return
        uploaded_at = datetime.now(timezone.utc).isoformat()
        metadatas = [{**c.metadata(), "uploaded_at": uploaded_at} for c in chunks]
        try:
            self._collection.add(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to store chunks in ChromaDB: {exc}") from exc

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks of a document. Returns how many were removed."""
        try:
            existing = self._collection.get(where={"document_id": document_id}, include=[])
            ids = existing["ids"]
            if not ids:
                raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
            self._collection.delete(ids=ids)
            return len(ids)
        except DocumentNotFoundError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete the document: {exc}") from exc

    def search(
        self, query_embedding: list[float], top_k: int, document_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        total = self.count()
        if total == 0:
            return []
        where = None
        if document_ids:
            where = {"document_id": document_ids[0]} if len(document_ids) == 1 else {
                "document_id": {"$in": document_ids}
            }
        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, total),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB search failed: {exc}") from exc

        hits: list[RetrievedChunk] = []
        for chunk_id, text, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(RetrievedChunk(chunk_id=chunk_id, text=text, metadata=meta, score=1.0 - float(dist)))
        return hits

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB error: {exc}") from exc

    def list_documents(self) -> list[dict]:
        """One summary dict per document (aggregated from chunk metadata)."""
        try:
            data = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Failed to list documents: {exc}") from exc
        docs: dict[str, dict] = {}
        for meta in data["metadatas"]:
            doc = docs.setdefault(
                meta["document_id"],
                {
                    "document_id": meta["document_id"],
                    "filename": meta.get("filename", "unknown"),
                    "file_type": meta.get("file_type", ""),
                    "uploaded_at": meta.get("uploaded_at"),
                    "chunk_count": 0,
                },
            )
            doc["chunk_count"] += 1
        return sorted(docs.values(), key=lambda d: d["uploaded_at"] or "", reverse=True)

    def get_document(self, document_id: str) -> dict:
        for doc in self.list_documents():
            if doc["document_id"] == document_id:
                return doc
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
