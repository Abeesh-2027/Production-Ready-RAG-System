from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import Settings, get_settings
from app.errors import EmptyDocumentError
from app.rag.chunker import chunk_document
from app.rag.cleaner import clean_text
from app.rag.context import select_context
from app.rag.conversation import ConversationStore
from app.rag.embedder import BaseEmbedder, build_embedder
from app.rag.generator import OpenRouterGenerator
from app.rag.loader import load_document
from app.rag.prompt import NO_ANSWER_TEXT, NO_DOCUMENTS_TEXT, build_messages
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import BaseReranker, build_reranker
from app.rag.retriever import Retriever
from app.rag.vector_store import RetrievedChunk, VectorStore


@dataclass
class IngestResult:
    document: dict
    pages: int
    characters: int


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict]
    conversation_id: str
    rewritten_question: str
    # Intermediate results of every stage; used by scripts/demo_flow.py and tests.
    trace: dict = field(default_factory=dict)


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedder,
        store: VectorStore,
        reranker: BaseReranker,
        generator: OpenRouterGenerator,
    ):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.generator = generator
        self.retriever = Retriever(embedder, store, settings.top_k)
        self.rewriter = QueryRewriter(generator, settings.enable_query_rewrite)
        self.conversations = ConversationStore(max_turns=settings.max_history_turns)

    # ------------------------------------------------------------ ingestion
    def ingest(self, path: Path, original_filename: str, document_id: str) -> IngestResult:
        s = self.settings
        loaded = load_document(path, original_filename)                       # 1. extract
        cleaned = [(p.page_number, clean_text(p.text)) for p in loaded.pages]  # 2. clean
        cleaned = [(n, t) for n, t in cleaned if t]
        if not cleaned:
            raise EmptyDocumentError(
                "No readable text was found in this document. Scanned (image-only) PDFs need OCR, "
                "which is not supported."
            )
        chunks = chunk_document(loaded, cleaned, document_id, s.chunk_size, s.chunk_overlap)  # 3. chunk
        embeddings = self.embedder.embed_documents([c.text for c in chunks])   # 4. embed
        self.store.add_chunks(chunks, embeddings)                              # 5. store
        return IngestResult(
            document=self.store.get_document(document_id),
            pages=len(loaded.pages),
            characters=sum(len(t) for _, t in cleaned),
        )

    # ------------------------------------------------------------- answering
    def ask(
        self, question: str, conversation_id: str | None = None, document_ids: list[str] | None = None
    ) -> AnswerResult:
        s = self.settings
        conversation_id = conversation_id or uuid.uuid4().hex
        history = self.conversations.get_history(conversation_id)
        trace: dict = {"question": question, "history_turns": len(history) // 2}

        if self.store.count() == 0:
            return AnswerResult(NO_DOCUMENTS_TEXT, [], conversation_id, question, trace)

        rewritten = self.rewriter.rewrite(question, history)                    # query rewriting
        trace["rewritten_question"] = rewritten

        candidates = self.retriever.retrieve(rewritten, document_ids)           # embed query + search
        trace["retrieved"] = [_summary(c) for c in candidates]

        reranked = self.reranker.rerank(rewritten, list(candidates))            # rerank
        trace["reranker"] = self.reranker.name
        trace["reranked"] = [_summary(c) for c in reranked]

        context = select_context(                                               # context selection
            reranked, s.final_k, s.min_relevance, s.duplicate_threshold, s.max_context_chars
        )
        trace["context"] = [_summary(c) for c in context]

        if not context:
            return AnswerResult(NO_ANSWER_TEXT, [], conversation_id, rewritten, trace)

        # Use the ORIGINAL question in the prompt (what the user actually asked);
        # history lets the LLM resolve references on its own too.
        messages = build_messages(question, context, history)                   # prompt
        trace["messages"] = messages
        answer = self.generator.complete(messages)                              # generate
        trace["answer"] = answer

        self.conversations.add_turn(conversation_id, question, answer)
        sources = [_source(c) for c in context]                                 # citations
        return AnswerResult(answer, sources, conversation_id, rewritten, trace)

    def clear_conversation(self, conversation_id: str) -> None:
        self.conversations.clear(conversation_id)


def _summary(chunk: RetrievedChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "page": chunk.metadata.get("page_number"),
        "vector_score": round(chunk.score, 4),
        "rerank_score": chunk.rerank_score,
        "preview": chunk.text[:120].replace("\n", " "),
    }


def _source(chunk: RetrievedChunk) -> dict:
    return {
        "filename": chunk.metadata.get("filename", "unknown"),
        "page": chunk.metadata.get("page_number"),
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.metadata.get("document_id", ""),
        "score": round(chunk.rerank_score if chunk.rerank_score is not None else chunk.score, 3),
    }


@lru_cache
def get_pipeline() -> RagPipeline:
    """Build the pipeline once per process (FastAPI dependency)."""
    settings = get_settings()
    embedder = build_embedder(settings)
    store = VectorStore(settings.chroma_path, settings.collection_name, embedder.model_name)
    return RagPipeline(settings, embedder, store, build_reranker(settings), OpenRouterGenerator(settings))
