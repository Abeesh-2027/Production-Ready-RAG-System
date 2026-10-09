from dataclasses import replace

import httpx
import pytest

from app.config import get_settings
from app.errors import (
    DocumentNotFoundError,
    EmbeddingError,
    GenerationError,
    MissingApiKeyError,
    UpstreamTimeoutError,
    VectorStoreError,
)
from app.rag.embedder import OpenRouterEmbedder
from app.rag.generator import OpenRouterGenerator
from app.rag.pipeline import RagPipeline
from app.rag.reranker import LexicalReranker
from app.rag.vector_store import VectorStore
from tests.fakes import EchoGenerator, HashEmbedder

DOC = (
    "RAG stands for Retrieval-Augmented Generation. It retrieves relevant passages from documents "
    "and gives them to a language model.\n\n"
    "Embeddings are numerical vectors that capture the meaning of text.\n\n"
    "The cafeteria opens at 8am and serves vegetarian lunch on Fridays."
)


def make_pipeline(tmp_path, generator=None):
    settings = replace(get_settings(), chroma_path=tmp_path / "chroma", upload_path=tmp_path / "up",
                       chunk_size=20, chunk_overlap=5, min_relevance=0.1, openrouter_api_key="test")
    embedder = HashEmbedder()
    store = VectorStore(settings.chroma_path, "documents", embedder.model_name)
    return RagPipeline(settings, embedder, store, LexicalReranker(), generator or EchoGenerator())


def ingest(pipe, tmp_path, name="rag.txt", text=DOC, doc_id="a" * 32):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return pipe.ingest(path, name, doc_id)


def test_ingest_stores_chunks_with_metadata(tmp_path):
    pipe = make_pipeline(tmp_path)
    result = ingest(pipe, tmp_path)
    assert result.document["chunk_count"] >= 2 and result.document["filename"] == "rag.txt"
    hits = pipe.retriever.retrieve("what is RAG")
    meta = hits[0].metadata
    assert {"document_id", "filename", "file_type", "chunk_id", "chunk_index"} <= set(meta)


def test_retrieval_returns_relevant_chunk_first(tmp_path):
    pipe = make_pipeline(tmp_path)
    ingest(pipe, tmp_path)
    hits = pipe.retriever.retrieve("when does the cafeteria open")
    assert "cafeteria" in hits[0].text and hits[0].score > hits[-1].score


def test_chroma_persists_across_restart(tmp_path):
    pipe = make_pipeline(tmp_path)
    ingest(pipe, tmp_path)
    reopened = make_pipeline(tmp_path)  # a "restart": new client on the same folder
    assert [d["filename"] for d in reopened.store.list_documents()] == ["rag.txt"]


def test_delete_document_and_not_found(tmp_path):
    pipe = make_pipeline(tmp_path)
    ingest(pipe, tmp_path)
    assert pipe.store.delete_document("a" * 32) >= 1
    assert pipe.store.count() == 0
    with pytest.raises(DocumentNotFoundError):
        pipe.store.delete_document("a" * 32)


def test_embedding_model_mismatch_is_rejected(tmp_path):
    pipe = make_pipeline(tmp_path)
    ingest(pipe, tmp_path)
    with pytest.raises(VectorStoreError):
        VectorStore(tmp_path / "chroma", "documents", "some-other-model")


def test_full_answer_flow_with_sources(tmp_path):
    gen = EchoGenerator()
    pipe = make_pipeline(tmp_path, gen)
    ingest(pipe, tmp_path)
    result = pipe.ask("What does the document say about the cafeteria?")
    assert "cafeteria" in result.answer
    assert result.sources and result.sources[0]["filename"] == "rag.txt"
    assert {"filename", "page", "chunk_id", "score"} <= set(result.sources[0])
    assert result.trace["retrieved"] and result.trace["reranked"] and result.trace["context"]


def test_unrelated_question_returns_not_found_without_calling_llm(tmp_path):
    gen = EchoGenerator()
    pipe = make_pipeline(tmp_path, gen)
    ingest(pipe, tmp_path)
    result = pipe.ask("zzzz qqqq xxxx")
    assert result.sources == [] and "couldn't find" in result.answer and gen.calls == []


def test_no_documents_message(tmp_path):
    assert "No documents" in make_pipeline(tmp_path).ask("hello").answer


def test_follow_up_uses_history_and_rewriter(tmp_path):
    gen = EchoGenerator()
    pipe = make_pipeline(tmp_path, gen)
    ingest(pipe, tmp_path)
    first = pipe.ask("What is RAG?")
    second = pipe.ask("Why is it useful?", first.conversation_id)
    assert second.trace["history_turns"] == 1
    assert any("Latest question:" in c[-1]["content"] for c in gen.calls)  # rewriter was invoked
    pipe.clear_conversation(first.conversation_id)
    assert pipe.conversations.get_history(first.conversation_id) == []


# ------------------------------------------------------------------ embedder / generator HTTP
class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


def _patch_post(monkeypatch_target, response=None, exc=None):
    def fake_post(*a, **k):
        if exc:
            raise exc
        return response
    return fake_post


def test_embedder_batches_and_orders():
    settings = replace(get_settings(), openrouter_api_key="k", embedding_batch_size=2)
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append(json["input"])
        data = [{"index": i, "embedding": [float(len(t))]} for i, t in reversed(list(enumerate(json["input"])))]
        return _Resp(200, {"data": data})

    original = httpx.post
    httpx.post = fake_post
    try:
        vecs = OpenRouterEmbedder(settings).embed_documents(["a", "bb", "ccc"])
    finally:
        httpx.post = original
    assert calls == [["a", "bb"], ["ccc"]] and vecs == [[1.0], [2.0], [3.0]]


def test_embedder_and_generator_errors():
    no_key = replace(get_settings(), openrouter_api_key="")
    with pytest.raises(MissingApiKeyError):
        OpenRouterEmbedder(no_key).embed_documents(["x"])
    with pytest.raises(MissingApiKeyError):
        OpenRouterGenerator(no_key).complete([{"role": "user", "content": "x"}])

    keyed = replace(get_settings(), openrouter_api_key="k")
    original = httpx.post
    try:
        httpx.post = _patch_post(None, exc=httpx.ReadTimeout("slow"))
        with pytest.raises(UpstreamTimeoutError):
            OpenRouterGenerator(keyed).complete([{"role": "user", "content": "x"}])
        httpx.post = _patch_post(None, response=_Resp(401, {"error": {"message": "bad key"}}))
        with pytest.raises(EmbeddingError):
            OpenRouterEmbedder(keyed).embed_documents(["x"])
        with pytest.raises(GenerationError):
            OpenRouterGenerator(keyed).complete([{"role": "user", "content": "x"}])
        httpx.post = _patch_post(None, response=_Resp(200, {"choices": [{"message": {"content": " hi "}}]}))
        assert OpenRouterGenerator(keyed).complete([{"role": "user", "content": "x"}]) == "hi"
    finally:
        httpx.post = original
