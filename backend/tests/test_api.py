from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.rag.pipeline import RagPipeline, get_pipeline
from app.rag.reranker import LexicalReranker
from app.rag.vector_store import VectorStore
from app.config import get_settings as _gs
from tests.fakes import EchoGenerator, HashEmbedder


def make_client(tmp_path, max_upload_mb=1):
    settings = replace(get_settings(), chroma_path=tmp_path / "chroma", upload_path=tmp_path / "up",
                       min_relevance=0.1, max_upload_mb=max_upload_mb)
    emb = HashEmbedder()
    pipe = RagPipeline(settings, emb, VectorStore(settings.chroma_path, "documents", emb.model_name),
                       LexicalReranker(), EchoGenerator())
    app.dependency_overrides[get_pipeline] = lambda: pipe
    app.dependency_overrides[_gs] = lambda: settings
    return TestClient(app), settings


def test_health(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_chat_list_delete_flow(tmp_path):
    client, settings = make_client(tmp_path)
    r = client.post("/upload", files={"file": ("notes.txt", b"The launch date is March 3rd.\n\nBudget is 40k.", "text/plain")})
    assert r.status_code == 200, r.text
    doc_id = r.json()["document"]["document_id"]
    assert (settings.upload_path / f"{doc_id}.txt").exists()

    chat = client.post("/chat", json={"question": "When is the launch date?"})
    assert chat.status_code == 200 and chat.json()["sources"][0]["filename"] == "notes.txt"
    assert client.get("/documents").json()["documents"][0]["document_id"] == doc_id

    assert client.delete(f"/documents/{doc_id}").status_code == 200
    assert client.get("/documents").json()["documents"] == []
    assert not list(settings.upload_path.glob(f"{doc_id}*"))
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_rejects_bad_input(tmp_path):
    client, settings = make_client(tmp_path)
    assert client.post("/upload", files={"file": ("a.exe", b"MZ", "application/octet-stream")}).status_code == 415
    assert client.post("/upload", files={"file": ("a.pdf", b"not a pdf", "application/pdf")}).status_code == 415
    assert client.post("/upload", files={"file": ("e.txt", b"", "text/plain")}).status_code == 400
    assert client.post("/upload", files={"file": ("big.txt", b"x " * 700_000, "text/plain")}).status_code == 413
    assert client.post("/chat", json={"question": "   "}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422
    assert client.delete("/documents/../../etc/passwd").status_code in (404, 400)
    assert client.delete("/documents/not-a-valid-id").status_code == 400
    assert list(settings.upload_path.glob("*")) == []  # failed uploads leave nothing behind


def test_path_traversal_filename_is_harmless(tmp_path):
    client, settings = make_client(tmp_path)
    r = client.post("/upload", files={"file": ("../../evil.txt", b"Some harmless text here.", "text/plain")})
    assert r.status_code == 200 and r.json()["document"]["filename"] == "evil.txt"
    assert all(p.parent == settings.upload_path for p in settings.upload_path.rglob("*") if p.is_file())
