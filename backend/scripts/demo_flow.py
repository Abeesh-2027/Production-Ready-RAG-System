from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.rag.embedder import build_embedder  # noqa: E402
from app.rag.generator import OpenRouterGenerator  # noqa: E402
from app.rag.pipeline import RagPipeline  # noqa: E402
from app.rag.reranker import build_reranker  # noqa: E402
from app.rag.vector_store import VectorStore  # noqa: E402

PAGES = [
    "Introduction to RAG\n\nRetrieval-Augmented Generation (RAG) combines a retrieval system with a large language model. "
    "Instead of relying only on what the model memorised during training, RAG first looks up relevant passages in your documents "
    "and gives them to the model as context. This reduces hallucinations and lets answers cite their sources.",
    "Embeddings and Vector Search\n\nAn embedding model converts each chunk of text into a vector. Chunks with similar meaning "
    "have vectors that are close together. A vector database such as ChromaDB stores these vectors and finds the nearest "
    "chunks to a question using cosine similarity.",
    "Reranking\n\nVector search is fast but approximate. A reranking stage re-scores the top candidates by looking at the "
    "question and each chunk together, and keeps only the most relevant five. Reranking improves precision, "
    "especially for exact terms such as names and numbers.",
    "Company Cafeteria\n\nThe cafeteria opens at 8am on weekdays and serves vegetarian lunch on Fridays.",
]


def make_sample_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    for text in PAGES:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(60, 60, 540, 780), text, fontsize=12)
    doc.save(str(path))


def show(title: str, rows: list[dict]) -> None:
    print(f"\n--- {title} ---")
    for r in rows:
        print(f"  p.{r['page']}  vector={r['vector_score']:<7} rerank={r['rerank_score']}  {r['chunk_id']}  | {r['preview']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use offline test doubles instead of OpenRouter")
    ap.add_argument("--question", default="What does this document say about reranking?")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="rag-demo-"))
    settings = replace(get_settings(), chroma_path=tmp / "chroma", upload_path=tmp / "uploads", chunk_size=60, chunk_overlap=10)

    if args.offline:
        from tests.fakes import EchoGenerator, HashEmbedder

        embedder, generator = HashEmbedder(), EchoGenerator()
        print("MODE: OFFLINE (bag-of-words embedder + stub LLM; proves wiring, not answer quality)")
    else:
        embedder, generator = build_embedder(settings), OpenRouterGenerator(settings)
        print(f"MODE: LIVE  embeddings={settings.embedding_model}  llm={settings.openrouter_model}")

    store = VectorStore(settings.chroma_path, settings.collection_name, embedder.model_name)
    pipe = RagPipeline(settings, embedder, store, build_reranker(settings), generator)

    pdf = tmp / "sample.pdf"
    make_sample_pdf(pdf)
    print(f"\n[1] UPLOAD  {pdf.name}")
    result = pipe.ingest(pdf, "sample.pdf", "d" * 32)
    print(f"    pages={result.pages} characters={result.characters} chunks={result.document['chunk_count']}")

    print(f"\n[2] QUESTION  {args.question!r}")
    answer = pipe.ask(args.question)
    t = answer.trace
    print(f"\n[3] QUERY REWRITE  -> {t['rewritten_question']!r}  (no history, so unchanged)")
    show("[4] RETRIEVAL: top-K by vector similarity", t["retrieved"])
    show(f"[5] RERANKING ({t['reranker']}): new order", t["reranked"])
    show("[6] CONTEXT SELECTION: chunks sent to the LLM", t["context"])
    print("\n[7] PROMPT sent to the LLM (system prompt omitted):\n")
    print("    " + t["messages"][1]["content"][:700].replace("\n", "\n    ") + "\n    ...")
    print(f"\n[8] ANSWER\n    {answer.answer}")
    print("\n[9] SOURCES")
    for s in answer.sources:
        print(f"    📄 {s['filename']} — Page {s['page']}  (score {s['score']}, {s['chunk_id']})")


if __name__ == "__main__":
    main()
