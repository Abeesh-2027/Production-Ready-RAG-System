# RAG Chatbot

A complete Retrieval-Augmented Generation chatbot: upload PDF / DOCX / TXT files, then ask questions and get answers **grounded in your documents, with sources**.

- **Frontend:** React + Vite + Axios (light theme, Inter font, responsive) → deploy on **Vercel**
- **Backend:** FastAPI + ChromaDB (persistent) + OpenRouter → deploy on **Render**
- **No LangChain.** Every RAG stage is its own small, readable module.

> **Verification status.** The sandbox this was written in blocks PyPI and npm, so FastAPI, ChromaDB, PyMuPDF, httpx, React and Vite could not be installed there. What *was* run: the cleaner, chunker, TXT/DOCX loaders, reranker, context selection and prompt tests (18 tests, all passing) and a Python compile check. What was *not* run: the ChromaDB/pipeline/API tests, the PDF loader, the live OpenRouter calls and `npm run build`. Run the checks in [section 17](#17-verify-your-install) on your machine first.

---

## 1. Architecture

```
┌──────────────┐   HTTPS    ┌──────────────────────────────┐   HTTPS   ┌────────────┐
│ React (Vite) │ ─────────► │ FastAPI                      │ ────────► │ OpenRouter │
│ on Vercel    │ ◄───────── │ on Render                    │ ◄──────── │ LLM + emb. │
└──────────────┘            │  └─ ChromaDB (disk, persistent)           └────────────┘
                            └──────────────────────────────┘
```

The browser talks **only** to FastAPI. The OpenRouter key lives in the backend environment and is never sent to React.

## 2. RAG pipeline

**Ingestion** (on upload) — `routes/upload.py` → `rag/pipeline.py`

```
file → loader → cleaner → chunker → embedder → ChromaDB
```

**Query** (on every question)

```
question → query rewrite → query embedding → retrieval (top 10) → rerank
         → context selection (top ≤5) → prompt → OpenRouter → answer + sources
```

| Stage | File | What it does |
|---|---|---|
| Load | `rag/loader.py` | PDF (PyMuPDF), DOCX (python-docx), TXT → common `LoadedDocument` of `Page`s. Add a format by registering one function in `LOADERS`. |
| Clean | `rag/cleaner.py` | Normalises whitespace/unicode, re-joins hyphenated line breaks, keeps paragraph boundaries. Never lowercases or strips numbers. |
| Chunk | `rag/chunker.py` | Paragraph-aware packing up to `CHUNK_SIZE` words with `CHUNK_OVERLAP` words carried over. Chunks never cross pages, so page numbers stay exact. |
| Embed | `rag/embedder.py` | `BaseEmbedder` + `OpenRouterEmbedder` (batched, retries 429/5xx). One instance embeds documents **and** queries, so the model always matches. |
| Store | `rag/vector_store.py` | Persistent ChromaDB collection `documents`, cosine distance. Add / search / delete / list / get. Refuses to start if `EMBEDDING_MODEL` differs from the one the DB was built with. |
| Retrieve | `rag/retriever.py` | Embeds the question, fetches `TOP_K` candidates with scores. |
| Rerank | `rag/reranker.py` | Second-opinion scoring (see §13). |
| Select | `rag/context.py` | Drops low-relevance + near-duplicate chunks, caps at `FINAL_K` and `MAX_CONTEXT_CHARS`. |
| Rewrite | `rag/query_rewriter.py` | Turns "Why is it useful?" into "Why is RAG useful?" using chat history. Optional. |
| Prompt | `rag/prompt.py` | Grounded system prompt + `[Source n]` blocks + history + question. |
| Generate | `rag/generator.py` | OpenRouter `/chat/completions` with timeout + error mapping. |

## 3. Project structure

```
rag-chatbot-app/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app, CORS, error handlers, /health
│   │   ├── config.py          all settings, from environment variables
│   │   ├── errors.py          error classes that carry HTTP status codes
│   │   ├── routes/            upload.py (upload/list/delete), chat.py
│   │   ├── rag/               the pipeline stages listed above (+ pipeline.py, conversation.py)
│   │   └── schemas/           Pydantic request/response models
│   ├── data/uploads/  data/chroma/
│   ├── scripts/demo_flow.py   end-to-end demo that prints every stage
│   ├── tests/                 per-stage tests + offline test doubles
│   ├── .env.example  .env     (.env is git-ignored)
│   └── requirements.txt
├── frontend/
│   ├── src/ (App.jsx, api.js, components/{Chat,Message,FileUpload,DocumentList,Sources}.jsx)
│   ├── vercel.json  package.json  index.html
├── render.yaml                Render blueprint
└── README.md
```

## 4. Backend installation

Python 3.11 or 3.12 recommended.

**macOS / Linux**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**Windows (PowerShell)**
```powershell
cd backend
py -m venv .venv
.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
copy .env.example .env
```

## 5. Frontend installation

Node 18+.
```bash
cd frontend
npm install
cp .env.example .env        # Windows: copy .env.example .env
```

## 6. `.env` configuration

Edit `backend/.env`. The only required value is the key:

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | – | **Required.** From https://openrouter.ai/keys |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Any OpenRouter chat model id |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Used for documents *and* queries |
| `CHROMA_PATH` / `UPLOAD_PATH` | `./data/chroma` / `./data/uploads` | Storage locations |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | In words |
| `TOP_K` / `FINAL_K` | `10` / `5` | Candidates retrieved / chunks kept |
| `RERANKER` | `lexical` | `lexical`, `cross_encoder`, or `none` |
| `MIN_RELEVANCE` | `0.15` | Chunks scoring lower after reranking are dropped |
| `MAX_CONTEXT_CHARS` | `12000` | Cap on context sent to the LLM |
| `TEMPERATURE` / `MAX_TOKENS` | `0.2` / `1000` | Generation settings |
| `ENABLE_QUERY_REWRITE` | `true` | Rewrite follow-ups using history |
| `MAX_UPLOAD_MB` | `15` | Upload size limit |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated allowed frontend origins |

Frontend `frontend/.env`: `VITE_API_URL` (backend URL). Never put secrets in `VITE_*` variables.

## 7. Run FastAPI

```bash
cd backend
# venv activated
uvicorn app.main:app --reload --port 8000
```
Interactive docs: http://localhost:8000/docs · Health: http://localhost:8000/health

## 8. Run React

```bash
cd frontend
npm run dev
```
Open http://localhost:5173.

## 9. Upload a document

Click **Choose file** (or drag a file) in the sidebar. The status line shows *Uploading → Indexing → Indexed N chunks*. Or with curl:
```bash
curl -F "file=@sample.pdf" http://localhost:8000/upload
```

## 10. How retrieval works

The question is embedded with the same model as the documents; ChromaDB returns the `TOP_K` chunks whose vectors are most similar (cosine). Each hit carries its text, metadata and a similarity score (`1 − distance`). This stage is fast and approximate, so it fetches more than needed (10) and leaves precision to the reranker.

## 11. How ChromaDB works

A `PersistentClient` stores each chunk's id, vector, text and metadata (`document_id, filename, file_type, page_number, chunk_id, chunk_index`) under `CHROMA_PATH`. The collection is opened with `get_or_create_collection`, so restarting FastAPI never wipes data. Deleting a document removes every chunk with that `document_id`.

## 12. How embeddings work

An embedding model maps text to a vector so that similar meaning ⇒ nearby vectors. Chunks are embedded once at upload; the question is embedded at query time with the **same** model. If you change `EMBEDDING_MODEL`, delete `backend/data/chroma` and re-upload — the app refuses to run against a database built with another model rather than return meaningless results.

## 13. How reranking works

Vector search can miss exact keywords and never reads question and chunk together. The default `LexicalReranker` re-scores the 10 candidates from their **text**: BM25 + query-term coverage + bigram (phrase) matches, blended with the vector score. It can and does change the order (there is a test proving it). It is the lightweight fallback: strong on keywords, weaker on pure paraphrase.

For a true cross-encoder: `pip install sentence-transformers` and set `RERANKER=cross_encoder` (large download; too heavy for small free hosts). Other rerankers just subclass `BaseReranker`.

After reranking, **context selection** removes chunks below `MIN_RELEVANCE`, drops near-duplicates (Jaccard ≥ 0.85), and keeps at most `FINAL_K` chunks within `MAX_CONTEXT_CHARS`. If nothing survives, the API answers "I couldn't find anything relevant…" without calling the LLM. If answers say that too often, lower `MIN_RELEVANCE`.

## 14. How OpenRouter is used

`generator.py` POSTs to `{OPENROUTER_BASE_URL}/chat/completions` and `embedder.py` to `/embeddings`, with `Authorization: Bearer $OPENROUTER_API_KEY`. Calls happen server-side only. Timeouts → HTTP 504; auth/credit/rate-limit/model errors → HTTP 502 with a readable message; missing key → HTTP 503. Check that your chosen embedding model is available to your OpenRouter account.

## 15. API documentation

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | – | `{"status":"ok"}` |
| POST | `/upload` | multipart `file` | `{document:{document_id,filename,file_type,chunk_count,uploaded_at}, pages, characters, message}` |
| GET | `/documents` | – | `{documents:[...]}` |
| DELETE | `/documents/{document_id}` | – | `{status, document_id, chunks_removed}` |
| POST | `/chat` | `{"question":"…","conversation_id":"optional","document_ids":["optional"]}` | `{answer, sources:[{filename,page,chunk_id,document_id,score}], conversation_id, rewritten_question}` |
| DELETE | `/conversations/{id}` | – | `{status:"cleared"}` |

Error format: `{"detail": "readable message"}`. Codes: 400 invalid request · 404 unknown document · 413 file too large · 415 unsupported type · 422 empty/unreadable document or invalid input · 502 upstream (OpenRouter/embeddings) error · 503 missing API key · 504 upstream timeout.

Conversation history is kept in server memory (last 6 turns per `conversation_id`); it resets on restart.

## 16. Deployment

**Backend → Render**
1. Push the repo to GitHub. In Render: *New → Blueprint*, pick the repo (uses `render.yaml`), or create a Web Service with root `backend`, build `pip install -r requirements.txt`, start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
2. Set `OPENROUTER_API_KEY` and `CORS_ORIGINS` (your Vercel URL) in the dashboard.
3. **Persistence:** ChromaDB needs a disk. Free web services have an ephemeral filesystem — documents vanish on restart/redeploy/idle spin-down. Use a paid plan + persistent disk (as in `render.yaml`), or accept re-uploading. Free services also sleep after idle; the first request can take ~1 minute (the UI shows a "Connecting…" notice).
4. Memory: free instances have 512 MB. This project embeds via API (no local model), which keeps memory low; avoid `cross_encoder` there.

**Frontend → Vercel**
1. *Add New → Project*, import the repo, set **Root Directory** to `frontend` (Vite is auto-detected).
2. Add environment variable `VITE_API_URL` = your Render URL (no trailing slash), then deploy. Redeploy after changing it (Vite bakes it in at build time).
3. Add the resulting `https://….vercel.app` URL to `CORS_ORIGINS` on Render.

## 17. Verify your install

```bash
cd backend && pip install -r requirements.txt
python -m pytest -q                         # all stage tests, offline (no API key needed)
python scripts/demo_flow.py --offline       # full flow, prints every stage, no network
python scripts/demo_flow.py                 # same flow with real OpenRouter (needs your key)
cd ../frontend && npm install && npm run build
```
The offline mode uses a bag-of-words test embedder and a stub LLM (`tests/fakes.py`) — it proves the wiring, not answer quality. The live run builds `sample.pdf` itself, uploads it, asks *"What does this document say about reranking?"* and prints question → retrieval → reranking → context → prompt → answer → sources.

## 18. Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENROUTER_API_KEY is not set` (503) | Add it to `backend/.env` and restart uvicorn (Render: dashboard env var). |
| Browser: CORS error | Put the exact frontend origin in `CORS_ORIGINS` (scheme + host, no trailing slash) and restart. |
| UI says "Cannot reach the server" | Backend not running, or wrong `VITE_API_URL` (rebuild/redeploy after changing). |
| "database was built with embedding model X" | You changed `EMBEDDING_MODEL`. Revert it, or delete `backend/data/chroma` and re-upload. |
| "No readable text was found" | Scanned PDF (images only). OCR is not supported. |
| Always "couldn't find anything relevant" | Lower `MIN_RELEVANCE`, raise `TOP_K`, or rephrase. |
| Embedding 404/400 from OpenRouter | The model id isn't available to your account; pick another `EMBEDDING_MODEL`. |
| PowerShell won't activate venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Documents disappear on Render | Free plan filesystem is ephemeral — see §16. |
| Large file times out | Raise `REQUEST_TIMEOUT_SECONDS`; keep files under `MAX_UPLOAD_MB`. |

## Security notes

API key only in env vars (`.env` is git-ignored); uploads limited by size and extension, checked by magic bytes, stored under random ids (user filename never touches the filesystem); inputs validated by Pydantic; the prompt tells the model to treat retrieved text as data, not instructions (reduces, but cannot eliminate, prompt-injection risk). There is no authentication: anyone who can reach the API can upload, query and delete documents — add auth before exposing sensitive data.
#   P r o d u c t i o n - R e a d y - R A G - S y s t e m  
 