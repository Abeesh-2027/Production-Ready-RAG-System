# Production-Ready RAG System

Chat with your own documents. Upload a PDF, DOCX or TXT file, ask a question, and get an answer that is **based on your document** and shows **which pages it came from**.

- **Frontend:** React + Vite (deployed on Vercel)
- **Backend:** FastAPI + ChromaDB + OpenRouter (deployed on Render)
- **No LangChain.** Each RAG step is a small, readable Python file.

---

## How it works

```
 Browser (React)  ──►  FastAPI backend  ──►  OpenRouter (LLM + embeddings)
                            │
                            └──►  ChromaDB (stores your document chunks)
```

The browser only talks to your backend. The OpenRouter API key stays on the server and is never sent to the browser.

### When you upload a file

```
File → Extract text → Clean → Split into chunks → Create embeddings → Save in ChromaDB
```

### When you ask a question

```
Question → Rewrite (for follow-ups) → Find top 10 chunks → Rerank → Keep best 5
        → Build prompt → Ask the LLM → Answer + sources
```

| Step | File | What it does |
|---|---|---|
| Load | `rag/loader.py` | Reads PDF, DOCX, TXT into plain text (with page numbers for PDFs) |
| Clean | `rag/cleaner.py` | Removes extra spaces and broken line breaks, keeps paragraphs |
| Chunk | `rag/chunker.py` | Splits text into ~800-word pieces with 150 words of overlap |
| Embed | `rag/embedder.py` | Turns text into vectors using one embedding model for both documents and questions |
| Store | `rag/vector_store.py` | Saves and searches vectors in ChromaDB (data survives restarts) |
| Retrieve | `rag/retriever.py` | Finds the 10 most similar chunks |
| Rerank | `rag/reranker.py` | Re-scores those 10 by keyword match and keeps the best 5 |
| Select | `rag/context.py` | Drops duplicates and low-relevance chunks |
| Rewrite | `rag/query_rewriter.py` | Turns "Why is it useful?" into "Why is RAG useful?" |
| Prompt | `rag/prompt.py` | Tells the LLM to use only the document and to ignore instructions hidden in it |
| Generate | `rag/generator.py` | Calls OpenRouter and handles timeouts and errors |

---

## Project structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app, CORS, /health
│   │   ├── config.py        # all settings (from environment variables)
│   │   ├── routes/          # upload.py, chat.py
│   │   ├── rag/             # the pipeline steps above
│   │   └── schemas/         # request / response models
│   ├── tests/               # per-step tests
│   ├── scripts/demo_flow.py # prints every pipeline step for a sample PDF
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   ├── src/                 # App.jsx, api.js, components/
│   ├── vercel.json
│   └── .env.example
├── render.yaml              # Render deploy settings
└── README.md
```

---

## Run it on your computer

**You need:** Python 3.11 or 3.12, Node 18+, and an [OpenRouter API key](https://openrouter.ai/keys).

### 1. Backend

macOS / Linux:
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Windows (PowerShell):
```powershell
cd backend
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Open `backend/.env`, set `OPENROUTER_API_KEY=...`, then start the server:
```bash
uvicorn app.main:app --reload --port 8000
```
Check http://localhost:8000/health. It should show `{"status":"ok"}`.

### 2. Frontend (new terminal)

```bash
cd frontend
npm install
cp .env.example .env      # Windows: copy .env.example .env
npm run dev
```
Open http://localhost:5173, upload a file, and ask a question.

### 3. Run the tests (optional)

```bash
cd backend
python -m pytest -q
python scripts/demo_flow.py --offline   # no API key needed
python scripts/demo_flow.py             # real run using your OpenRouter key
```

---

## Settings

All backend settings are environment variables (see `backend/.env.example`). The important ones:

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | none | **Required.** Your OpenRouter key |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | The chat model |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | The embedding model |
| `CORS_ORIGINS` | `http://localhost:5173` | Frontend URLs allowed to call the API |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Chunk size and overlap, in words |
| `TOP_K` / `FINAL_K` | `10` / `5` | Chunks retrieved / chunks kept after reranking |
| `MIN_RELEVANCE` | `0.15` | Lower this if you often get "couldn't find anything relevant" |
| `ENABLE_QUERY_REWRITE` | `true` | Rewrite follow-up questions using chat history |
| `MAX_UPLOAD_MB` | `15` | Maximum upload size |

Frontend setting: `VITE_API_URL` is the backend address (for example `https://your-api.onrender.com`).

> If you change `EMBEDDING_MODEL`, delete `backend/data/chroma` and upload your documents again. Vectors from different models can't be compared.

---

## Deploy

### Backend on Render

1. Create a **Web Service** from your GitHub repo.
2. Set **Root Directory** to `backend`.
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Under **Environment**, add:
   - `OPENROUTER_API_KEY` = your key
   - `CORS_ORIGINS` = your Vercel URL, for example `https://my-app.vercel.app` (no trailing slash)
   - `PYTHON_VERSION` = `3.12.7`

> **Free plan warning:** Render's free plan has a temporary disk. Uploaded documents are lost when the service restarts or goes idle, and the first request after idle can take about a minute. Use a paid plan with a persistent disk (see `render.yaml`) to keep documents.

### Frontend on Vercel

1. Import the same repo and set **Root Directory** to `frontend`.
2. Add the environment variable `VITE_API_URL` = your Render URL, starting with `https://` and with no trailing slash.
3. Deploy. If you change `VITE_API_URL` later, **redeploy** because Vite reads it at build time.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Returns `{"status":"ok"}` |
| POST | `/upload` | Upload and index a document (form field `file`) |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{document_id}` | Delete a document and its chunks |
| POST | `/chat` | Ask a question |
| DELETE | `/conversations/{id}` | Clear a conversation's history |

Example chat request and response:
```json
POST /chat
{ "question": "What is RAG?", "conversation_id": "optional" }

{
  "answer": "RAG combines retrieval with generation ... [Source 1]",
  "sources": [
    { "filename": "rag.pdf", "page": 5, "chunk_id": "…_chunk_0004", "score": 0.91 }
  ],
  "conversation_id": "…"
}
```
Errors look like `{"detail": "readable message"}`. Common codes: 413 file too large, 415 unsupported type, 422 empty or unreadable document, 502 OpenRouter error, 503 missing API key, 504 timeout.

Interactive docs are at `/docs` on your backend.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Browser console says **CORS policy / No Access-Control-Allow-Origin** | On Render, set `CORS_ORIGINS` to your exact Vercel URL (no trailing slash, no spaces) and wait for the redeploy |
| Upload shows **Server error (405)** or the Documents list shows an error | `VITE_API_URL` on Vercel is wrong. It must be your full Render URL starting with `https://`. Fix it, then **redeploy** |
| **"Cannot reach the backend"** | Render may be waking up (wait a minute), or `VITE_API_URL` is wrong |
| **"OPENROUTER_API_KEY is not set"** | Add the key to `backend/.env` locally, or to Render's Environment settings |
| Documents disappear on Render | Free plan disk is temporary. Re-upload, or use a paid plan with a disk |
| **"No readable text was found"** | The PDF is a scan (images only). OCR isn't supported |
| Always says it can't find anything relevant | Lower `MIN_RELEVANCE` or raise `TOP_K` |
| Error about the embedding model changing | Restore the old `EMBEDDING_MODEL`, or delete `backend/data/chroma` and re-upload |
| PowerShell won't activate the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |

---

## Security notes

- The API key lives only in server environment variables. `.env` is git-ignored.
- Uploads are limited by size and type, checked by file signature, and saved under random names.
- The prompt tells the model to treat document text as data, not instructions. This reduces prompt-injection risk but can't remove it.
- There is **no login**. Anyone who can reach the API can upload, ask and delete. Add authentication before using private documents.

## Limits

- Scanned PDFs (no text layer) aren't supported.
- Conversation history is kept in server memory and resets on restart.
- The default reranker is keyword-based. For a cross-encoder, install `sentence-transformers` and set `RERANKER=cross_encoder`. It's a large download and too heavy for small free hosts.
