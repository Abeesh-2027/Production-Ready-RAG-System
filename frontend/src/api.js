// All communication with the backend lives here. The browser talks ONLY to FastAPI;
// it never calls OpenRouter and never sees the OpenRouter API key.
import axios from "axios";

// Make sure the base URL is absolute. A value like "my-api.onrender.com" (no https://)
// would be treated as a RELATIVE path, so requests would go to the Vercel site itself
// instead of the backend - which answers GET with index.html and POST with HTTP 405.
function normalizeBaseUrl(raw) {
  let url = (raw || "").trim().replace(/\/+$/, "");
  if (url && !/^https?:\/\//i.test(url)) {
    url = (/^(localhost|127\.0\.0\.1)/.test(url) ? "http://" : "https://") + url;
  }
  return url;
}

export const API_URL = normalizeBaseUrl(
  import.meta.env.VITE_API_URL || "https://production-ready-rag-system.onrender.com"
);

export const MAX_UPLOAD_MB = Number(import.meta.env.VITE_MAX_UPLOAD_MB || 15);

const client = axios.create({ baseURL: API_URL, timeout: 90_000 });

/** Turn any axios error into a short, human-readable message. */
export function getErrorMessage(error) {
  if (error.code === "ECONNABORTED") {
    return "The request timed out. The server may be waking up - please try again in a moment.";
  }
  if (error.response) {
    const detail = error.response.data?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg;
    const status = error.response.status;
    const where = `${error.config?.baseURL || ""}${error.config?.url || ""}`;
    if (status === 405 || status === 404) {
      // Non-JSON 404/405 almost always means the request hit the wrong server.
      return `Server error (${status}) from ${where}. The frontend may be pointing at the wrong backend - check VITE_API_URL.`;
    }
    return `Server error (${status}).`;
  }
  if (error.request) {
    return "Cannot reach the server. Check your connection or try again shortly.";
  }
  return error.message || "Something went wrong.";
}

/** Wakes a sleeping backend (free Render instances sleep when idle). */
export async function checkHealth() {
  const { data } = await client.get("/health", { timeout: 60_000 });
  // A static host (e.g. Vercel) answers with HTML, not {"status":"ok"}.
  if (data?.status !== "ok") {
    throw new Error(`${API_URL} is not the RAG backend (unexpected /health response).`);
  }
  return data;
}

export async function uploadDocument(file, onProgress) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await client.post("/upload", form, {
    timeout: 180_000, // indexing a large PDF can take a while
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100));
    },
  });
  return data;
}

export async function listDocuments() {
  const { data } = await client.get("/documents");
  if (!data || !Array.isArray(data.documents)) {
    throw new Error(`${API_URL} did not return a document list - is it the RAG backend?`);
  }
  return data.documents;
}

export async function deleteDocument(documentId) {
  const { data } = await client.delete(`/documents/${documentId}`);
  return data;
}

export async function sendChat(question, conversationId) {
  const { data } = await client.post("/chat", {
    question,
    conversation_id: conversationId || null,
  });
  return data;
}

export async function clearConversation(conversationId) {
  if (!conversationId) return;
  await client.delete(`/conversations/${encodeURIComponent(conversationId)}`);
}