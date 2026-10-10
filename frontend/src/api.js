// All communication with the backend lives here. The browser talks ONLY to FastAPI;
// it never calls OpenRouter and never sees the OpenRouter API key.
import axios from "axios";

export const API_URL = (import.meta.env.VITE_API_URL || "https://production-ready-rag-system.onrender.com").replace(/\/$/, "");
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
    return `Server error (${error.response.status}).`;
  }
  if (error.request) {
    return "Cannot reach the server. Check your connection or try again shortly.";
  }
  return error.message || "Something went wrong.";
}

/** Wakes a sleeping backend (free Render instances sleep when idle). */
export async function checkHealth() {
  const { data } = await client.get("/health", { timeout: 60_000 });
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
