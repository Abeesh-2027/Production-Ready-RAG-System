import { useCallback, useEffect, useState } from "react";
import {
  checkHealth,
  clearConversation,
  deleteDocument,
  getErrorMessage,
  listDocuments,
  sendChat,
} from "./api.js";
import Chat from "./components/Chat.jsx";
import DocumentList from "./components/DocumentList.jsx";
import FileUpload from "./components/FileUpload.jsx";

let nextId = 1;
const newId = () => `m${nextId++}`;

export default function App() {
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [docsLoading, setDocsLoading] = useState(true);
  const [deletingId, setDeletingId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [serverStatus, setServerStatus] = useState("waking"); // "waking" | "ready" | "down"
  const [globalError, setGlobalError] = useState(null);

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
      setGlobalError(null);
    } catch (err) {
      setGlobalError(getErrorMessage(err));
    } finally {
      setDocsLoading(false);
    }
  }, []);

  // On first load: wake the backend (free Render instances sleep), then fetch documents.
  useEffect(() => {
    checkHealth()
      .then(() => setServerStatus("ready"))
      .catch(() => setServerStatus("down"))
      .finally(refreshDocuments);
  }, [refreshDocuments]);

  async function ask(question) {
    setLoading(true);
    try {
      const data = await sendChat(question, conversationId);
      setConversationId(data.conversation_id);
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "assistant", content: data.answer, sources: data.sources },
      ]);
      setServerStatus("ready");
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "assistant", content: getErrorMessage(err), error: true, question },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleSend(question) {
    setMessages((prev) => [...prev, { id: newId(), role: "user", content: question }]);
    ask(question);
  }

  function handleRetry() {
    const last = messages[messages.length - 1];
    if (!last?.error) return;
    setMessages((prev) => prev.slice(0, -1)); // drop the error bubble, keep the user's question
    ask(last.question);
  }

  async function handleClear() {
    const id = conversationId;
    setMessages([]);
    if (id) clearConversation(id).catch(() => {}); // best effort; UI is already cleared
  }

  function handleNew() {
    handleClear();
    setConversationId(null);
  }

  async function handleDelete(doc) {
    if (!window.confirm(`Delete "${doc.filename}" and all its indexed chunks?`)) return;
    setDeletingId(doc.document_id);
    try {
      await deleteDocument(doc.document_id);
      setDocuments((prev) => prev.filter((d) => d.document_id !== doc.document_id));
    } catch (err) {
      setGlobalError(getErrorMessage(err));
    } finally {
      setDeletingId(null);
    }
  }

  function handleUploaded() {
    refreshDocuments();
    setSidebarOpen(false);
  }

  const hasMessages = messages.length > 0;

  return (
    <div className="app">
      <header className="header">
        <button
          className="btn menu-btn"
          onClick={() => setSidebarOpen(true)}
          aria-label="Open documents panel"
        >
          ☰
        </button>
        <div className="header-title">
          <h1 className="title">RAG Chatbot</h1>
          <p className="subtitle">Ask questions about your own documents</p>
        </div>
        <div className="header-actions">
          <button className="btn" onClick={handleClear} disabled={!hasMessages || loading} aria-label="Clear conversation">
            <span aria-hidden="true">✕</span>
            <span className="btn-label">Clear</span>
          </button>
          <button className="btn btn-primary" onClick={handleNew} disabled={loading} aria-label="New conversation">
            <span aria-hidden="true">＋</span>
            <span className="btn-label">New conversation</span>
          </button>
        </div>
      </header>

      <div className="layout">
        <div
          className={`sidebar-backdrop${sidebarOpen ? " open" : ""}`}
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
        <aside className={`sidebar${sidebarOpen ? " open" : ""}`} aria-label="Documents">
          <div>
            <h2 className="section-heading">Upload</h2>
            <p className="section-hint">Documents are indexed so the chatbot can search them.</p>
            <FileUpload onUploaded={handleUploaded} />
          </div>
          <div>
            <h2 className="section-heading">Documents</h2>
            {serverStatus === "down" && (
              <div className="upload-status error" role="alert">
                The server is not responding. Try again in a moment.
              </div>
            )}
            {globalError && serverStatus !== "down" && (
              <div className="upload-status error" role="alert">
                {globalError}
              </div>
            )}
            <DocumentList
              documents={documents}
              loading={docsLoading}
              onDelete={handleDelete}
              deletingId={deletingId}
            />
          </div>
          <button className="btn menu-btn" onClick={() => setSidebarOpen(false)}>
            Close
          </button>
        </aside>

        <Chat
          messages={messages}
          loading={loading}
          hasDocuments={documents.length > 0}
          serverStatus={serverStatus}
          onSend={handleSend}
          onRetry={handleRetry}
        />
      </div>
    </div>
  );
}
