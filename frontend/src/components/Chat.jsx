import { useEffect, useRef, useState } from "react";
import Message from "./Message.jsx";

const SUGGESTIONS = ["Summarize this document", "What are the key points?"];

export default function Chat({ messages, loading, hasDocuments, serverStatus, onSend, onRetry }) {
  const [input, setInput] = useState("");
  const endRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  function submit(text) {
    const question = (text ?? input).trim();
    if (!question || loading) return;
    onSend(question);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }

  function onKeyDown(e) {

    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  }

  function autoGrow(e) {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
  }

  const lastIndex = messages.length - 1;

  return (
    <section className="chat" aria-label="Chat">
      <div className="messages" aria-live="polite">
        <div className="messages-inner">
          {messages.length === 0 && (
            <div className="empty-state">
              <h2 className="section-heading">Ask your documents</h2>
              <p>
                {hasDocuments
                  ? "Ask a question and the answer will be grounded in your uploaded documents, with sources."
                  : "Upload a PDF, DOCX or TXT file in the sidebar, then ask questions about it."}
              </p>
              {hasDocuments && (
                <div className="suggestions">
                  {SUGGESTIONS.map((s) => (
                    <button key={s} className="btn" onClick={() => submit(s)} disabled={loading}>
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {messages.map((m, i) => (
            <Message key={m.id} message={m} onRetry={m.error && i === lastIndex ? onRetry : undefined} />
          ))}

          {loading && (
            <div className="message assistant" aria-label="Assistant is typing">
              <div className="typing">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>
      </div>

      <div className="composer">
        {serverStatus === "waking" && (
          <div className="status-line" role="status">
            Connecting to the server… the first request can take up to a minute on free hosting.
          </div>
        )}
        <div className="composer-inner">
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            placeholder="Ask a question about your documents…"
            aria-label="Your question"
            onChange={autoGrow}
            onKeyDown={onKeyDown}
            maxLength={4000}
          />
          <button className="btn btn-primary" onClick={() => submit()} disabled={loading || !input.trim()}>
            Send
          </button>
        </div>
      </div>
    </section>
  );
}
