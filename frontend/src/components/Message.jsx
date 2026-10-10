import Sources from "./Sources.jsx";
export default function Message({ message, onRetry }) {
  const { role, content, sources, error } = message;
  return (
    <div className={`message ${role}${error ? " error" : ""}`}>
      <div className="bubble">{content}</div>
      {role === "assistant" && !error && <Sources sources={sources} />}
      {error && onRetry && (
        <button className="btn retry" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}
