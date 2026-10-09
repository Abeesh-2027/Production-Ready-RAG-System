// Shows where an answer came from: "[1] 📄 rag.pdf — Page 5".
// The numbers match the [Source n] citations the model writes in its answer.
export default function Sources({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="sources" aria-label="Sources">
      <div className="sources-title">Sources</div>
      <ul>
        {sources.map((s, i) => (
          <li key={s.chunk_id}>
            <span className="source-n">[{i + 1}]</span>
            <span>
              📄 {s.filename}
              {s.page != null ? ` — Page ${s.page}` : ""}
            </span>
            <span className="source-score">relevance {Math.round(s.score * 100)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
