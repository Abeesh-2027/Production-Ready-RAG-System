// Lists indexed documents with a delete button for each.
export default function DocumentList({ documents, loading, onDelete, deletingId }) {
  const documentList = Array.isArray(documents) ? documents : [];

  if (loading) return <p className="empty-note">Loading documents…</p>;

  if (documentList.length === 0) {
    return <p className="empty-note">No documents yet. Upload one to start asking questions.</p>;
  }

  return (
    <ul className="doc-list">
      {documentList.map((doc) => (
        <li className="doc-item" key={doc.document_id}>
          <div className="doc-info">
            <div className="doc-name" title={doc.filename}>
              📄 {doc.filename}
            </div>
            <div className="doc-meta">
              {doc.file_type.toUpperCase()} · {doc.chunk_count} chunks
            </div>
          </div>
          <button
            className="btn btn-danger-text doc-delete"
            disabled={deletingId === doc.document_id}
            onClick={() => onDelete(doc)}
            aria-label={`Delete ${doc.filename}`}
          >
            {deletingId === doc.document_id ? "…" : "Delete"}
          </button>
        </li>
      ))}
    </ul>
  );
}