import { useRef, useState } from "react";
import { MAX_UPLOAD_MB, getErrorMessage, uploadDocument } from "../api.js";

const ALLOWED = [".pdf", ".txt", ".docx"];

// Drag-and-drop or click-to-browse upload with progress and status messages.
export default function FileUpload({ onUploaded }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState(null); // { kind: "info"|"success"|"error", text }

  function validate(file) {
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED.includes(ext)) return `"${file.name}" is not supported. Use PDF, DOCX or TXT.`;
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `"${file.name}" is larger than ${MAX_UPLOAD_MB} MB.`;
    if (file.size === 0) return `"${file.name}" is empty.`;
    return null;
  }

  async function handleFiles(fileList) {
    const files = Array.from(fileList || []);
    if (files.length === 0 || busy) return;
    setBusy(true);
    for (const file of files) {
      const problem = validate(file);
      if (problem) {
        setStatus({ kind: "error", text: problem });
        continue;
      }
      setProgress(0);
      setStatus({ kind: "info", text: `Uploading ${file.name}…` });
      try {
        const result = await uploadDocument(file, (pct) => {
          setProgress(pct);
          if (pct >= 100) setStatus({ kind: "info", text: `Indexing ${file.name} (extracting, chunking, embedding)…` });
        });
        setStatus({ kind: "success", text: result.message });
        onUploaded?.(result.document);
      } catch (err) {
        setStatus({ kind: "error", text: getErrorMessage(err) });
      }
    }
    setBusy(false);
    setProgress(0);
    if (inputRef.current) inputRef.current.value = "";
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  }

  const indexing = busy && progress >= 100;

  return (
    <div>
      <div
        className={`dropzone${dragging ? " is-over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <p>Drag a file here, or</p>
        <button className="btn btn-primary" disabled={busy} onClick={() => inputRef.current?.click()}>
          {busy ? "Working…" : "Choose file"}
        </button>
        <p>PDF, DOCX or TXT · up to {MAX_UPLOAD_MB} MB</p>
        <input
          ref={inputRef}
          type="file"
          accept={ALLOWED.join(",")}
          multiple
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {busy && (
        <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}>
          <div className={`progress-bar${indexing ? " indeterminate" : ""}`} style={indexing ? undefined : { width: `${progress}%` }} />
        </div>
      )}
      {status && (
        <div className={`upload-status ${status.kind}`} role="status">
          {status.text}
        </div>
      )}
    </div>
  );
}
