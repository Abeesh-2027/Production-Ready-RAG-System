from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.errors import (
    DocumentExtractionError,
    FileTooLargeError,
    InvalidRequestError,
    UnsupportedFileTypeError,
)
from app.rag.loader import supported_extensions
from app.rag.pipeline import RagPipeline, get_pipeline
from app.schemas.document import DocumentInfo, DocumentList, UploadResponse

router = APIRouter(tags=["documents"])

_READ_CHUNK = 1024 * 1024  # read uploads 1 MB at a time so the size limit is enforced early
_DOC_ID = re.compile(r"^[a-f0-9]{32}$")


def sanitize_filename(name: str) -> str:
    """Display name only (never used as a disk path). Strips directories and odd chars."""
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip().strip(".")
    return name[:150] or "document"


def _check_magic_bytes(ext: str, head: bytes) -> None:
    """Cheap content check so a renamed .exe can't pass as .pdf."""
    if ext == ".pdf" and not head.lstrip().startswith(b"%PDF"):
        raise UnsupportedFileTypeError("This file has a .pdf extension but is not a valid PDF.")
    if ext == ".docx" and not head.startswith(b"PK"):
        raise UnsupportedFileTypeError("This file has a .docx extension but is not a valid DOCX file.")
    if ext == ".txt" and b"\x00" in head and not head.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise UnsupportedFileTypeError("This file does not look like a text file.")


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    pipeline: RagPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    display_name = sanitize_filename(file.filename or "")
    ext = Path(display_name).suffix.lower()
    if ext not in supported_extensions():
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext or 'unknown'}'. Allowed: {', '.join(supported_extensions())}."
        )

    # Safe on-disk name: a random id + the validated extension. The user-supplied
    # filename never touches the filesystem, so path traversal is impossible.
    document_id = uuid.uuid4().hex
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    saved_path = settings.upload_path / f"{document_id}{ext}"

    size = 0
    try:
        with saved_path.open("wb") as out:
            first = True
            while True:
                block = await file.read(_READ_CHUNK)
                if not block:
                    break
                if first:
                    _check_magic_bytes(ext, block[:1024])
                    first = False
                size += len(block)
                if size > settings.max_upload_bytes:
                    raise FileTooLargeError(f"File is too large. The limit is {settings.max_upload_mb} MB.")
                out.write(block)
        if size == 0:
            raise InvalidRequestError("The uploaded file is empty.")
        # Indexing is blocking work (parsing + HTTP calls + disk). Run it in a worker
        # thread so the event loop keeps serving /health and other requests meanwhile.
        result = await run_in_threadpool(pipeline.ingest, saved_path, display_name, document_id)
    except Exception:
        saved_path.unlink(missing_ok=True)  # never keep a file that failed to index
        raise

    return UploadResponse(
        document=DocumentInfo(**result.document),
        pages=result.pages,
        characters=result.characters,
        message=f"Indexed {result.document['chunk_count']} chunks from {display_name}.",
    )


@router.get("/documents", response_model=DocumentList)
def list_documents(pipeline: RagPipeline = Depends(get_pipeline)) -> DocumentList:
    return DocumentList(documents=[DocumentInfo(**d) for d in pipeline.store.list_documents()])


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    pipeline: RagPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> dict:
    if not _DOC_ID.match(document_id):
        raise InvalidRequestError("Invalid document id.")
    removed = pipeline.store.delete_document(document_id)
    for path in settings.upload_path.glob(f"{document_id}.*"):  # id is validated hex: safe glob
        path.unlink(missing_ok=True)
    return {"status": "deleted", "document_id": document_id, "chunks_removed": removed}
