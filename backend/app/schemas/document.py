from __future__ import annotations

from pydantic import BaseModel


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    uploaded_at: str | None = None


class UploadResponse(BaseModel):
    document: DocumentInfo
    pages: int
    characters: int
    message: str


class DocumentList(BaseModel):
    documents: list[DocumentInfo]
